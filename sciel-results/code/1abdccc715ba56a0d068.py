"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, LinearConstraint

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s out of 150s budget, leaving 40s safety margin

    # Trivial seed: a grid of equal circles, first n used.
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    seed_arr = np.array(cells[:N])
    best_result = seed_arr
    best_value = np.sum(seed_arr[:, 2])

    def _compute_value(positions):
        """Compute sum of radii given positions (n,2)."""
        n = len(positions)
        # Radii are determined by distances to walls and to other circles
        # For each circle i, radius is limited by:
        #   - distance to nearest wall: min(x_i, 1-x_i, y_i, 1-y_i)
        #   - half the distance to nearest other circle j: (dist_ij - r_j)/2 is not quite right
        # Actually, we need to find radii r_i such that:
        #   r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
        #   r_i + r_j <= dist(i,j) for all i != j
        #   r_i >= 0
        # We want to maximize sum(r_i).
        # This is a linear program! Let's solve it.
        from scipy.optimize import linprog
        # Variables: r_0, ..., r_{n-1}
        # Objective: maximize sum(r_i) -> minimize -sum(r_i)
        c = -np.ones(n)
        
        # Constraints:
        # 1) r_i >= 0 (handled by bounds)
        # 2) r_i <= min(x_i, 1-x_i, y_i, 1-y_i) for each i
        #    This is n inequalities: r_i - min_i <= 0
        #    But also need r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
        #    That's 4n inequalities.
        # 3) r_i + r_j <= dist(i,j) for all i < j
        #    That's C(n,2) inequalities.
        
        n_pairs = n * (n - 1) // 2
        n_wall = 4 * n
        n_ineq = n_wall + n_pairs
        
        # Build A_ub and b_ub such that A_ub @ r <= b_ub
        A_rows = []
        b_rows = []
        
        # Wall constraints
        for i in range(n):
            x_i, y_i = positions[i]
            # r_i <= x_i
            row = np.zeros(n)
            row[i] = 1.0
            A_rows.append(row)
            b_rows.append(x_i)
            # r_i <= 1 - x_i
            row = np.zeros(n)
            row[i] = 1.0
            A_rows.append(row)
            b_rows.append(1.0 - x_i)
            # r_i <= y_i
            row = np.zeros(n)
            row[i] = 1.0
            A_rows.append(row)
            b_rows.append(y_i)
            # r_i <= 1 - y_i
            row = np.zeros(n)
            row[i] = 1.0
            A_rows.append(row)
            b_rows.append(1.0 - y_i)
        
        # Pairwise constraints
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt((positions[i][0] - positions[j][0])**2 + (positions[i][1] - positions[j][1])**2)
                row = np.zeros(n)
                row[i] = 1.0
                row[j] = 1.0
                A_rows.append(row)
                b_rows.append(d)
        
        A_ub = np.array(A_rows)
        b_ub = np.array(b_rows)
        
        bounds = [(0, None)] * n
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                return -res.fun
            else:
                return 0.0
        except:
            return 0.0

    def _objective(positions_flat):
        """Objective for differential evolution: maximize sum of radii."""
        if time.time() > deadline:
            return 0.0
        positions = positions_flat.reshape(-1, 2)
        val = _compute_value(positions)
        return -val  # minimize negative

    # Define bounds for positions: each coordinate in [0, 1]
    bounds = [(0.0, 1.0)] * (2 * N)

    # Run differential evolution
    try:
        de_result = differential_evolution(
            _objective,
            bounds=bounds,
            seed=42,
            maxiter=200,
            popsize=15,
            tol=1e-6,
            polish=True,
            init='sobol',
            updating='deferred',
            callback=lambda x, convergence: (None if time.time() <= deadline else None)
        )
        
        if de_result.success and de_result.fun < 0:
            best_positions = de_result.x.reshape(-1, 2)
            best_radii = np.zeros(N)
            
            # Recompute radii via LP for the best positions
            from scipy.optimize import linprog
            n = N
            c = -np.ones(n)
            A_rows = []
            b_rows = []
            
            for i in range(n):
                x_i, y_i = best_positions[i]
                row = np.zeros(n); row[i] = 1.0
                A_rows.append(row); b_rows.append(x_i)
                row = np.zeros(n); row[i] = 1.0
                A_rows.append(row); b_rows.append(1.0 - x_i)
                row = np.zeros(n); row[i] = 1.0
                A_rows.append(row); b_rows.append(y_i)
                row = np.zeros(n); row[i] = 1.0
                A_rows.append(row); b_rows.append(1.0 - y_i)
            
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.sqrt((best_positions[i][0] - best_positions[j][0])**2 + (best_positions[i][1] - best_positions[j][1])**2)
                    row = np.zeros(n); row[i] = 1.0; row[j] = 1.0
                    A_rows.append(row); b_rows.append(d)
            
            A_ub = np.array(A_rows)
            b_ub = np.array(b_rows)
            bounds_lp = [(0, None)] * n
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method='highs')
            if res.success:
                best_radii = res.x
                val = np.sum(best_radii)
                if val > best_value:
                    result = np.column_stack([best_positions, best_radii])
                    best_result = result
                    best_value = val
    except Exception:
        pass

    # Verify the result is valid
    if len(best_result) != N:
        return seed_arr
    
    # Ensure all radii are non-negative and circles are inside
    radii = best_result[:, 2]
    if np.any(radii < -1e-10):
        return seed_arr
    
    # Check disjointness and containment
    valid = True
    for i in range(N):
        x, y, r = best_result[i]
        if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
            valid = False
            break
        if r < -1e-9:
            valid = False
            break
    if valid:
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((best_result[i, 0] - best_result[j, 0])**2 + (best_result[i, 1] - best_result[j, 1])**2)
                if d < best_result[i, 2] + best_result[j, 2] - 1e-9:
                    valid = False
                    break
            if not valid:
                break
    
    if not valid:
        return seed_arr
    
    return best_result

# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment

def construct():
    # We want to pack 21 circles in a rectangle with perimeter 4.
    # Perimeter = 2*(w+h) = 4 => w+h = 2.
    # We maximize sum of radii.
    
    best_sum = -1
    best_result = None
    
    # Try different aspect ratios
    # w + h = 2
    # w in [0.5, 1.5]
    for w in np.linspace(0.5, 1.5, 21):
        h = 2.0 - w
        
        # 1. Generate Grid
        # High resolution grid for initial matching
        grid_res = 40
        xs = np.linspace(0, w, grid_res)
        ys = np.linspace(0, h, grid_res)
        X, Y = np.meshgrid(xs, ys)
        X = X.flatten()
        Y = Y.flatten()
        
        # Calculate distance to boundary for each grid point
        # d_bound = min(x, w-x, y, h-y)
        d_bound = np.minimum(np.minimum(X, w - X), np.minimum(Y, h - Y))
        
        # Filter out points too close to boundary (optional, but helps)
        # Actually, we want points that CAN hold a circle.
        # For now, keep all, but cost will penalize small d_bound.
        
        # 2. Hungarian Algorithm to select 21 points
        # We want to maximize the sum of d_bound?
        # This will pick the 21 points closest to the center.
        # This creates a cluster.
        # To spread them, we might need a different cost.
        # However, the local search will spread them.
        # The Hungarian step ensures we start with "good" boundary-distance points.
        
        # Cost matrix: -d_bound
        # We have grid_res*grid_res points. We need to pick N=21.
        # linear_sum_assignment expects a square matrix or rectangular.
        # We can create a dummy assignment: 21 circles to M cells.
        # Cost[i, j] = -d_bound[j] for all i?
        # This will assign all 21 circles to the same best cell?
        # No, linear_sum_assignment enforces unique assignment.
        # So it will pick the top 21 cells with highest d_bound.
        
        # Create cost matrix: N x M
        # Since all circles are identical in this step, we can just take the top N indices.
        # But to follow the directive "Hungarian", we use the solver.
        
        M = len(X)
        # If M < N, skip (should not happen with res 40)
        if M < N:
            continue
            
        # Cost matrix for Hungarian
        # We want to minimize sum of costs.
        # Cost = -d_bound
        # We have N rows (circles) and M cols (cells).
        # To use linear_sum_assignment, we can pad or just use the fact that it finds min cost assignment.
        # Since cost is independent of row i, it will just pick the N cells with min cost (max d_bound).
        
        # Construct a large cost matrix
        # Note: linear_sum_assignment is efficient for sparse or structured matrices.
        # Here it's dense.
        
        # Optimization: Just sort and take top N?
        # The directive says "using the Hungarian algorithm".
        # So we must use it.
        
        # Create N x M matrix
        # This might be memory heavy if M is large.
        # M = 40*40 = 1600. N=21.
        # 21 * 1600 = 33600 floats. Fine.
        
        cost_matrix = -d_bound[None, :].repeat(N, axis=0)
        
        # Solve
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # The selected cells are col_ind
        # Note: row_ind will be 0..20, col_ind will be the indices of the selected cells.
        
        init_xs = X[col_ind]
        init_ys = Y[col_ind]
        
        # 3. Local Search
        # Positions P (N, 2)
        P = np.column_stack((init_xs, init_ys))
        
        # Function to solve LP for radii given positions
        def solve_radii(positions, w, h):
            n = len(positions)
            # Variables: r_0 ... r_{n-1}
            # Maximize sum(r) -> Minimize -sum(r)
            c = -np.ones(n)
            
            A_ub = []
            b_ub = []
            
            # Pairwise constraints: r_i + r_j <= d_ij
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.linalg.norm(positions[i] - positions[j])
                    # r_i + r_j <= d
                    row = np.zeros(n)
                    row[i] = 1
                    row[j] = 1
                    A_ub.append(row)
                    b_ub.append(d)
            
            # Boundary constraints: r_i <= min(x_i, w-x_i, y_i, h-y_i)
            for i in range(n):
                db = min(positions[i][0], w - positions[i][0], positions[i][1], h - positions[i][1])
                # r_i <= db
                row = np.zeros(n)
                row[i] = 1
                A_ub.append(row)
                b_ub.append(db)
            
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            
            # Bounds: r_i >= 0
            bounds = [(0, None)] * n
            
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if res.success:
                return res.x, -res.fun
            else:
                # Fallback: uniform radius based on min distance
                # This shouldn't happen if positions are valid
                return np.zeros(n), 0.0
        
        # Initial LP solve
        radii, current_sum = solve_radii(P, w, h)
        
        # Local Search Loop
        # Use random perturbations
        np.random.seed(42) # For reproducibility in debugging, but maybe random is better?
        # Let's use a fixed seed for deterministic behavior in evaluation?
        # The evaluator might run multiple times. Randomness is okay.
        
        for _ in range(300):
            # Pick a random circle
            k = np.random.randint(N)
            
            # Save old position
            old_x, old_y = P[k]
            
            # Perturb
            # Step size depends on current radius?
            step = 0.05
            new_x = old_x + np.random.uniform(-step, step)
            new_y = old_y + np.random.uniform(-step, step)
            
            # Check boundaries
            if new_x < 0 or new_x > w or new_y < 0 or new_y > h:
                continue
                
            P[k] = [new_x, new_y]
            
            # Solve LP
            new_radii, new_sum = solve_radii(P, w, h)
            
            if new_sum > current_sum:
                current_sum = new_sum
                radii = new_radii
                # Accept
            else:
                # Revert
                P[k] = [old_x, old_y]
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_result = np.column_stack((P, radii))
            
    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END

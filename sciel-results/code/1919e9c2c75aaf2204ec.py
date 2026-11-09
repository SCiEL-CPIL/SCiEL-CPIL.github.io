# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

def squares_in_square10() -> tuple:
    """
    Pack 10 axis-aligned squares in [0,1]^2 maximizing sum of side lengths.
    
    Strategy:
    1. Define a separation pattern: for each pair (i,j), decide if they are separated
       in x (left/right) or y (up/down).
    2. Use SLSQP to find the optimal sizes/positions for a given pattern.
    3. Use Simulated Annealing to explore different separation patterns,
       keeping the best feasible solution found.
    """
    t0 = time.time()
    deadline = t0 + 580.0  # safety margin before 630s kill

    n = 10
    tol = 1e-7
    pairs = list(combinations(range(n), 2))
    n_pairs = len(pairs) # 45

    def is_valid(centers, sides):
        if centers.shape != (n, 2) or sides.shape != (n,):
            return False
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if np.any(sides <= 0):
            return False
        # boundary
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s/2 < -tol or cx + s/2 > 1 + tol:
                return False
            if cy - s/2 < -tol or cy + s/2 > 1 + tol:
                return False
        # overlap
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                if abs(centers[i,0] - centers[j,0]) < t and abs(centers[i,1] - centers[j,1]) < t:
                    return False
        return True

    def objective(x):
        """x: [s_0, ..., s_9, cx_0, ..., cx_9, cy_0, ..., cy_9]"""
        s = x[:n]
        cx = x[n:2*n]
        cy = x[2*n:3*n]
        return -np.sum(s)

    def constraints_factory(separation_pattern):
        """
        separation_pattern: array of length n_pairs.
        0: separated in x (i left of j)
        1: separated in y (i below j)
        """
        cons = []
        
        # Boundary constraints:
        # cx_i - s_i/2 >= 0  =>  cx_i - s_i/2 >= 0
        # cx_i + s_i/2 <= 1  =>  1 - cx_i - s_i/2 >= 0
        # cy_i - s_i/2 >= 0  =>  cy_i - s_i/2 >= 0
        # cy_i + s_i/2 <= 1  =>  1 - cy_i - s_i/2 >= 0
        for i in range(n):
            # x_low
            def x_low(x, i=i):
                return x[n+i] - x[i]/2
            cons.append({'type': 'ineq', 'fun': x_low})
            
            # x_high
            def x_high(x, i=i):
                return 1 - x[n+i] - x[i]/2
            cons.append({'type': 'ineq', 'fun': x_high})
            
            # y_low
            def y_low(x, i=i):
                return x[2*n+i] - x[i]/2
            cons.append({'type': 'ineq', 'fun': y_low})
            
            # y_high
            def y_high(x, i=i):
                return 1 - x[2*n+i] - x[i]/2
            cons.append({'type': 'ineq', 'fun': y_high})

        # Separation constraints
        for k, (i, j) in enumerate(pairs):
            if separation_pattern[k] == 0:
                # i is left of j: cx_j - cx_i >= (s_i + s_j)/2
                # cx_j - cx_i - (s_i + s_j)/2 >= 0
                def sep_x(x, i=i, j=j):
                    return x[n+j] - x[n+i] - (x[i] + x[j])/2
                cons.append({'type': 'ineq', 'fun': sep_x})
            else:
                # i is below j: cy_j - cy_i >= (s_i + s_j)/2
                # cy_j - cy_i - (s_i + s_j)/2 >= 0
                def sep_y(x, i=i, j=j):
                    return x[2*n+j] - x[2*n+i] - (x[i] + x[j])/2
                cons.append({'type': 'ineq', 'fun': sep_y})
        
        return cons

    def solve_pattern(separation_pattern, x0=None, maxiter=200):
        """
        Solve for a given separation pattern.
        Returns (value, x) or (None, None) if failed.
        """
        if time.time() > deadline - 10:
            return None, None
            
        if x0 is None:
            # Initial guess: all squares small, placed in a grid
            s0 = np.full(n, 0.1)
            cx0 = np.linspace(0.1, 0.9, n)
            cy0 = np.linspace(0.1, 0.9, n)
            x0 = np.concatenate([s0, cx0, cy0])
        
        bounds = [(0, 1)] * (3*n)
        
        try:
            res = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_factory(separation_pattern),
                options={'maxiter': maxiter, 'ftol': 1e-12}
            )
            
            if res.success and res.fun < 0:
                val = -res.fun
                # Check validity strictly
                s = res.x[:n]
                cx = res.x[n:2*n]
                cy = res.x[2*n:3*n]
                centers = np.column_stack([cx, cy])
                if is_valid(centers, s):
                    return val, res.x
        except Exception:
            pass
        return None, None

    # --- Simulated Annealing over Separation Patterns ---
    
    # Initial pattern: try a grid-like separation
    # Sort indices by x then y? Or just random.
    # Let's start with a "row-major" pattern:
    # (0,1) x, (0,2) x, ... 
    # Actually, let's just start with all X separations, then flip some.
    
    best_val = 0.0
    best_x = None
    best_pattern = None
    
    # Generate initial pattern based on a 3x3 grid + 1 extra
    # Indices: 0,1,2 (top), 3,4,5 (mid), 6,7,8 (bot), 9 (extra)
    # This is complex to map directly. Let's just use a random pattern and refine.
    
    # A good heuristic initial pattern:
    # 0,1,2: x-sep
    # 3,4,5: x-sep
    # 6,7,8: x-sep
    # 9: y-sep with all?
    
    # Let's try a few specific structured patterns first.
    structured_patterns = []
    
    # Pattern 1: 3 rows of 3, 1 in corner
    # Rows: y-sep between rows. Cols: x-sep within rows.
    p = np.zeros(n_pairs, dtype=int)
    for k, (i, j) in enumerate(pairs):
        # Assign rows
        row_i = i // 3
        row_j = j // 3
        if row_i != row_j:
            p[k] = 1 # y-sep
        else:
            p[k] = 0 # x-sep
    structured_patterns.append(p)
    
    # Pattern 2: 4 rows of 2-3
    p = np.zeros(n_pairs, dtype=int)
    for k, (i, j) in enumerate(pairs):
        row_i = i // 3
        row_j = j // 3
        if row_i != row_j:
            p[k] = 1
        else:
            p[k] = 0
    # Modify for 4 rows: 0-1, 2-3, 4-5, 6-7, 8-9? 
    # Let's just use random perturbations of the best found.
    
    current_pattern = structured_patterns[0].copy()
    current_val, current_x = solve_pattern(current_pattern)
    
    if current_val and current_val > best_val:
        best_val = current_val
        best_x = current_x
        best_pattern = current_pattern.copy()
        
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.95
    max_sa_iters = 2000
    
    sa_iter = 0
    while T > T_min and time.time() < deadline - 60 and sa_iter < max_sa_iters:
        sa_iter += 1
        
        # Pick a random pair to flip
        k = np.random.randint(n_pairs)
        new_pattern = current_pattern.copy()
        new_pattern[k] = 1 - new_pattern[k]
        
        # Solve for new pattern
        # Use current_x as initial guess for speed, but it might be infeasible for new pattern.
        # SLSQP can handle this if we start close, but might fail.
        # Let's try with current_x first, if fails, try random.
        
        new_val, new_x = solve_pattern(new_pattern, x0=current_x, maxiter=50)
        
        if new_val is None:
            # If it failed, try a fresh random start? Too slow.
            # Or try with a perturbed current_x?
            pass
        else:
            delta = new_val - current_val
            if delta > 0 or np.random.random() < np.exp(delta / T):
                current_pattern = new_pattern
                current_val = new_val
                current_x = new_x
                
                if current_val > best_val:
                    best_val = current_val
                    best_x = current_x
                    best_pattern = current_pattern.copy()
        
        T *= cooling
    
    # Final refinement on best pattern with more iterations
    if best_x is not None and time.time() < deadline - 30:
        final_val, final_x = solve_pattern(best_pattern, x0=best_x, maxiter=1000)
        if final_val and final_val > best_val:
            best_val = final_val
            best_x = final_x
    
    # Fallback if SA failed completely
    if best_x is None:
        # Grid fallback
        k = 4
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)
    else:
        s = best_x[:n]
        cx = best_x[n:2*n]
        cy = best_x[2*n:3*n]
        centers = np.column_stack([cx, cy]).astype(np.float64)
        sides = s.astype(np.float64)
        
        # Final validity check
        if not is_valid(centers, sides):
             # Grid fallback
             k = 4
             side = 1.0 / k
             idx = np.arange(n)
             cols = idx % k
             rows = idx // k
             cx = (cols + 0.5) * side
             cy = (rows + 0.5) * side
             centers = np.stack([cx, cy], axis=1).astype(np.float64)
             sides = np.full(n, side, dtype=np.float64)

    return centers, sides

# EVOLVE-BLOCK-END

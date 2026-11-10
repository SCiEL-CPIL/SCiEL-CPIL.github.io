import numpy as np
import scipy.optimize as so
import time

def squares_in_square10() -> tuple:
    """
    Maximize the sum of side lengths of 10 squares in the unit square.
    Uses a two-stage approach:
    1. Global optimization with penalty method to find a good topology.
    2. Exact Linear Programming (LP) to optimize sizes/positions within that topology.
    """
    n = 10
    start_time = time.time()
    deadline = 600  # seconds
    
    best_score = -np.inf
    best_centers = None
    best_sides = None
    
    # --- Stage 1: Global Search ---
    
    def objective(params):
        # params: [x1..xn, y1..yn, s1..sn]
        x = params[0:n]
        y = params[n:2*n]
        s = params[2*n:3*n]
        
        # Boundary penalties
        pen_bound = 0.0
        for i in range(n):
            # x bounds
            if x[i] - s[i]/2 < 0:
                pen_bound += (x[i] - s[i]/2)**2
            if x[i] + s[i]/2 > 1:
                pen_bound += (x[i] + s[i]/2 - 1)**2
            # y bounds
            if y[i] - s[i]/2 < 0:
                pen_bound += (y[i] - s[i]/2)**2
            if y[i] + s[i]/2 > 1:
                pen_bound += (y[i] + s[i]/2 - 1)**2
                
        # Overlap penalties
        pen_overlap = 0.0
        for i in range(n):
            for j in range(i+1, n):
                dx = (s[i] + s[j])/2 - abs(x[i] - x[j])
                dy = (s[i] + s[j])/2 - abs(y[i] - y[j])
                # Use smooth max to allow gradients
                # softplus(v) = log(1 + exp(v)) / beta
                # Here we use a simple quadratic penalty if overlapping
                if dx > 0 and dy > 0:
                    pen_overlap += dx * dy
                    
        # Maximize sum(s) -> Minimize -sum(s) + penalties
        return -np.sum(s) + 1000.0 * pen_bound + 1000.0 * pen_overlap

    # Random restarts
    np.random.seed(42)
    n_restarts = 100
    
    for k in range(n_restarts):
        if time.time() - start_time > deadline - 60:
            break
            
        # Initial guess
        # Try different distributions
        if k % 2 == 0:
            s_init = np.random.uniform(0.1, 0.3, n)
        else:
            s_init = np.random.uniform(0.05, 0.5, n)
            
        x_init = np.random.uniform(0, 1, n)
        y_init = np.random.uniform(0, 1, n)
        params_init = np.concatenate([x_init, y_init, s_init])
        
        # Bounds
        bounds = [(0, 1)] * n + [(0, 1)] * n + [(0.01, 1)] * n
        
        try:
            res = so.minimize(objective, params_init, method='L-BFGS-B', 
                              bounds=bounds,
                              options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.success:
                x_opt = res.x[0:n]
                y_opt = res.x[n:2*n]
                s_opt = res.x[2*n:3*n]
                
                # Check validity roughly
                valid = True
                for i in range(n):
                    if x_opt[i] - s_opt[i]/2 < -1e-5 or x_opt[i] + s_opt[i]/2 > 1 + 1e-5:
                        valid = False
                    if y_opt[i] - s_opt[i]/2 < -1e-5 or y_opt[i] + s_opt[i]/2 > 1 + 1e-5:
                        valid = False
                if not valid:
                    continue
                    
                score = np.sum(s_opt)
                if score > best_score:
                    best_score = score
                    best_centers = np.column_stack([x_opt, y_opt])
                    best_sides = s_opt
                    
        except Exception:
            continue

    if best_centers is None:
        # Fallback to baseline if search failed
        k_grid = 4
        side = 1.0 / k_grid
        idx = np.arange(n)
        cols = idx % k_grid
        rows = idx // k_grid
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        return np.column_stack([cx, cy]), np.full(n, side)

    # --- Stage 2: LP Polish ---
    
    # Extract topology
    # For each pair, determine relation
    # Relations: 0: Left, 1: Right, 2: Below, 3: Above
    # We store constraints for linprog
    # Variables: x1..xn, y1..yn, s1..sn (3n vars)
    # Minimize -sum(s) -> c = [0]*n + [0]*n + [-1]*n
    
    c = np.zeros(3*n)
    c[2*n:3*n] = -1.0
    
    A_ub = []
    b_ub = []
    
    # Boundary constraints
    # x_i - s_i/2 >= 0  => -x_i + s_i/2 <= 0
    # x_i + s_i/2 <= 1  =>  x_i + s_i/2 <= 1
    # y_i - s_i/2 >= 0  => -y_i + s_i/2 <= 0
    # y_i + s_i/2 <= 1  =>  y_i + s_i/2 <= 1
    
    for i in range(n):
        # -x_i + 0.5 s_i <= 0
        row = np.zeros(3*n)
        row[i] = -1.0
        row[2*n+i] = 0.5
        A_ub.append(row)
        b_ub.append(0.0)
        
        # x_i + 0.5 s_i <= 1
        row = np.zeros(3*n)
        row[i] = 1.0
        row[2*n+i] = 0.5
        A_ub.append(row)
        b_ub.append(1.0)
        
        # -y_i + 0.5 s_i <= 0
        row = np.zeros(3*n)
        row[n+i] = -1.0
        row[2*n+i] = 0.5
        A_ub.append(row)
        b_ub.append(0.0)
        
        # y_i + 0.5 s_i <= 1
        row = np.zeros(3*n)
        row[n+i] = 1.0
        row[2*n+i] = 0.5
        A_ub.append(row)
        b_ub.append(1.0)
        
    # Pair constraints
    # Use best_centers, best_sides to determine topology
    for i in range(n):
        for j in range(i+1, n):
            xi, yi, si = best_centers[i,0], best_centers[i,1], best_sides[i]
            xj, yj, sj = best_centers[j,0], best_centers[j,1], best_sides[j]
            
            # Check x separation
            gap_x = xj - xj/2 - (xi + xi/2) # Wait, center based
            # Distance between right edge of i and left edge of j
            right_i = xi + si/2
            left_j = xj - sj/2
            right_j = xj + sj/2
            left_i = xi - si/2
            
            # Check if i is left of j
            if right_i <= left_j + 1e-5:
                # x_i + s_i/2 <= x_j - s_j/2
                # x_i + 0.5 s_i - x_j + 0.5 s_j <= 0
                row = np.zeros(3*n)
                row[i] = 1.0
                row[2*n+i] = 0.5
                row[j] = -1.0
                row[2*n+j] = 0.5
                A_ub.append(row)
                b_ub.append(0.0)
            elif right_j <= left_i + 1e-5:
                # x_j + s_j/2 <= x_i - s_i/2
                # -x_i + 0.5 s_i + x_j - 0.5 s_j <= 0
                row = np.zeros(3*n)
                row[i] = -1.0
                row[2*n+i] = 0.5
                row[j] = 1.0
                row[2*n+j] = -0.5
                A_ub.append(row)
                b_ub.append(0.0)
            else:
                # Overlap in x, must be separated in y
                top_i = yi + si/2
                bottom_j = yj - sj/2
                top_j = yj + sj/2
                bottom_i = yi - si/2
                
                if top_i <= bottom_j + 1e-5:
                    # i below j
                    # y_i + s_i/2 <= y_j - s_j/2
                    # y_i + 0.5 s_i - y_j + 0.5 s_j <= 0
                    row = np.zeros(3*n)
                    row[n+i] = 1.0
                    row[2*n+i] = 0.5
                    row[n+j] = -1.0
                    row[2*n+j] = 0.5
                    A_ub.append(row)
                    b_ub.append(0.0)
                elif top_j <= bottom_i + 1e-5:
                    # j below i (i above j)
                    # y_j + s_j/2 <= y_i - s_i/2
                    # -y_i + 0.5 s_i + y_j - 0.5 s_j <= 0
                    row = np.zeros(3*n)
                    row[n+i] = -1.0
                    row[2*n+i] = 0.5
                    row[n+j] = 1.0
                    row[2*n+j] = -0.5
                    A_ub.append(row)
                    b_ub.append(0.0)
                else:
                    # Ambiguous or overlapping in both.
                    # Pick one arbitrarily, e.g., i left of j
                    row = np.zeros(3*n)
                    row[i] = 1.0
                    row[2*n+i] = 0.5
                    row[j] = -1.0
                    row[2*n+j] = 0.5
                    A_ub.append(row)
                    b_ub.append(0.0)

    # Solve LP
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    try:
        res_lp = so.linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
        if res_lp.success:
            x_lp = res_lp.x[0:n]
            y_lp = res_lp.x[n:2*n]
            s_lp = res_lp.x[2*n:3*n]
            
            # Ensure sides are positive
            s_lp = np.maximum(s_lp, 1e-6)
            
            # Calculate score
            score_lp = np.sum(s_lp)
            
            if score_lp > best_score:
                best_score = score_lp
                best_centers = np.column_stack([x_lp, y_lp])
                best_sides = s_lp
                
    except Exception:
        pass
        
    return best_centers, best_sides


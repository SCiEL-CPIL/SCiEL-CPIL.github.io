import numpy as np
from scipy.optimize import minimize
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin

    # Precompute indices for constraints to speed up vectorization
    # We need pairs (i, j) with i < j
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_count = len(i_idx)

    def objective(params):
        # Minimize negative sum of radii
        # params: [x0, y0, x1, y1, ..., xN-1, yN-1, r0, r1, ..., rN-1]
        r = params[2*N:]
        return -np.sum(r)

    def constraint_overlaps(params):
        # Returns array of shape (pair_count,)
        # Constraint: dist(i,j) - ri - rj >= 0
        xy = params[:2*N].reshape(N, 2)
        r = params[2*N:]
        
        dx = xy[i_idx, 0] - xy[j_idx, 0]
        dy = xy[i_idx, 1] - xy[j_idx, 1]
        dist = np.sqrt(dx**2 + dy**2)
        
        return dist - r[i_idx] - r[j_idx]

    def constraint_left(params):
        # x_i - r_i >= 0
        return params[0:2*N:2] - params[2*N:]

    def constraint_right(params):
        # 1 - x_i - r_i >= 0
        return 1.0 - params[0:2*N:2] - params[2*N:]

    def constraint_bottom(params):
        # y_i - r_i >= 0
        return params[1:2*N:2] - params[2*N:]

    def constraint_top(params):
        # 1 - y_i - r_i >= 0
        return 1.0 - params[1:2*N:2] - params[2*N:]

    constraints = [
        {'type': 'ineq', 'fun': constraint_overlaps},
        {'type': 'ineq', 'fun': constraint_left},
        {'type': 'ineq', 'fun': constraint_right},
        {'type': 'ineq', 'fun': constraint_bottom},
        {'type': 'ineq', 'fun': constraint_top}
    ]

    bounds = [(0.0, 1.0) for _ in range(2*N)] + [(1e-6, 0.5) for _ in range(N)]

    def is_valid(params, tol=1e-8):
        # Quick validity check for tracking best result
        # Overlaps
        if np.any(constraint_overlaps(params) < -tol):
            return False
        # Boundaries
        if np.any(constraint_left(params) < -tol):
            return False
        if np.any(constraint_right(params) < -tol):
            return False
        if np.any(constraint_bottom(params) < -tol):
            return False
        if np.any(constraint_top(params) < -tol):
            return False
        return True

    def get_score(params):
        return np.sum(params[2*N:])

    best_score = -np.inf
    best_params = None
    rng = np.random.RandomState(42)

    # Initializations
    def init_grid():
        side = 6 # 6x6 grid covers 36, we take 32
        # Better: 8x4 grid? 8*4=32. 
        # Let's try 8x4
        cols, rows = 8, 4
        xy = np.zeros((N, 2))
        r = np.zeros(N)
        for i in range(N):
            row = i // cols
            col = i % cols
            xy[i] = [(col + 0.5) / cols, (row + 0.5) / rows]
            r[i] = 0.5 / cols - 0.01 # Slight margin
        return np.concatenate([xy.flatten(), r])

    def init_hexagonal():
        # Hexagonal packing approx
        # Rows of 6, 5, 6, 5, ...? 
        # 6+5+6+5+6+4 = 32?
        # Let's just use a dense random grid and let optimizer sort it out
        # Or a specific hex pattern
        xy = np.zeros((N, 2))
        r = np.zeros(N)
        # Generate hex grid points
        dx = 0.15
        dy = 0.15 * np.sqrt(3)
        count = 0
        row = 0
        while count < N:
            col = 0
            while count < N:
                x = col * dx + (0.075 if row % 2 else 0)
                y = row * dy
                if 0 <= x <= 1 and 0 <= y <= 1:
                    xy[count] = [x + 0.05, y + 0.05] # Shift to center
                    r[count] = 0.06
                    count += 1
                col += 1
            row += 1
        # Ensure we have N
        if count < N:
             # Fill remaining with random
             for k in range(count, N):
                 xy[k] = rng.uniform(0.1, 0.9, size=2)
                 r[k] = 0.05
        return np.concatenate([xy.flatten(), r])

    def init_random():
        xy = rng.uniform(0.1, 0.9, size=(N, 2))
        r = rng.uniform(0.05, 0.08, size=N)
        return np.concatenate([xy.flatten(), r])

    inits = [
        init_grid,
        init_hexagonal,
        init_random,
        init_random,
        init_random,
        init_random,
        init_random,
        init_random
    ]

    # Run optimization for multiple restarts
    for i, init_func in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        try:
            x0 = init_func()
            
            # Run SLSQP
            # maxiter is high, but ftol/gtol control stopping
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           constraints=constraints, 
                           options={'maxiter': 300, 'ftol': 1e-12, 'disp': False})
            
            if res.success or res.fun < best_score:
                # SLSQP might return a slightly invalid solution due to numerical errors
                # Check validity
                if is_valid(res.x, tol=1e-6):
                    score = get_score(res.x)
                    if score > best_score:
                        best_score = score
                        best_params = res.x.copy()
                else:
                    # Try to repair or just accept if it's close? 
                    # For now, skip if invalid to ensure final output is valid
                    # We can try a tiny shrink
                    r = res.x[2*N:].copy()
                    r *= 0.99
                    res.x[2*N:] = r
                    if is_valid(res.x, tol=1e-6):
                        score = get_score(res.x)
                        if score > best_score:
                            best_score = score
                            best_params = res.x.copy()
        except Exception as e:
            pass

    # Final Polish: If we have a best_params, run one more SLSQP with tighter tolerance
    if best_params is not None and time.time() < deadline - 2:
        try:
            res = minimize(objective, best_params, method='SLSQP', bounds=bounds, 
                           constraints=constraints, 
                           options={'maxiter': 200, 'ftol': 1e-14, 'disp': False})
            if res.success:
                if is_valid(res.x, tol=1e-7):
                    score = get_score(res.x)
                    if score > best_score:
                        best_score = score
                        best_params = res.x.copy()
        except Exception:
            pass

    if best_params is None:
        # Fallback: simple grid
        side = 6
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            x = (col + 0.5) / side
            y = (row + 0.5) / side
            r = 0.5 / side - 0.01
            cells.append([x, y, r])
        return np.array(cells)
    else:
        # Format output: (x, y, r)
        xy = best_params[:2*N].reshape(N, 2)
        r = best_params[2*N:]
        return np.column_stack([xy, r])

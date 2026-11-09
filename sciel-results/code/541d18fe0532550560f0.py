import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    best_score = 0.0
    best_config = None
    
    # Helper to check validity and return penalty (0 if valid) - Vectorized
    def validity_penalty(x, y, r):
        # Check bounds
        if np.any(r < 0):
            return 1e10
        if np.any(x - r < -1e-9) or np.any(y - r < -1e-9):
            return 1e10
        if np.any(x + r > 1 + 1e-9) or np.any(y + r > 1 + 1e-9):
            return 1e10
        
        # Check overlaps using broadcasting
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        r_sum = r[:, None] + r[None, :]
        
        # We only need to check i < j
        iu = np.triu_indices(N, k=1)
        if np.any(d[iu] < r_sum[iu] - 1e-9):
            return 1e10
        return 0.0

    # Helper to repair configuration by shrinking radii if invalid
    def repair_config(x, y, r):
        # First, clamp to boundaries
        r_max_bound = np.minimum(
            np.minimum(x, 1 - x),
            np.minimum(y, 1 - y)
        )
        r = np.minimum(r, r_max_bound)
        r = np.maximum(r, 0.0)
        
        # Iteratively shrink if overlaps persist
        for _ in range(50):
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(d, np.inf)
            
            r_sum = r[:, None] + r[None, :]
            overlaps = d < r_sum - 1e-9
            np.fill_diagonal(overlaps, False)
            
            if not np.any(overlaps):
                return x, y, r
            
            # Shrink all radii slightly
            r *= 0.95
        
        return x, y, r

    # Smooth Objective Function for L-BFGS-B
    def objective(v):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        
        # 1. Maximize sum of radii -> Minimize -sum(r)
        obj = -np.sum(r)
        
        # 2. Boundary Penalty (Squared)
        bound_pen = 0.0
        bound_pen += np.sum(np.maximum(0, r - x)**2)
        bound_pen += np.sum(np.maximum(0, r - (1 - x))**2)
        bound_pen += np.sum(np.maximum(0, r - y)**2)
        bound_pen += np.sum(np.maximum(0, r - (1 - y))**2)
        
        # 3. Overlap Penalty (Squared)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        r_sum = r[:, None] + r[None, :]
        
        # Only consider upper triangle
        iu = np.triu_indices(N, k=1)
        overlaps = np.maximum(0, r_sum[iu] - d[iu])
        overlap_pen = np.sum(overlaps**2)
        
        # Weighting
        lambda_bound = 10.0
        lambda_overlap = 10.0
        
        return obj + lambda_bound * bound_pen + lambda_overlap * overlap_pen

    def optimize_scipy(x, y, r, time_limit=10.0):
        v0 = np.concatenate([x, y, r])
        
        # Bounds for x, y, r
        # x, y in [0, 1]
        # r in [0, 0.5] (max radius in unit square is 0.5)
        bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N
        
        try:
            res = minimize(
                objective, 
                v0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-5}
            )
            
            if res.success or res.fun < objective(v0):
                v_opt = res.x
                x_opt = v_opt[0::3]
                y_opt = v_opt[1::3]
                r_opt = v_opt[2::3]
                
                # Post-process to ensure strict validity
                x_opt, y_opt, r_opt = repair_config(x_opt, y_opt, r_opt)
                
                return x_opt, y_opt, r_opt
        except Exception:
            pass
            
        return x, y, r

    # Local Search Refinement
    def local_search(x, y, r, max_iter=200):
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        best_val = np.sum(best_r)
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            # Perturb a random circle
            idx = np.random.randint(N)
            candidate_x = best_x.copy()
            candidate_y = best_y.copy()
            candidate_r = best_r.copy()
            
            # Adaptive step size
            step = 0.02 * (1 - it/max_iter) + 0.005
            dx = np.random.normal(0, step)
            dy = np.random.normal(0, step)
            dr = np.random.normal(0, step * 0.5)
            
            candidate_x[idx] += dx
            candidate_y[idx] += dy
            candidate_r[idx] += dr
            
            # Clamp
            candidate_x[idx] = np.clip(candidate_x[idx], 0, 1)
            candidate_y[idx] = np.clip(candidate_y[idx], 0, 1)
            candidate_r[idx] = np.clip(candidate_r[idx], 0.001, 0.5)
            
            # Check validity
            if validity_penalty(candidate_x, candidate_y, candidate_r) == 0:
                val = np.sum(candidate_r)
                if val > best_val:
                    best_x, best_y, best_r = candidate_x, candidate_y, candidate_r
                    best_val = val
        
        return best_x, best_y, best_r

    # Multiple restarts
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        np.random.seed(42 + restart)
        
        # Grid-like initialization with jitter
        side = int(np.ceil(np.sqrt(N)))
        xs = []
        ys = []
        rs = []
        
        # Shuffle indices to vary which cells are used if N < side^2
        indices = list(range(side * side))
        np.random.shuffle(indices)
        
        for idx in indices:
            if len(xs) >= N:
                break
            i = idx % side
            j = idx // side
            x = (i + 0.5) / side + np.random.uniform(-0.02, 0.02)
            y = (j + 0.5) / side + np.random.uniform(-0.02, 0.02)
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            xs.append(x)
            ys.append(y)
            rs.append(0.05)
        
        x = np.array(xs)
        y = np.array(ys)
        r = np.array(rs)
        
        # Run SciPy Optimizer
        x, y, r = optimize_scipy(x, y, r)
        
        # Local Search
        x, y, r = local_search(x, y, r, max_iter=100)
        
        # Final Repair
        x, y, r = repair_config(x, y, r)
        
        # Check Validity
        if validity_penalty(x, y, r) == 0:
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_config = np.column_stack((x, y, r))
        
        # If time is running out, try a quick polish on the current best
        if time.time() > deadline - 20:
            break

    if best_config is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_config

import numpy as np
from scipy.optimize import minimize
import time

N = 32

def construct():
    start_time = time.time()
    max_time = 145.0  # Safety margin
    
    best_score = -np.inf
    best_solution = None
    
    # Fixed seeds for reproducibility of the search process
    rng = np.random.RandomState(42)

    def clean_and_score(params):
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        r = np.clip(r, 0, 0.5)
        
        d_bound = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
        r = np.minimum(r, d_bound)
        
        # Iterative reduction for validity
        for _ in range(20):
            overlap_found = False
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = r[i] + r[j]
                    if dist < min_dist:
                        excess = min_dist - dist
                        # Reduce radii to resolve overlap
                        # Proportional reduction
                        if r[i] + r[j] > 0:
                            r[i] -= excess * (r[i] / (r[i] + r[j]))
                            r[j] -= excess * (r[j] / (r[i] + r[j]))
                        else:
                            r[i] = 0
                            r[j] = 0
                        overlap_found = True
            if not overlap_found:
                break
        
        r = np.clip(r, 0, 0.5)
        
        score = np.sum(r)
        
        result = np.zeros((N, 3))
        result[:, 0] = x
        result[:, 1] = y
        result[:, 2] = r
        return score, result

    def objective(params):
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        sum_r = np.sum(r)
        
        d_bound = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
        bound_viol = np.maximum(0, r - d_bound)
        bound_pen = np.sum(bound_viol**2)
        
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        
        np.fill_diagonal(min_dist, 0)
        np.fill_diagonal(dist, np.inf)
        
        overlap_viol = np.maximum(0, min_dist - dist)
        overlap_pen = np.sum(overlap_viol**2) / 2
        
        return -sum_r + 10.0 * (bound_pen + overlap_pen)
    
    def hex_init(offset_x=0.0, offset_y=0.0):
        pts = []
        row = 0
        while len(pts) < N + 5:
            y_pos = (row * 0.1732) + offset_y
            for col in range(10):
                x_pos = (col * 0.2) + (0.1 if row % 2 else 0.0) + offset_x
                if 0 <= x_pos <= 1 and 0 <= y_pos <= 1:
                    pts.append([x_pos, y_pos])
            row += 1
        
        pts = np.array(pts[:N])
        
        r_init = 0.05 + rng.uniform(0, 0.01, N)
        
        params = np.zeros(N * 3)
        params[0::3] = pts[:, 0]
        params[1::3] = pts[:, 1]
        params[2::3] = r_init
        return params

    def random_init():
        x = rng.uniform(0.1, 0.9, N)
        y = rng.uniform(0.1, 0.9, N)
        r = rng.uniform(0.05, 0.1, N)
        params = np.zeros(N * 3)
        params[0::3] = x
        params[1::3] = y
        params[2::3] = r
        return params

    restarts = 0
    # Pre-defined offsets for hex init to ensure diversity without randomness
    hex_offsets = [(0.0, 0.0), (0.05, 0.05), (0.1, 0.0), (0.0, 0.1), (0.05, -0.05)]
    
    while time.time() - start_time < max_time:
        # Cycle through init strategies
        idx = restarts % (len(hex_offsets) + 2)
        
        if idx < len(hex_offsets):
            ox, oy = hex_offsets[idx]
            x0 = hex_init(ox, oy)
        else:
            x0 = random_init()
            
        # Add small deterministic noise based on restart count
        noise = rng.normal(0, 0.01, size=x0.shape)
        x0 = x0 + noise
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-5, 'gtol': 1e-5})
            
            if res.success or res.status == 0:
                score, result = clean_and_score(res.x)
                if score > best_score:
                    best_score = score
                    best_solution = result
        except Exception:
            pass
            
        restarts += 1
        
    if best_solution is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_solution = np.array(cells[:N])
        
    return best_solution

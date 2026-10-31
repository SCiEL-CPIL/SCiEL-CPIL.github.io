"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    deadline = time.time() + 115.0  # Safety margin before 150s limit
    best_result = None
    best_score = 0.0
    best_w = 1.0

    # Aspect ratio search space: w from 0.5 to 1.5 (since w+h=2, h=2-w)
    # We'll search w in [0.5, 1.5]
    
    def unpack(vars_):
        w = vars_[0]
        x = vars_[1::3]
        y = vars_[2::3]
        r = vars_[3::3]
        return w, x, y, r

    def objective(vars_, penalty_weight):
        w, x, y, r = unpack(vars_)
        h = 2.0 - w
        
        # Sum of radii to maximize (minimize negative)
        sum_r = np.sum(r)
        
        # Overlap Penalty
        # d_ij = sqrt((x_i-x_j)^2 + (y_i-y_j)^2)
        # Penalty = max(0, r_i + r_j - d_ij)^2
        overlap_pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                overlap = r[i] + r[j] - d
                if overlap > 0:
                    overlap_pen += overlap * overlap
        
        # Wall Penalty
        wall_pen = 0.0
        for i in range(N):
            # Left wall: x - r >= 0 -> viol = r - x
            v1 = r[i] - x[i]
            if v1 > 0: wall_pen += v1 * v1
            
            # Right wall: w - x - r >= 0 -> viol = r + x - w
            v2 = r[i] + x[i] - w
            if v2 > 0: wall_pen += v2 * v2
            
            # Bottom wall: y - r >= 0 -> viol = r - y
            v3 = r[i] - y[i]
            if v3 > 0: wall_pen += v3 * v3
            
            # Top wall: h - y - r >= 0 -> viol = r + y - h
            v4 = r[i] + y[i] - h
            if v4 > 0: wall_pen += v4 * v4

        return -sum_r + penalty_weight * (overlap_pen + wall_pen)

    def validate(x, y, r, w, h):
        if np.any(r < -1e-6):
            return False
        if np.any(x < r - 1e-6) or np.any(x > w - r + 1e-6):
            return False
        if np.any(y < r - 1e-6) or np.any(y > h - r + 1e-6):
            return False
        
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - 1e-6:
                    return False
        return True

    def hex_init(w, h, seed):
        rng = np.random.default_rng(seed)
        # Estimate grid dimensions
        # Aim for roughly square cells
        n_cols = max(1, int(np.round(np.sqrt(N * w / h))))
        n_rows = max(1, int(np.ceil(N / n_cols)))
        
        while n_cols * n_rows < N:
            if n_cols <= n_rows:
                n_cols += 1
            else:
                n_rows += 1

        xs = np.zeros(N)
        ys = np.zeros(N)
        
        for i in range(N):
            col = i % n_cols
            row = i // n_cols
            
            x = (col + 0.5) * w / n_cols
            y = (row + 0.5) * h / n_rows
            
            if row % 2 == 1:
                x += 0.5 * w / n_cols
                if x > w:
                    x = w - (x - w) # Reflect
                elif x < 0:
                    x = -x

            xs[i] = x
            ys[i] = y

        # Jitter
        jitter_scale = 0.1 * min(w/n_cols, h/n_rows)
        xs += rng.normal(0, jitter_scale, N)
        ys += rng.normal(0, jitter_scale, N)
        
        # Rough clip
        xs = np.clip(xs, 0.01, w - 0.01)
        ys = np.clip(ys, 0.01, h - 0.01)
        
        # Initial radii: small
        est_r = min(w/n_cols, h/n_rows) * 0.4
        rs = np.full(N, est_r)
        
        return xs, ys, rs

    # Search parameters
    # We will try a few aspect ratios
    aspect_ratios = [0.8, 0.9, 1.0, 1.1, 1.2] # w values
    # Penalty weights to try (increasing to enforce constraints)
    penalty_weights = [10, 100, 1000]
    restarts_per_config = 3

    total_configs = len(aspect_ratios) * len(penalty_weights) * restarts_per_config
    
    # Time budget per optimization call
    # 115s / total_configs ~ 115/45 ~ 2.5s per call
    # L-BFGS-B is fast. maxiter=1000 should be safe.

    for w_init in aspect_ratios:
        if time.time() > deadline - 5:
            break
        h_init = 2.0 - w_init
        
        for penalty_weight in penalty_weights:
            if time.time() > deadline - 2:
                break
                
            for restart in range(restarts_per_config):
                if time.time() > deadline - 1:
                    break
                    
                seed = restart * 100 + int(w_init * 100)
                xs, ys, rs = hex_init(w_init, h_init, seed)
                
                # Variable vector: [w, x0, y0, r0, x1, y1, r1, ...]
                x0 = np.zeros(1 + 3*N)
                x0[0] = w_init
                x0[1::3] = xs
                x0[2::3] = ys
                x0[3::3] = rs
                
                # Bounds
                # w in [0.5, 1.5]
                # x, y, r > 0
                # r < min(w, h)/2 approx, but let's be loose
                lb = np.zeros(1 + 3*N)
                ub = np.zeros(1 + 3*N)
                lb[0] = 0.5
                ub[0] = 1.5
                for i in range(N):
                    lb[1 + 3*i] = 0.0 # x
                    ub[1 + 3*i] = 2.0
                    lb[2 + 3*i] = 0.0 # y
                    ub[2 + 3*i] = 2.0
                    lb[3 + 3*i] = 0.0 # r
                    ub[3 + 3*i] = 1.0

                try:
                    result = minimize(
                        objective, x0, args=(penalty_weight,),
                        method='L-BFGS-B',
                        bounds=list(zip(lb, ub)),
                        options={'maxiter': 200, 'ftol': 1e-9}
                    )
                    
                    w, x, y, r = unpack(result.x)
                    h = 2.0 - w
                    
                    # Check validity
                    # If penalty is high and objective is low, it should be valid
                    # We strictly validate
                    if validate(x, y, r, w, h):
                        score = np.sum(r)
                        if score > best_score:
                            best_score = score
                            best_result = np.column_stack([x, y, r])
                            best_w = w
                            
                except Exception:
                    continue

    # Fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3))
        return np.array(cells)

    return best_result
# EVOLVE-BLOCK-END

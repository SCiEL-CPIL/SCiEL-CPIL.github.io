import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    best_score = -np.inf
    best_result = None
    
    # Search over aspect ratios w (width). h = 2 - w.
    # w must be in [0.5, 1.5] to keep h >= 0.5 and aspect ratio reasonable.
    # Actually, due to symmetry w and h are interchangeable, so w in [0.5, 1.0] is sufficient if we consider w <= h.
    # But let's just search w in [0.5, 1.5].
    
    w_values = np.linspace(0.5, 1.5, 21)
    
    for w in w_values:
        if time.time() > deadline:
            break
            
        h = 2.0 - w
        
        # Initialize positions and radii
        # Use a grid initialization
        # Approximate number of circles per row/col
        # Let's just use a 5x5 grid and take 21
        cols = 5
        rows = 5
        x_init = np.array([(i + 0.5) / cols * w for i in range(cols) for j in range(rows)])
        y_init = np.array([(j + 0.5) / rows * h for j in range(rows) for i in range(cols)])
        
        # We need 21 circles.
        # The grid has 25. We take the first 21.
        # This might bias towards top-left. 
        # Better: create a grid that covers the area more evenly or just use random init?
        # Random init with multiple restarts is often better for global optimization.
        
        # Let's do a few random restarts for each aspect ratio if time permits.
        # But first, try the grid init.
        
        def objective_and_grad(x):
            # x is flattened: [x1..xN, y1..yN, r1..rN]
            xs = x[:N]
            ys = x[N:2*N]
            rs = x[2*N:3*N]
            
            # Penalty for boundaries
            pen = 0.0
            # x bounds
            pen += np.sum(np.maximum(0, rs - xs)**2)
            pen += np.sum(np.maximum(0, rs - ys)**2)
            pen += np.sum(np.maximum(0, rs - (w - xs))**2)
            pen += np.sum(np.maximum(0, rs - (h - ys))**2)
            
            # Penalty for overlaps
            # Vectorized calculation of pairwise distances
            # diff_x = xs[:, None] - xs[None, :]
            # diff_y = ys[:, None] - ys[None, :]
            # dist = np.sqrt(diff_x**2 + diff_y**2)
            # sum_r = rs[:, None] + rs[None, :]
            # violations = np.maximum(0, sum_r - dist)
            # pen += np.sum(violations**2)
            
            # To save time, we can skip the vectorized version and just loop if N is small, 
            # but vectorized is faster in Python/NumPy.
            diff_x = xs[:, None] - xs[None, :]
            diff_y = ys[:, None] - ys[None, :]
            dist_sq = diff_x**2 + diff_y**2
            # Avoid division by zero in gradient if needed, but here we just use dist
            # dist = np.sqrt(dist_sq)
            # sum_r = rs[:, None] + rs[None, :]
            # violations = np.maximum(0, sum_r - dist)
            # pen += np.sum(violations**2)
            
            # We can use a simpler penalty: (sum_r^2 - dist_sq) if positive?
            # Let's stick to the distance penalty.
            dist = np.sqrt(dist_sq + 1e-12)
            sum_r = rs[:, None] + rs[None, :]
            violations = np.maximum(0, sum_r - dist)
            pen += np.sum(violations**2)
            
            val = -np.sum(rs) + 1000.0 * pen
            return val
            
        # We don't provide gradient to L-BFGS-B, it will estimate it.
        # Providing gradient is faster but complex. Let's rely on numerical gradient first.
        
        x0 = np.concatenate([x_init[:N], y_init[:N], np.full(N, 0.1)])
        
        # Bounds:
        # x in [0, w], y in [0, h], r in [0, min(w,h)/2]
        bounds = []
        for i in range(N):
            bounds.append((0, w))
        for i in range(N):
            bounds.append((0, h))
        for i in range(N):
            bounds.append((0, min(w, h)/2))
            
        try:
            res = minimize(objective_and_grad, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.success or res.fun < 100: # Check if penalty is low
                xs = res.x[:N]
                ys = res.x[N:2*N]
                rs = res.x[2*N:3*N]
                
                # Verify validity
                valid = True
                # Check boundaries
                if np.any(xs < -1e-6) or np.any(ys < -1e-6) or np.any(xs > w + 1e-6) or np.any(ys > h + 1e-6):
                    valid = False
                if np.any(rs < -1e-6):
                    valid = False
                if np.any(rs > xs + 1e-6) or np.any(rs > ys + 1e-6) or np.any(rs > w - xs + 1e-6) or np.any(rs > h - ys + 1e-6):
                    valid = False
                    
                # Check overlaps
                for i in range(N):
                    for j in range(i+1, N):
                        dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                        if dist < rs[i] + rs[j] - 1e-6:
                            valid = False
                            break
                    if not valid: break
                
                if valid:
                    score = np.sum(rs)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack([xs, ys, rs])
                        
        except Exception:
            pass
            
    if best_result is not None:
        return best_result
    else:
        # Fallback to seed if optimization failed completely
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_score = 0.0
    best_solution = None
    
    # Try different aspect ratios
    # W + H = 2. W in [0.8, 1.2] covers a good range around square.
    W_values = np.linspace(0.7, 1.3, 15)
    
    for W in W_values:
        if time.time() > deadline:
            break
        H = 2.0 - W
        
        # Initial guess: Grid packing
        # Fit as many as possible in a grid
        # For N=21, a 5x5 grid is 25, so we take 21.
        # Or 4x6 = 24, take 21.
        # Let's use a 5x5 grid subset.
        side = 5
        # Scale grid to fit in W x H
        # Grid cell size: W/side, H/side
        # Radius max: 0.5 * min(W/side, H/side)
        
        # We will optimize positions and radii.
        # Variables: x[0..20], y[0..20], r[0..20]
        # Total 63 variables.
        
        # Initialize
        xs = np.array([(i + 0.5) * W / side for i in range(5)] * 5)
        ys = np.array([(j + 0.5) * H / side for j in range(5) for i in range(5)])
        # Take first 21
        xs = xs[:21]
        ys = ys[:21]
        r0 = 0.5 * min(W, H) / side
        rs = np.full(21, r0)
        
        x0 = np.concatenate([xs, ys, rs])
        
        def penalty_and_obj(x):
            xs = x[:21]
            ys = x[21:42]
            rs = x[42:63]
            
            # Objective: -sum(r)
            obj = -np.sum(rs)
            
            # Penalty
            pen = 0.0
            
            # Boundary constraints
            # r <= x <= W-r  => x-r >= 0, W-x-r >= 0
            # r <= y <= H-r  => y-r >= 0, H-y-r >= 0
            
            viol_x1 = np.maximum(0, rs - xs)
            viol_x2 = np.maximum(0, rs - (W - xs))
            viol_y1 = np.maximum(0, rs - ys)
            viol_y2 = np.maximum(0, rs - (H - ys))
            
            pen += np.sum(viol_x1**2) + np.sum(viol_x2**2) + np.sum(viol_y1**2) + np.sum(viol_y2**2)
            
            # Overlap constraints
            # d_ij >= r_i + r_j
            # (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
            
            # Vectorized calculation
            # dx = xs[:, None] - xs[None, :]
            # dy = ys[:, None] - ys[None, :]
            # dr = rs[:, None] + rs[None, :]
            # d2 = dx**2 + dy**2
            # dr2 = dr**2
            # viol = np.maximum(0, dr2 - d2)
            # pen += np.sum(viol**2) # This is heavy, 21x21 matrix
            
            # To save time, we might skip vectorization if it's too slow, but 21 is small.
            dx = xs[:, None] - xs[None, :]
            dy = ys[:, None] - ys[None, :]
            dr = rs[:, None] + rs[None, :]
            d2 = dx**2 + dy**2
            dr2 = dr**2
            viol = np.maximum(0, dr2 - d2)
            # Only sum upper triangle to avoid double counting, but for penalty it doesn't matter much
            pen += np.sum(viol**2)
            
            # We want to maximize sum(r), so minimize -sum(r) + mu * pen
            # If pen is 0, we return -sum(r).
            # If pen > 0, we add a large penalty.
            mu = 100.0
            return obj + mu * pen

        # We want to minimize the penalty first, then maximize radii?
        # Or just minimize -sum(r) + penalty.
        
        # Run optimization
        # Use L-BFGS-B
        # Bounds:
        # x: [0, W]
        # y: [0, H]
        # r: [0, min(W,H)/2]
        
        bounds = []
        for _ in range(21):
            bounds.append((0.0, W))
        for _ in range(21):
            bounds.append((0.0, H))
        for _ in range(21):
            bounds.append((0.0, 0.5 * min(W, H)))

        try:
            res = minimize(
                penalty_and_obj, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-8}
            )
            
            if res.success or res.status == 0:
                x_opt = res.x
                xs_opt = x_opt[:21]
                ys_opt = x_opt[21:42]
                rs_opt = x_opt[42:63]
                
                # Check validity strictly
                valid = True
                if np.any(rs_opt < 1e-6):
                    valid = False
                if np.any(xs_opt < rs_opt - 1e-6) or np.any(xs_opt > W - rs_opt + 1e-6):
                    valid = False
                if np.any(ys_opt < rs_opt - 1e-6) or np.any(ys_opt > H - rs_opt + 1e-6):
                    valid = False
                
                # Check overlaps
                dx = xs_opt[:, None] - xs_opt[None, :]
                dy = ys_opt[:, None] - ys_opt[None, :]
                dr = rs_opt[:, None] + rs_opt[None, :]
                d2 = dx**2 + dy**2
                # Check if any pair has d2 < dr2
                # Exclude diagonal
                mask = ~np.eye(21, dtype=bool)
                if np.any(d2[mask] < dr[mask]**2 - 1e-6):
                    valid = False
                
                if valid:
                    score = np.sum(rs_opt)
                    if score > best_score:
                        best_score = score
                        best_solution = np.column_stack([xs_opt, ys_opt, rs_opt])
                        
        except Exception:
            pass
            
    if best_solution is not None:
        return best_solution
    else:
        # Fallback to seed solution
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        # Scale to perimeter 4
        # Seed returns unit square? No, seed returns x,y,r.
        # Seed: side=5. x in [0,1], y in [0,1]. Perimeter 4.
        # So x,y are in [0,1], r ~ 0.1.
        # My solution is in [0, W] x [0, H] with W+H=2.
        # The evaluator expects a rectangle with perimeter <= 4.
        # If I return x,y in [0,W] and [0,H], the bounding box is W x H.
        # Perimeter 2(W+H) = 4. Correct.
        
        # The seed program returns values in [0,1] for a 1x1 square (perimeter 4).
        # My program returns values in [0,W]x[0,H].
        # This is consistent.
        
        # If no valid solution found, return seed.
        return np.array(cells[:N])

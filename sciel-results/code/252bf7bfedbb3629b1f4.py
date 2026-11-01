import numpy as np
import time

N = 200

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    # 1. Initialization: Hexagonal lattice
    # We want approx 200 points.
    # Hex grid with spacing s.
    # Let's generate a large hex grid and center it.
    
    def hex_grid(n_x, n_y, s):
        pts = []
        for i in range(n_x):
            for j in range(n_y):
                x = i * s
                y = j * s * np.sqrt(3)
                if i % 2 == 1:
                    y += s * np.sqrt(3) / 2
                pts.append([x, y])
        return np.array(pts)
    
    # Estimate s for 200 points in 1x1 square.
    # Area per point ~ 1/sqrt(200) ~ 0.07.
    # Hex area per point ~ s^2 * sqrt(3)/2.
    # s^2 ~ 0.07 * 2 / 1.732 ~ 0.08. s ~ 0.28.
    # Let's try s = 0.25.
    # n_x, n_y such that n_x * n_y ~ 200. 14x15 = 210.
    # Grid width ~ 14 * 0.25 = 3.5. Too wide.
    # We need to scale it down to fit in 1x1.
    
    # Generate raw grid
    n_x, n_y = 14, 15
    s_init = 1.0
    pts = hex_grid(n_x, n_y, s_init)
    
    # Center and scale to fit in [-0.5, 0.5]
    pts -= pts.mean(axis=0)
    max_extent = np.max(np.abs(pts))
    pts /= max_extent * 1.1  # 10% margin
    
    # We have 210 points. Need 200.
    # Remove points furthest from center? Or just take first 200?
    # Taking first 200 might bias. Let's remove 10 random points or the 10 furthest from center.
    # Removing furthest from center helps pack the center denser? No, we want uniform.
    # Let's just take the first 200. The optimizer will fix it.
    if len(pts) > N:
        # Sort by distance from center, keep closest N
        d2 = np.sum(pts**2, axis=1)
        idx = np.argsort(d2)[:N]
        pts = pts[idx]
    elif len(pts) < N:
        # If too few, pad with random points inside
        missing = N - len(pts)
        new_pts = np.random.uniform(-0.4, 0.4, (missing, 2))
        pts = np.vstack([pts, new_pts])
        
    C = pts.copy()
    
    # 2. Optimization
    # Objective: Maximize min distance.
    # Use soft-min gradient ascent.
    
    # Precompute indices for pairs
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    # Parameters
    beta = 50.0  # Sharpness of soft-min
    lr = 0.001   # Learning rate
    min_lr = 1e-5
    
    # Wall constraints
    # d_x_left = 0.5 - C[:,0]
    # d_x_right = 0.5 + C[:,0]
    # d_y_bottom = 0.5 - C[:,1]
    # d_y_top = 0.5 + C[:,1]
    
    best_score = 0.0
    best_C = C.copy()
    
    for step in range(5000):
        if time.time() - start_time > time_limit:
            break
            
        # Compute pair distances
        diff = C[i_idx] - C[j_idx]
        d_pairs = np.linalg.norm(diff, axis=1)
        
        # Compute wall distances
        d_walls = np.concatenate([
            0.5 - C[:, 0],
            0.5 + C[:, 0],
            0.5 - C[:, 1],
            0.5 + C[:, 1]
        ])
        
        # Combine all distances
        # Note: For pairs, the radius contribution is d/2.
        # For walls, the radius contribution is d.
        # To compare them in the min, we should scale them.
        # Let's define effective radius r_k for each constraint.
        r_pairs = d_pairs / 2.0
        r_walls = d_walls
        
        r_all = np.concatenate([r_pairs, r_walls])
        
        # Update best score (approximate)
        curr_score = np.min(r_all)
        if curr_score > best_score:
            best_score = curr_score
            best_C = C.copy()
            
        # Compute weights for soft-min
        # Avoid underflow/overflow
        r_min = np.min(r_all)
        # Shift to make max exponent 0
        r_shifted = r_all - r_min
        weights = np.exp(-beta * r_shifted)
        weights /= np.sum(weights)
        
        # Compute Gradient
        grad = np.zeros((N, 2))
        
        # Pair gradients
        # grad_pair_i = (C_i - C_j) / (2 * d_ij)  (since r = d/2, dr/dCi = (Ci-Cj)/d)
        # Wait, dr/dCi = d/dCi (0.5 * ||Ci-Cj||) = 0.5 * (Ci-Cj)/||Ci-Cj||
        # So grad_i = 0.5 * diff / d_pairs
        
        # We have weights for r_pairs.
        w_pairs = weights[:len(r_pairs)]
        
        # Vectorized gradient accumulation for pairs
        # grad[i] += w * (C[i] - C[j]) / d
        # grad[j] += w * (C[j] - C[i]) / d
        
        # diff = C[i] - C[j]
        # grad_contrib = w_pairs[:, np.newaxis] * (0.5 * diff / d_pairs[:, np.newaxis])
        
        # Use np.add.at for accumulation
        g_pairs = w_pairs[:, np.newaxis] * (0.5 * diff / np.maximum(d_pairs, 1e-8)[:, np.newaxis])
        
        np.add.at(grad, i_idx, g_pairs)
        np.add.at(grad, j_idx, -g_pairs)
        
        # Wall gradients
        w_walls = weights[len(r_pairs):]
        
        # d_walls order: x_left, x_right, y_bottom, y_top for each i
        # d_x_left = 0.5 - C[:,0] -> dr/dC_x = -1
        # d_x_right = 0.5 + C[:,0] -> dr/dC_x = 1
        # d_y_bottom = 0.5 - C[:,1] -> dr/dC_y = -1
        # d_y_top = 0.5 + C[:,1] -> dr/dC_y = 1
        
        # w_walls shape (4N)
        w_xl = w_walls[0::4]
        w_xr = w_walls[1::4]
        w_yb = w_walls[2::4]
        w_yt = w_walls[3::4]
        
        # grad_x += -w_xl + w_xr
        # grad_y += -w_yb + w_yt
        
        grad[:, 0] += (-w_xl + w_xr)
        grad[:, 1] += (-w_yb + w_yt)
        
        # Update
        C += lr * grad
        
        # Project to square (hard constraint to ensure validity)
        C[:, 0] = np.clip(C[:, 0], -0.499, 0.499)
        C[:, 1] = np.clip(C[:, 1], -0.499, 0.499)
        
        # Anneal
        if step % 100 == 0:
            lr *= 0.95
            beta *= 1.1
            if lr < min_lr:
                lr = min_lr
        
    # Final check and return
    # Calculate exact score for best_C
    def calc_score(C):
        i_idx, j_idx = np.triu_indices(N, k=1)
        d_pairs = np.linalg.norm(C[i_idx] - C[j_idx], axis=1)
        r_pairs = d_pairs / 2.0
        r_walls = np.min(np.min(0.5 - np.abs(C), axis=1))
        return min(np.min(r_pairs), r_walls)
    
    # Ensure best_C is valid
    if not np.all(np.isfinite(best_C)):
        return C
    
    # Clip to be strictly inside
    best_C[:, 0] = np.clip(best_C[:, 0], -0.5, 0.5)
    best_C[:, 1] = np.clip(best_C[:, 1], -0.5, 0.5)
    
    return best_C

import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    best_points = None
    best_score = -1.0
    
    # Initializations
    # 1. Grid
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    inits = [grid[:N].copy()]
    
    # 2. Random perturbations of grid
    rng = np.random.default_rng(42)
    for _ in range(10):
        p = inits[0].copy()
        p += rng.normal(0, 0.02, p.shape)
        p = np.clip(p, -0.5, 0.5)
        inits.append(p)
        
    # 3. Fully random
    for _ in range(5):
        p = rng.uniform(-0.45, 0.45, (N, 3))
        inits.append(p)

    def get_score(points):
        # Calculate min pairwise distance
        dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Calculate min wall distance
        # dist to wall: min(0.5 - x, x + 0.5) for each coord
        # Actually, distance to boundary of [-0.5, 0.5]
        # For a point x in [-0.5, 0.5], dist to -0.5 is x+0.5, dist to 0.5 is 0.5-x
        # min of these is min(x+0.5, 0.5-x) = 0.5 - |x|
        wall_dists = 0.5 - np.abs(points)
        min_wall_dist = np.min(wall_dists)
        
        r = min(min_pair_dist / 2.0, min_wall_dist)
        return r

    def optimize(points, max_steps=5000):
        # Gradient ascent on sum of inverse squared distances
        # S = sum_{i<j} 1/d_ij^2
        # grad_i = sum_{j != i} 2 * (x_i - x_j) / d_ij^4
        
        x = points.copy()
        lr = 0.001
        # Adaptive learning rate or fixed? Fixed with decay is safer.
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Compute distances
            diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, 1.0) # Avoid div by 0, but we will mask later
            
            # Mask diagonal
            mask = ~np.eye(N, dtype=bool)
            
            # We want to maximize sum 1/d^2
            # grad = sum_{j != i} 2 * diff_{ij} / dists_{ij}^4
            # Note: diff_{ij} = x_i - x_j
            # d/dx_i (1/||x_i - x_j||^2) = -2 (x_i - x_j) / ||x_i - x_j||^4
            # Wait, we want to MAXIMIZE.
            # Let f(d) = 1/d^2. f'(d) = -2/d^3.
            # d/dx_i f(||x_i - x_j||) = f'(d) * (x_i - x_j)/d = -2/d^3 * (x_i - x_j)/d = -2 (x_i - x_j) / d^4.
            # So gradient of S w.r.t x_i is sum_{j} -2 (x_i - x_j) / d_{ij}^4.
            # We want to move in direction of gradient to increase S.
            # So x_i += alpha * grad_i.
            
            # To avoid numerical issues with very small distances, we can clip dists
            # But if dists are small, the force is huge, which is good for pushing apart.
            # However, it might cause instability.
            
            inv_d4 = np.zeros_like(dists)
            inv_d4[mask] = 1.0 / (dists[mask] ** 4)
            
            # grad[i] = sum_j -2 * (x[i] - x[j]) * inv_d4[i,j]
            # diff[i,j] = x[i] - x[j]
            # grad = -2 * sum_j diff[i,j] * inv_d4[i,j]
            grad = -2.0 * np.sum(diff * inv_d4[:, :, np.newaxis], axis=1)
            
            x += lr * grad
            
            # Clamp to box
            x = np.clip(x, -0.5, 0.5)
            
            # Optional: If we detect overlap (dist < 2*current_best_r?), we might need to handle it,
            # but the gradient should push them apart.
            
        return x

    for init in inits:
        if time.time() > deadline:
            break
        
        # Run optimization
        # We can run for a fixed number of steps or until time limit
        # Let's run in chunks to check time
        optimized = optimize(init, max_steps=2000)
        
        score = get_score(optimized)
        if score > best_score:
            best_score = score
            best_points = optimized.copy()
            
        # If we have time, maybe do a second pass with smaller learning rate?
        # Or just try the next init.

    # Final check
    if best_points is None:
        return inits[0]
    
    return best_points

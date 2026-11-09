import numpy as np

N = 50

def construct():
    # Initialize with a grid, but add a small random jitter to break symmetry
    # The grid is a safe starting point.
    m = int(np.ceil(np.sqrt(N)))
    # We use a grid of m x m, taking the first N points
    # To make it more "circular" in distribution, we can center the grid points
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Add a very small random perturbation to avoid exact symmetry locks
    np.random.seed(42) # Fixed seed for reproducibility in this context, 
                       # though random would be fine if we averaged multiple runs.
                       # However, deterministic is safer for evaluation consistency if needed.
                       # Let's use a fixed small perturbation.
    centers += np.random.normal(0, 0.001, centers.shape)
    
    # Ensure initial centers are within bounds
    centers = np.clip(centers, -0.5 + 1e-9, 0.5 - 1e-9)

    # Optimization parameters
    max_iters = 2000
    learning_rate = 0.01
    # We will use a simple gradient ascent on the minimum distance.
    # Objective: Maximize r = min( min_{i<j} dist(i,j)/2, min_i dist(i, wall) )
    
    for it in range(max_iters):
        # Calculate current radius
        # 1. Min distance to walls
        dist_walls = np.minimum(
            np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5),
            np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5)
        )
        min_wall_dist = np.min(dist_walls)
        
        # 2. Min distance between circles
        # Calculate pairwise distances efficiently
        # dist[i, j] = norm(centers[i] - centers[j])
        # We only need the minimum, so we can compute all or use a trick.
        # For N=50, N^2 = 2500 is small.
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        # Set diagonal to infinity to ignore self-distance
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        current_r = min(min_wall_dist, min_pair_dist / 2.0)
        
        # If radius is not improving, we might be stuck, but for now just continue.
        # Gradient computation
        grad = np.zeros_like(centers)
        
        # Identify active constraints
        # Active if dist is close to the min (within a small epsilon)
        eps = 1e-5
        active_wall_mask = (dist_walls <= min_wall_dist + eps)
        active_pair_mask = (dists <= min_pair_dist + eps)
        
        # Gradient from walls
        # If a circle is close to a wall, push it inward.
        # x wall: if x is close to -0.5, push +x. if x is close to 0.5, push -x.
        # This is equivalent to pushing towards 0 if it's the limiting wall.
        # More precisely:
        # grad_x += sign(centers[:, 0]) if close to wall? 
        # Actually, the gradient of (0.5 - |x|) is -sign(x).
        # We want to MAXIMIZE the distance, so we move in direction of gradient.
        # dist = 0.5 - x (if x>0). grad = -1. Move by -grad = +1? No.
        # f(x) = 0.5 - x. f'(x) = -1. Ascent: x += lr * f'(x) = x - lr. Moves left (towards 0). Correct.
        # f(x) = x + 0.5 (if x<0). f'(x) = 1. Ascent: x += lr * 1. Moves right (towards 0). Correct.
        
        # Calculate wall gradients
        # For x
        gx = np.zeros(N)
        gy = np.zeros(N)
        
        # Check x walls
        close_left_x = (centers[:, 0] < 0) & (0.5 + centers[:, 0] <= min_wall_dist + eps)
        close_right_x = (centers[:, 0] >= 0) & (0.5 - centers[:, 0] <= min_wall_dist + eps)
        # If close to left wall (x < 0), dist = 0.5 + x, grad = 1
        gx[close_left_x] += 1.0
        # If close to right wall (x > 0), dist = 0.5 - x, grad = -1
        gx[close_right_x] += -1.0
        
        # Check y walls
        close_bottom_y = (centers[:, 1] < 0) & (0.5 + centers[:, 1] <= min_wall_dist + eps)
        close_top_y = (centers[:, 1] >= 0) & (0.5 - centers[:, 1] <= min_wall_dist + eps)
        gy[close_bottom_y] += 1.0
        gy[close_top_y] += -1.0
        
        grad[:, 0] += gx
        grad[:, 1] += gy
        
        # Gradient from pairs
        # If dist(i,j) is min, push i away from j and j away from i.
        # dist = |ci - cj|. 
        # grad_i = (ci - cj) / |ci - cj|
        # grad_j = (cj - ci) / |cj - cj|
        
        # We only consider active pairs
        # Note: dists is symmetric.
        for i in range(N):
            for j in range(i+1, N):
                if active_pair_mask[i, j]:
                    diff_ij = centers[i] - centers[j]
                    dist_ij = dists[i, j]
                    if dist_ij > 1e-10:
                        unit_vec = diff_ij / dist_ij
                        # Push i away from j
                        grad[i] += unit_vec
                        # Push j away from i
                        grad[j] -= unit_vec
        
        # Update centers
        # Adaptive learning rate? Start high, decay.
        lr = learning_rate * (0.99 ** (it // 10))
        
        centers += lr * grad
        
        # Clamp to square to ensure validity (centers must be inside)
        # If a center goes outside, it reduces the radius to 0 or negative.
        # Clamping prevents invalid states.
        centers = np.clip(centers, -0.5, 0.5)
        
        # Optional: Add small noise to escape local minima
        if it % 100 == 0 and it > 0:
             centers += np.random.normal(0, 0.0001, centers.shape)
             centers = np.clip(centers, -0.5, 0.5)

    return centers

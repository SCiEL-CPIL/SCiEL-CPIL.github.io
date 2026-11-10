import numpy as np
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a Fekete point optimization on the unit sphere.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin
    
    best_score = -1.0
    best_points = None

    # Multiple restarts to avoid local minima
    n_restarts = 20
    max_steps = 5000
    lr = 0.01  # Learning rate

    for r in range(n_restarts):
        # Check time
        if time.time() > deadline:
            break
            
        # Initialize points randomly on the unit sphere
        # Use a deterministic seed for reproducibility of the "random" part if needed, 
        # but here we want variety. 
        # However, for a deterministic output, we should fix the seed.
        # The task asks for a function returning an array. It doesn't say it must be deterministic 
        # across calls, but usually, these tasks expect a specific high-quality configuration.
        # Using a fixed seed ensures the same good configuration is returned every time.
        np.random.seed(42 + r)
        
        # Generate random points on sphere
        vecs = np.random.randn(n, d)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        points = vecs / norms
        
        # Optimization loop
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
            # Compute pairwise distances
            # We need gradients of log(prod(dist))
            # grad_i = sum_{j != i} (p_i - p_j) / ||p_i - p_j||^2
            
            # Efficient calculation
            # diff[i, j] = p_i - p_j
            # dist2[i, j] = ||p_i - p_j||^2
            
            # Use broadcasting
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dist2 = np.sum(diff**2, axis=2)
            
            # Avoid division by zero
            np.fill_diagonal(dist2, 1.0)
            
            # Gradient term: (p_i - p_j) / dist2[i,j]
            # We want sum over j for each i
            # grad[i] = sum_j (p_i - p_j) / dist2[i,j]
            
            # We can compute this as:
            # weights = 1 / dist2
            # weights[i,i] = 0
            np.fill_diagonal(dist2, 1.0) # Already done, but ensure no zeros in off-diagonal for safety? 
            # Actually, if points coincide, dist2 is 0. 
            # Adding a small epsilon or handling separately.
            # For Fekete, if points coincide, gradient is undefined/inf. 
            # We can add a small jitter or break if dist2 is too small.
            
            # Let's just use a safe division
            inv_dist2 = 1.0 / dist2
            np.fill_diagonal(inv_dist2, 0.0)
            
            # grad = diff * inv_dist2[:, :, np.newaxis]
            # Sum over j (axis 1)
            grads = np.sum(diff * inv_dist2[:, :, np.newaxis], axis=1)
            
            # Project gradient onto tangent plane
            # p_i is unit vector
            # dot = sum(grads * points, axis=1, keepdims=True)
            dots = np.sum(grads * points, axis=1, keepdims=True)
            grads_tangent = grads - dots * points
            
            # Update
            points = points + lr * grads_tangent
            
            # Normalize
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            
        # Calculate score for this configuration
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max > 0:
            score = (d_min / d_max) ** 2
        else:
            score = 0.0
            
        if score > best_score:
            best_score = score
            best_points = points.copy()

    if best_points is None:
        # Fallback to random
        np.random.seed(42)
        best_points = np.random.randn(n, d)
        
    return best_points

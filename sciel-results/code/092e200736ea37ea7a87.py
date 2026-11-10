# EVOLVE-BLOCK-START
import numpy as np
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a gradient-based repulsion optimization on the unit sphere with multiple restarts.
    """
    n = 14
    d = 3
    deadline = time.time() + 500.0  # Leave margin for evaluation overhead
    
    best_points = None
    best_score = 0.0
    
    # Number of restarts to try within the time budget
    # Each restart involves a few thousand gradient steps, which is very fast in numpy
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialize points on the unit sphere
        # Use a fixed seed per restart to be deterministic, or random
        np.random.seed(restart * 12345 + 67890)
        points = np.random.randn(n, d)
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Optimization parameters
        steps = 2000
        # Adaptive learning rate
        alpha = 0.05
        
        for step in range(steps):
            if time.time() > deadline:
                break
                
            # Compute pairwise distances
            # dist[i, j] = ||p_i - p_j||^2
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            
            # Avoid division by zero for i=j
            np.fill_diagonal(dist_sq, 1.0)
            
            # Force: F_i = sum_j (p_i - p_j) / dist_ij^4
            # We want to maximize min dist, so we repel.
            # Gradient of sum(1/d^2) is -2 * sum (p_i-p_j)/d^4 ... wait.
            # Let's just use the repulsive force vector.
            # Force on i from j is proportional to (p_i - p_j) / dist^4
            # This minimizes sum(1/dist^2)
            
            # dist_sq is matrix of squared distances.
            # We need dist^4 = dist_sq^2
            # Force term for pair (i,j) on point i is (p_i - p_j) / dist_sq^2
            # Note: diff[i,j] is p_i - p_j.
            
            # We sum over j for each i.
            # F = diff / dist_sq^2 (broadcasting)
            # But we must sum over the axis of the other point.
            
            # dist_sq shape (n, n)
            # diff shape (n, n, d)
            
            # weights shape (n, n)
            weights = 1.0 / (dist_sq**2)
            np.fill_diagonal(weights, 0.0)
            
            # F_i = sum_j weights[i,j] * diff[i,j]
            forces = np.sum(weights[:, :, np.newaxis] * diff, axis=1)
            
            # Update points
            points += alpha * forces
            
            # Project back to unit sphere
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            # Avoid division by zero if a point collapses to origin (unlikely with repulsion)
            norms[norms < 1e-10] = 1.0
            points /= norms
            
            # Adaptive alpha: decrease if step is too large or oscillating?
            # Simple decay
            if step % 100 == 0 and step > 0:
                alpha *= 0.95
                
        # Evaluate the score for this configuration
        # Normalize points so that max distance is 1?
        # The metric is (d_min / d_max)^2.
        # Since we are on the unit sphere, d_max <= 2.
        # Let's compute d_min and d_max.
        
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # Get pairwise distances (upper triangle)
        iu = np.triu_indices(n, k=1)
        pairwise_dists = dists[iu]
        
        d_min = np.min(pairwise_dists)
        d_max = np.max(pairwise_dists)
        
        if d_max > 0:
            score = (d_min / d_max) ** 2
        else:
            score = 0.0
            
        if score > best_score:
            best_score = score
            best_points = points.copy()
            
    # If no points found (should not happen), return random
    if best_points is None:
        np.random.seed(42)
        best_points = np.random.randn(n, d)
        
    return best_points

# EVOLVE-BLOCK-END

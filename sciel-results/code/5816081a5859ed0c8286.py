import numpy as np

N = 150

def construct():
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Gradient Descent on the sphere
    # Energy E = sum_{i<j} log(1/d_ij)
    # Gradient w.r.t x_i: sum_{j != i} (x_j - x_i) / ||x_i - x_j||^2
    
    # We use a simple gradient descent with a decaying learning rate
    # and normalization to the sphere.
    
    # To improve convergence, we can use a larger number of steps
    # and a careful learning rate schedule.
    
    # Initial learning rate
    lr = 0.05
    decay = 0.9999
    
    # We run for a fixed number of steps. 
    # 20,000 steps is safe for 150 points.
    for _ in range(20000):
        # Compute pairwise differences
        # diff[i, j] = x_i - x_j
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # Avoid division by zero (should not happen with good init, but safe)
        # dist_sq[i, i] = 0
        np.fill_diagonal(dist_sq, 1.0) # Temporary placeholder, we mask later
        
        # Force term: (x_j - x_i) / dist^2
        # Note: diff is x_i - x_j. So we need -diff / dist^2
        # grad[i] = sum_j (x_j - x_i) / dist_ij^2
        # Let's compute the matrix of forces
        # inv_dist_sq = 1 / dist_sq
        # force_matrix[i, j] = (x_j - x_i) / dist_sq[i, j]
        # This is equivalent to -diff[i, j] / dist_sq[i, j]
        
        # We only want j != i
        mask = ~np.eye(N, dtype=bool)
        
        # Calculate gradient for each point
        # grad[i] = sum_{j != i} (x_j - x_i) / dist_sq[i, j]
        
        # Vectorized calculation:
        # diff is (N, N, 3)
        # dist_sq is (N, N)
        # We want to sum over j for each i.
        
        # term[i, j] = -diff[i, j] / dist_sq[i, j, None]
        # But dist_sq[i, i] is 0, so we mask it.
        
        # To avoid memory issues or NaNs, we set diagonal to 1 before division
        # and then zero out the diagonal in the result.
        
        dist_sq_safe = dist_sq.copy()
        np.fill_diagonal(dist_sq_safe, 1.0)
        
        # term[i, j] = (x_j - x_i) / dist_sq[i, j]
        # diff[i, j] = x_i - x_j
        # So we use -diff[i, j]
        
        forces = -diff / dist_sq_safe[:, :, np.newaxis]
        
        # Zero out diagonal
        for i in range(N):
            forces[i, i] = 0.0
            
        # Sum over j to get gradient for each i
        grad = np.sum(forces, axis=1)
        
        # Update points
        pts = pts + lr * grad
        
        # Normalize to unit sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        
        # Decay learning rate
        lr *= decay
        
    return pts

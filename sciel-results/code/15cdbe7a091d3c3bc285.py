"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Constructs 50 points on the unit sphere by minimizing a repulsive energy
    function (Riesz energy with s=2) using gradient descent on the sphere.
    This approach encourages a uniform distribution, which minimizes the
    covering radius.
    """
    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Ensure they are exactly on the sphere
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    # 2. Optimization: Gradient Descent with Repulsive Forces
    # Energy: E = sum_{i != j} 1 / ||p_i - p_j||^2
    # Gradient: grad_i E = -2 * sum_{j != i} (p_i - p_j) / ||p_i - p_j||^4
    
    # We use a learning rate schedule
    alpha = 0.01
    decay = 0.999
    
    for step in range(5000):
        # Compute pairwise distances
        # dist^2[i, j] = ||p_i - p_j||^2 = 2 - 2 * p_i . p_j
        dots = np.dot(points, points.T)
        dist_sq = 2.0 - 2.0 * dots
        np.fill_diagonal(dist_sq, 1.0) # Avoid division by zero, though we mask it
        
        # Compute forces
        # Force on i from j: -2 * (p_i - p_j) / dist_sq^2
        # We can vectorize this.
        # F_i = sum_j -2 * (p_i - p_j) / dist_sq[i,j]^2
        # Let's compute the sum of (p_i - p_j) / dist_sq[i,j]^2 for each i.
        
        # dist_sq_inv_sq = 1 / dist_sq^2
        dist_sq_inv_sq = 1.0 / (dist_sq ** 2)
        np.fill_diagonal(dist_sq_inv_sq, 0.0)
        
        # Sum over j for each i:
        # sum_j (p_i - p_j) / dist_sq[i,j]^2 = sum_j p_i / dist_sq[i,j]^2 - sum_j p_j / dist_sq[i,j]^2
        # = p_i * sum_j (1 / dist_sq[i,j]^2) - sum_j (p_j / dist_sq[i,j]^2)
        
        # term1: p_i * row_sum
        row_sums = np.sum(dist_sq_inv_sq, axis=1)
        term1 = points * row_sums[:, np.newaxis]
        
        # term2: sum_j (p_j / dist_sq[i,j]^2)
        # This is (dist_sq_inv_sq @ points)
        term2 = dist_sq_inv_sq @ points
        
        # Gradient of Energy w.r.t p_i is -2 * (term1 - term2)
        grad = -2.0 * (term1 - term2)
        
        # Project gradient onto tangent plane of the sphere
        # g_proj = g - (g . p_i) p_i
        dot_gp = np.sum(grad * points, axis=1, keepdims=True)
        grad_proj = grad - dot_gp * points
        
        # Update points
        points += alpha * grad_proj
        
        # Normalize points to stay on the sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points /= norms
        
        # Decay learning rate
        alpha *= decay
        
        # Optional: Check for convergence
        if alpha < 1e-8:
            break

    # Final normalization
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    return points
# EVOLVE-BLOCK-END

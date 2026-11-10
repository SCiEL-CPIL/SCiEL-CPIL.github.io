import numpy as np
from scipy.spatial.distance import pdist

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses deterministic Fibonacci sphere initialization followed by gradient descent 
    to minimize the Thomson energy (sum of inverse square distances), which effectively 
    maximizes the minimum distance between points on the sphere.
    """
    n = 14
    
    # 1. Initialize points using Fibonacci Sphere for uniform distribution
    points = np.zeros((n, 3))
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        r = np.sqrt(1 - z*z)
        theta = 2 * np.pi * i / golden_ratio
        points[i] = np.array([r * np.cos(theta), r * np.sin(theta), z])
        
    # Normalize to unit sphere
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    # 2. Gradient Descent to minimize Thomson Energy E = sum_{i<j} 1/d_ij^2
    # This pushes points apart, maximizing the minimum distance.
    lr = 0.01
    num_steps = 5000
    
    for step in range(num_steps):
        forces = np.zeros_like(points)
        
        # Calculate forces (negative gradient of energy)
        # dE/dp_i = sum_{j != i} 2 * (p_i - p_j) / d_ij^4
        for i in range(n):
            for j in range(i + 1, n):
                diff = points[i] - points[j]
                dist_sq = np.dot(diff, diff)
                if dist_sq < 1e-12:
                    continue
                dist = np.sqrt(dist_sq)
                # Force magnitude factor
                # Gradient of 1/d^2 is -2/d^3 * unit_vec. 
                # We want to minimize energy, so we move in direction of -grad.
                # -grad_i = sum 2/d^4 * (p_i - p_j) ... wait.
                # Let's check: E = sum 1/d^2. 
                # d(1/d^2)/dp_i = -2 * d^{-3} * d(d)/dp_i = -2 * d^{-3} * (p_i - p_j)/d = -2 * (p_i - p_j) / d^4.
                # So -grad = sum 2 * (p_i - p_j) / d^4.
                # This force pushes i away from j if i is "outside" relative to j?
                # If p_i - p_j points from j to i, and we add this to force, we push i further away from j. Correct.
                f_mag = 2.0 / (dist_sq * dist_sq)
                f_vec = diff * f_mag
                forces[i] += f_vec
                forces[j] -= f_vec
        
        # Project forces onto tangent space (remove radial component)
        for i in range(n):
            radial = np.dot(forces[i], points[i]) * points[i]
            forces[i] -= radial
            
        # Update points
        points += lr * forces
        
        # Re-normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        # Adaptive learning rate
        if step % 1000 == 0:
            lr *= 0.5
            
    # Final centering and normalization
    points -= np.mean(points, axis=0)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    return points

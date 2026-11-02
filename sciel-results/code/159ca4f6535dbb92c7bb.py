import numpy as np
import scipy.spatial
import time

N = 130

def construct():
    # Start with Fibonacci spiral for good initial distribution
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Generate a grid of test points to estimate the covering radius and gradients
    # We use a Fibonacci spiral for the grid as well, with more points
    num_grid = 20000
    i_grid = np.arange(num_grid) + 0.5
    z_grid = 1.0 - 2.0 * i_grid / num_grid
    phi_grid = np.pi * (1.0 + np.sqrt(5.0)) * i_grid
    r_grid = np.sqrt(1.0 - z_grid * z_grid)
    grid_points = np.column_stack([r_grid * np.cos(phi_grid), r_grid * np.sin(phi_grid), z_grid])
    
    # Optimization parameters
    num_iterations = 300
    alpha = 0.1
    repulsion_strength = 0.5
    attraction_strength = 1.0
    top_k_fraction = 0.05 # Consider the worst 5% of grid points for attraction
    
    for it in range(num_iterations):
        # 1. Compute distances from grid points to all configuration points
        # dist_grid[i, j] is distance between grid_point[i] and point[j]
        # Using dot product for speed: cos(d) = dot(p1, p2)
        dots = np.dot(grid_points, points.T) # shape (num_grid, N)
        # Clip for numerical stability
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots) # shape (num_grid, N)
        
        # Find nearest neighbor for each grid point
        min_dists = np.min(dists, axis=1) # shape (num_grid,)
        min_indices = np.argmin(dists, axis=1) # shape (num_grid,)
        
        # 2. Identify the "worst" grid points (largest covering distance)
        # We want to minimize the max, so we focus on the tail of the distribution
        threshold_idx = int(num_grid * (1.0 - top_k_fraction))
        # Get indices of the top_k largest distances
        # argsort in descending order
        sorted_indices = np.argsort(-min_dists)
        worst_grid_indices = sorted_indices[:threshold_idx]
        
        # 3. Compute forces
        forces = np.zeros_like(points)
        
        # Attraction towards the centers of the worst holes
        # For each worst grid point, pull its nearest neighbor towards it
        for idx in worst_grid_indices:
            g = grid_points[idx]
            nn_idx = min_indices[idx]
            p = points[nn_idx]
            # Vector from p to g
            vec = g - p
            # Project to tangent space at p to ensure movement stays on sphere
            # Tangent projection: v - (v.p)p
            proj = vec - np.dot(vec, p) * p
            forces[nn_idx] += attraction_strength * proj
        
        # Repulsion between points to prevent clustering
        # This acts like a pressure term in the density estimation
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                p_i = points[i]
                p_j = points[j]
                # Repulsion is stronger when points are close
                # Use a soft potential: 1 / (1 + dist^2) or similar
                d = np.linalg.norm(p_i - p_j)
                if d < 1e-6:
                    continue
                # Force direction: p_i - p_j (push away)
                # Magnitude: decay with distance
                # We use a simple inverse square law in 3D space projected to tangent
                vec = p_i - p_j
                proj = vec - np.dot(vec, p_i) * p_i
                # Scale by 1/d^2 to make it strong at close range
                force_mag = repulsion_strength / (d * d + 0.1) 
                forces[i] += force_mag * proj / np.linalg.norm(proj) if np.linalg.norm(proj) > 1e-6 else 0
        
        # 4. Update positions
        # Normalize forces to have unit magnitude to control step size
        for i in range(N):
            f_norm = np.linalg.norm(forces[i])
            if f_norm > 1e-10:
                step = alpha * forces[i] / f_norm
                # Project step to tangent space
                step = step - np.dot(step, points[i]) * points[i]
                points[i] = points[i] + step
                # Renormalize to sphere
                points[i] /= np.linalg.norm(points[i])
            else:
                # Small random perturbation if force is zero
                noise = np.random.randn(3) * 1e-4
                points[i] = points[i] + noise
                points[i] /= np.linalg.norm(points[i])
        
        # Decay alpha
        alpha *= 0.995
        
    return points

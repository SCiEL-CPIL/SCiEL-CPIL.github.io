"""Rewritten program for packing 200 equal circles in a unit square."""
import numpy as np
from scipy.spatial.distance import pdist

N = 200


def construct():
    """
    Construct centers of 200 equal non-overlapping circles in [-0.5, 0.5]^2
    maximizing the common radius.
    
    Strategy:
    1. Initialize with a Fibonacci lattice (sunflower pattern) for good initial spacing.
    2. Refine using a repulsive force-based simulation with constraints keeping
       points inside the square and preventing overlap.
    """
    # Step 1: Fibonacci lattice initialization
    # The sunflower pattern places points at:
    #   r_i = sqrt((i + 0.5) / N)
    #   theta_i = 2 * pi * (i + 0.5) * golden_ratio
    # Then scale to fit in [-0.5, 0.5]^2
    
    golden_ratio = (1 + np.sqrt(5)) / 2
    i = np.arange(N)
    r = np.sqrt((i + 0.5) / N)
    theta = 2 * np.pi * (i + 0.5) / golden_ratio
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # Scale to fit within [-0.5, 0.5]^2
    # The maximum radius in the Fibonacci lattice is 1.0, so scale by 0.5
    # But we need to ensure all points are inside the square.
    # The bounding circle has radius 1.0, but the square only fits a circle of radius 0.5/sqrt(2) ~ 0.354 if centered.
    # Actually, let's scale so that the max coordinate is 0.5.
    max_coord = np.max(np.abs(np.column_stack([x, y])))
    if max_coord > 0:
        scale = 0.5 / max_coord
        x *= scale
        y *= scale
    
    centers = np.column_stack([x, y])
    
    # Ensure all points are strictly inside the square
    centers = np.clip(centers, -0.4999, 0.4999)
    
    # Step 2: Repulsive force-based refinement
    # We want to maximize the minimum distance between any two points and from boundaries.
    # Use a simple iterative approach: for each point, move it in the direction that
    # increases the minimum distance to its nearest neighbor and to the boundaries.
    
    num_iterations = 500
    learning_rate = 0.01
    
    for iteration in range(num_iterations):
        # Compute all pairwise distances
        dists = pdist(centers, metric='euclidean')
        n = len(centers)
        dist_matrix = np.zeros((n, n))
        idx = np.triu_indices(n, k=1)
        dist_matrix[idx] = dists
        dist_matrix[idx[::-1]] = dists
        
        # Distance to boundaries
        # For each point, the distance to the nearest boundary
        # d_boundary = min(0.5 - x, 0.5 + x, 0.5 - y, 0.5 + y)
        d_boundary = np.min([
            0.5 - centers[:, 0],
            0.5 + centers[:, 0],
            0.5 - centers[:, 1],
            0.5 + centers[:, 1]
        ], axis=0)
        
        # For each point, find the nearest neighbor (by center-center distance)
        # and the boundary constraint
        # We want to maximize min(dists, 2*d_boundary) effectively
        # But in the force model, we treat it as:
        # - Repulsive force from other points
        # - Repulsive force from boundaries
        
        # Compute forces
        forces = np.zeros_like(centers)
        
        for i in range(n):
            # Force from other points
            # For each other point j, if dist(i,j) is small, push i away from j
            # Use inverse square law or similar
            for j in range(n):
                if i == j:
                    continue
                d = dist_matrix[i, j]
                if d < 1e-10:
                    # Coincident, random push
                    forces[i] += 0.01 * np.random.randn(2)
                    continue
                # Repulsive force proportional to 1/d^2 or similar
                # Only consider close neighbors to keep it efficient
                if d < 0.15:  # Only interact with close neighbors
                    force_magnitude = 1.0 / (d * d)
                    direction = (centers[i] - centers[j]) / d
                    forces[i] += force_magnitude * direction
            
            # Force from boundaries
            # Push away from nearest boundary
            x, y = centers[i]
            # Distance to each boundary
            d_right = 0.5 - x
            d_left = 0.5 + x
            d_top = 0.5 - y
            d_bottom = 0.5 + y
            
            # Find the nearest boundary
            min_d = min(d_right, d_left, d_top, d_bottom)
            
            # If close to boundary, push inward
            if min_d < 0.1:
                force_mag = 1.0 / (min_d * min_d + 1e-10)
                if min_d == d_right:
                    forces[i, 0] -= force_mag  # Push left (toward center)
                elif min_d == d_left:
                    forces[i, 0] += force_mag  # Push right
                elif min_d == d_top:
                    forces[i, 1] -= force_mag  # Push down
                elif min_d == d_bottom:
                    forces[i, 1] += force_mag  # Push up
        
        # Update positions with small step
        # Normalize forces to avoid large steps
        force_norms = np.linalg.norm(forces, axis=1)
        max_norm = np.max(force_norms)
        if max_norm > 0:
            forces = forces / max_norm * 0.01
        
        centers += learning_rate * forces
        
        # Project back into the square
        centers = np.clip(centers, -0.49999, 0.49999)
        
        # Reduce learning rate over time
        learning_rate *= 0.999
    
    # Final cleanup: ensure all points are inside the square
    centers = np.clip(centers, -0.49999, 0.49999)
    
    return centers

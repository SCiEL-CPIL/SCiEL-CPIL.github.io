"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_covering_radius(points):
    """
    Compute the covering radius by checking the convex hull facets.
    This is used for evaluation and potentially as a penalty in the optimization.
    """
    from scipy.spatial import ConvexHull
    points = np.asarray(points)
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi, []
    
    max_radius = 0.0
    worst_verts = []
    
    for simplex in hull.simplices:
        verts = points[simplex]
        if len(verts) == 3:
            e1 = verts[1] - verts[0]
            e2 = verts[2] - verts[0]
            pole = np.cross(e1, e2)
            pole_norm = np.linalg.norm(pole)
            if pole_norm < 1e-12:
                continue
            pole = pole / pole_norm
            centroid = np.mean(verts, axis=0)
            if np.dot(pole, centroid) < 0:
                pole = -pole
            cos_theta = np.clip(np.dot(verts[0], pole), -1.0, 1.0)
            theta = np.arccos(cos_theta)
            if theta > max_radius:
                max_radius = theta
                worst_verts = simplex
    return max_radius, worst_verts

def construct():
    # Random initialization to avoid symmetry biases of Fibonacci
    rng = np.random.default_rng(42)
    points = rng.standard_normal((N, 3))
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # Repulsive Force Optimization parameters
    n_iterations = 5000
    alpha = 0.05  # Initial step size
    decay = 0.9995
    
    # We use a repulsive force model: F_ij = k * (p_i - p_j) / ||p_i - p_j||^3
    # This is equivalent to potential V = 1/||p_i - p_j||
    # Projected onto tangent plane.
    
    for i in range(n_iterations):
        # Compute pairwise distances and forces vectorized
        # Diff: (N, N, 3)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        # Distances: (N, N)
        dist_sq = np.sum(diff * diff, axis=2)
        # Avoid division by zero
        dist_sq[dist_sq < 1e-10] = 1e-10
        
        # Force magnitude: 1 / dist^2 * unit_vector
        # F_ij = (p_i - p_j) / dist^3
        # dist = sqrt(dist_sq)
        # dist^3 = dist_sq * sqrt(dist_sq)
        dist = np.sqrt(dist_sq)
        dist_cubed = dist_sq * dist
        
        # Force matrix: (N, N, 3)
        # We want sum_j F_ij for each i
        # F_ij = diff_ij / dist_cubed_ij
        force = diff / dist_cubed[:, :, np.newaxis]
        
        # Sum forces for each point
        # Exclude self-interaction (diagonal is 0 anyway due to diff=0, but dist is clamped)
        # The diagonal elements are 0 / 1e-10 = 0.
        total_force = np.sum(force, axis=1)
        
        # Project force onto tangent plane at each point
        # Tangent projection: F_tan = F - (F . p) * p
        dot_fp = np.sum(total_force * points, axis=1, keepdims=True)
        tangent_force = total_force - dot_fp * points
        
        # Update positions
        points = points + alpha * tangent_force
        
        # Normalize
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        # Decay step size
        alpha *= decay
        
        # Optional: Add small noise to escape local minima
        if i % 100 == 50 and alpha > 0.001:
            noise = rng.standard_normal((N, 3)) * 0.01
            points = points + noise
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
            
    # Final normalization
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    return points
# EVOLVE-BLOCK-END

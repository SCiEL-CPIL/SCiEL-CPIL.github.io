import numpy as np
from scipy.spatial import cKDTree
import time

N = 50

def construct():
    # Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Local search
    # Sample points on the sphere for evaluation
    # Use a quasi-random sequence for better coverage
    num_samples = 50000
    # Generate random points on the sphere
    rng = np.random.default_rng(42)
    samples = rng.normal(size=(num_samples, 3))
    norms = np.linalg.norm(samples, axis=1, keepdims=True)
    samples = samples / norms
    
    # Precompute tree for samples? No, we need to query distance from samples to pts.
    # We will recompute the tree for pts in each iteration if we update pts.
    
    max_iter = 1000
    step_size = 0.1
    best_pts = pts.copy()
    best_radius = np.inf
    
    start_time = time.time()
    time_limit = 140  # Leave some margin for the 150s limit
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Build KDTree for current points
        tree = cKDTree(pts)
        
        # Find nearest point for each sample
        # dists is shape (num_samples, 1), indices is shape (num_samples, 1)
        dists, indices = tree.query(samples)
        
        # Find the sample with the maximum distance
        max_idx = np.argmax(dists)
        max_dist = dists[max_idx]
        
        if max_dist < best_radius:
            best_radius = max_dist
            best_pts = pts.copy()
            
        # If the radius is already very small, we can stop
        if max_dist < 18.3 * np.pi / 180.0: # 18.3 degrees in radians
            break
            
        # Identify the nearest configuration points to the worst sample
        worst_sample = samples[max_idx]
        # Find the 3 nearest configuration points to worst_sample
        # We can use the tree to query the worst_sample
        _, near_indices = tree.query(worst_sample, k=3)
        
        # Move these points towards worst_sample
        # The direction to move is the geodesic direction from the point to worst_sample
        for idx in near_indices:
            p = pts[idx]
            # Vector from p to worst_sample on the sphere
            # This is not simply worst_sample - p, but we can use the tangent vector
            # The direction on the sphere is the projection of (worst_sample - p) onto the tangent plane at p
            v = worst_sample - p
            # Project v onto the tangent plane at p: v_t = v - (v . p) p
            v_t = v - np.dot(v, p) * p
            # Normalize v_t
            norm_v_t = np.linalg.norm(v_t)
            if norm_v_t > 1e-10:
                v_t = v_t / norm_v_t
                # Move p in the direction of v_t
                new_p = p + step_size * v_t
                # Re-normalize to the unit sphere
                new_p = new_p / np.linalg.norm(new_p)
                pts[idx] = new_p
        
        # Decrease step size
        step_size *= 0.999
        
    return best_pts

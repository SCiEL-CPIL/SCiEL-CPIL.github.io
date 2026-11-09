import numpy as np
from scipy.spatial import cKDTree

def construct():
    N = 130
    
    # 1. Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Create a dense sampling of the sphere for Voronoi cell estimation
    # Using a Fibonacci spiral for the sampling points as well, but much denser
    M = 5000 # Number of sample points
    i_s = np.arange(M) + 0.5
    z_s = 1.0 - 2.0 * i_s / M
    phi_s = np.pi * (1.0 + np.sqrt(5.0)) * i_s
    r_s = np.sqrt(1.0 - z_s * z_s)
    samples = np.column_stack([r_s * np.cos(phi_s), r_s * np.sin(phi_s), z_s])
    
    # 3. Lloyd's Algorithm Iterations
    # We iterate a few times to refine the positions
    num_iters = 20
    
    for _ in range(num_iters):
        # Find nearest neighbor for each sample point
        tree = cKDTree(pts)
        # dists, idx = tree.query(samples, k=1)
        _, idx = tree.query(samples, k=1)
        
        # Compute new positions as centroids of Voronoi cells
        new_pts = np.zeros_like(pts)
        for i in range(N):
            # Get samples in the Voronoi cell of point i
            mask = (idx == i)
            if np.any(mask):
                centroid = np.mean(samples[mask], axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 1e-6:
                    new_pts[i] = centroid / norm
                else:
                    new_pts[i] = pts[i] # Fallback if centroid is origin
            else:
                # If no samples are closer to this point, keep it or move it slightly
                # This shouldn't happen with a good initial distribution and dense sampling
                new_pts[i] = pts[i]
        
        pts = new_pts
        
    # 4. Final refinement: Small random perturbations and local search?
    # Lloyd's algorithm usually gets close. Let's do a few steps of gradient descent
    # on the maximum distance to a dense set of points.
    
    # Re-compute nearest neighbors for the final check and potential fine-tuning
    # Actually, Lloyd's is quite good. Let's just return the result.
    
    # Ensure points are normalized
    norms = np.linalg.norm(pts, axis=1)
    pts = pts / norms[:, np.newaxis]
    
    return pts

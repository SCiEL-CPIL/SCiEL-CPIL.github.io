# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    Uses Lloyd's algorithm (Voronoi relaxation) on the sphere to find a near-optimal spherical code.
    """
    n = 14
    d = 3
    
    # Initialize points using Fibonacci sphere for good initial distribution
    points = np.zeros((n, d))
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        r = np.sqrt(1 - z**2)
        theta = 2 * np.pi * i / golden_ratio
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points[i] = [x, y, z]
    
    # Lloyd's algorithm iterations
    iterations = 100
    
    for _ in range(iterations):
        # Build a KD-tree of the current points for fast nearest neighbor search
        tree = cKDTree(points)
        
        # Generate a dense set of points on the sphere to represent the surface
        # We need to assign each point on the sphere to its nearest neighbor.
        # Instead of a fixed grid, we can use the current points and their neighbors,
        # but a better way for Lloyd's on a sphere is to sample many points,
        # assign them, compute centroids, and project back.
        
        # Sample 10000 points on the sphere
        n_samples = 10000
        np.random.seed(42) # For reproducibility in sampling, though Lloyd's is deterministic if init is
        # Actually, let's use a fixed set of sample points or random each time?
        # Standard Lloyd's uses the domain. For a sphere, we can sample uniformly.
        # To be deterministic and fast, let's use a fixed large sample set.
        
        # Generate fixed sample points on sphere (Fibonacci again, or just random once)
        # Let's generate random points once outside the loop? 
        # But the assignment changes, so we just need the sample points to be fixed.
        
        if not hasattr(min_max_dist_dim3_14, '_samples'):
            # Generate 20000 random points on sphere
            min_max_dist_dim3_14._samples = np.random.randn(n_samples, d)
            min_max_dist_dim3_14._samples /= np.linalg.norm(min_max_dist_dim3_14._samples, axis=1, keepdims=True)
        
        samples = min_max_dist_dim3_14._samples
        
        # Assign each sample point to the nearest neighbor in 'points'
        # distances, indices = tree.query(samples)
        # indices shape: (n_samples,)
        
        # To avoid empty cells, we can just compute the mean of assigned points.
        # If a cell is empty, the centroid is undefined. We can skip it or keep the old point.
        
        new_points = points.copy()
        
        # Efficient assignment and centroid calculation
        # We can use bincount or just loop if n is small (14 is small)
        # But n_samples is 10000.
        
        # Get indices
        _, indices = tree.query(samples)
        
        # Calculate centroids
        for i in range(n):
            mask = (indices == i)
            if np.any(mask):
                centroid = np.mean(samples[mask], axis=0)
                # Project to sphere
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    new_points[i] = centroid / norm
                # If norm is 0 (shouldn't happen), keep old point
            # If no points assigned, keep old point (new_points[i] is already points[i])
        
        points = new_points
        
    # The points are on the unit sphere.
    # The metric is scale invariant, so we can return them as is.
    # However, to be safe against any potential numerical issues with very small distances,
    # we can scale them slightly, but unit sphere is fine.
    
    return points

# EVOLVE-BLOCK-END

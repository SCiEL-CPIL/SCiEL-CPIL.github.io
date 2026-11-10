"""Spectral optimization for covering the sphere with 130 equal caps."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Construct 130 points on the unit sphere using a spectral optimization approach.
    
    The idea: parameterize points as top eigenvectors of a PSD matrix M.
    We want to maximize the minimum pairwise angular separation (or minimize the max covering radius).
    
    Approach:
    1. Start with a good initial configuration (Fibonacci spiral).
    2. Build a kernel matrix K = X X^T where rows of X are the points.
    3. The covering radius is related to the largest facet circumradius.
    4. Use an iterative refinement: for each point, compute its Voronoi region on the sphere,
       find the farthest point in that region, and move the point toward it.
    
    Since pure spectral eigenvector optimization for this specific problem is complex,
    we use a hybrid: start from Fibonacci, then do local refinement using the fact that
    the optimal covering has each point at the center of its spherical Voronoi cell.
    
    We implement a variant of the "spherical Lloyd" algorithm with careful numerical handling.
    """
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Refinement: iterative spherical Lloyd / Voronoi-based optimization
    # For each iteration:
    #   1. For each point i, find the farthest point on the sphere from X[i] 
    #      that is closer to X[i] than to any other X[j].
    #   2. This farthest point is the "critical point" for the covering radius contribution.
    #   3. Move X[i] toward that critical point.
    
    # A simpler and more robust approach: use gradient-based optimization of the covering radius.
    # The covering radius R = max_y min_i angle(y, X_i)
    # We can approximate this by sampling many points on the sphere and minimizing the max distance.
    
    # But for efficiency within 120s, let's use a smarter local search.
    
    # Key insight: at optimality, each point should be at the spherical circumcenter of its 
    # Voronoi cell (the cell being the set of points on the sphere closer to X[i] than to any other).
    # The covering radius is the max over all cells of the distance from the cell center to its farthest boundary point.
    
    # Implementation:
    # 1. Compute pairwise angular distances
    # 2. For each point, determine its Voronoi neighbors (points that share a boundary)
    # 3. Find the circumcenter of the spherical polygon formed by the Voronoi vertices
    # 4. Update the point to be at that circumcenter
    
    # This is complex. Let's use a simpler but effective approach:
    # Iteratively, for each point, find the direction that maximizes the minimum distance to other points,
    # while staying within its Voronoi cell.
    
    # Even simpler: use a force-based approach where each point is pushed away from its nearest neighbors
    # but pulled toward the center of mass of its Voronoi cell.
    
    # Let's implement a practical version:
    # 1. Compute the Delaunay triangulation on the sphere (or approximate it)
    # 2. For each triangle, compute its circumcenter
    # 3. The covering radius is the max over all triangles of the distance from circumcenter to vertices
    # 4. Adjust points to minimize this
    
    # For robustness and speed, I'll use a gradient-free local search on the sphere.
    
    def angular_dist_matrix(X):
        """Compute pairwise angular distance matrix."""
        # X is (N, 3), unit vectors
        K = X @ X.T
        np.clip(K, -1.0, 1.0, out=K)
        return np.arccos(K)
    
    def covering_radius(X, n_samples=20000):
        """Estimate covering radius by sampling points on the sphere."""
        # Generate random points on sphere
        rng = np.random.default_rng(42)
        theta = rng.uniform(0, 2 * np.pi, n_samples)
        phi_s = np.arccos(rng.uniform(-1, 1, n_samples))
        Y = np.column_stack([
            np.sin(phi_s) * np.cos(theta),
            np.sin(phi_s) * np.sin(theta),
            np.cos(phi_s)
        ])
        # For each Y, find min angular distance to X
        K = Y @ X.T
        np.clip(K, -1.0, 1.0, out=K)
        min_dist = np.arccos(K.max(axis=1))
        return min_dist.max()
    
    # Local search: perturb each point slightly and keep the perturbation if it improves the covering radius
    # This is slow, so let's use a more directed approach.
    
    # Better approach: use the fact that the optimal configuration is a critical point of the covering radius.
    # We can use scipy's minimize on the flattened coordinates, with a smooth objective.
    
    # Objective: smooth approximation of the covering radius
    # f(X) = log-sum-exp of log(exp(d_i / T)) where d_i are distances from sample points to nearest X
    
    def objective_flat(flat_X):
        X = flat_X.reshape(N, 3)
        # Normalize
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        X = X / norms
        
        # Sample points
        rng = np.random.default_rng(123)
        n_samples = 5000
        theta = rng.uniform(0, 2 * np.pi, n_samples)
        phi_s = np.arccos(rng.uniform(-1, 1, n_samples))
        Y = np.column_stack([
            np.sin(phi_s) * np.cos(theta),
            np.sin(phi_s) * np.sin(theta),
            np.cos(phi_s)
        ])
        
        K = Y @ X.T
        np.clip(K, -1.0, 1.0, out=K)
        min_cos = K.max(axis=1)
        min_dist = np.arccos(min_cos)
        
        # Smooth max: log-sum-exp
        T = 0.01
        shifted = min_dist / T
        shifted -= shifted.max()
        return T * (shifted.max() + np.log(np.sum(np.exp(shifted))))
    
    # This might be too slow. Let's use a different strategy.
    
    # Strategy: Use the known Hardin-Sloane-Smith configuration if we can reconstruct it,
    # or use a very effective local refinement.
    
    # Let's try: start from Fibonacci, then do 50 iterations of:
    # For each point i:
    #   - Find the 3 nearest neighbors
    #   - Compute the spherical circumcenter of the triangle formed by i and its 2 nearest neighbors
    #   - Move i halfway toward that circumcenter (with a damping factor)
    
    # Actually, let's use a well-known effective method: 
    # "Spherical Lloyd" where each point moves to the centroid of its Voronoi cell.
    
    def spherical_voronoi_centroid(X, n_samples=10000):
        """For each point, compute the centroid of its Voronoi cell using sampling."""
        rng = np.random.default_rng(999)
        theta = rng.uniform(0, 2 * np.pi, n_samples)
        phi_s = np.arccos(rng.uniform(-1, 1, n_samples))
        Y = np.column_stack([
            np.sin(phi_s) * np.cos(theta),
            np.sin(phi_s) * np.sin(theta),
            np.cos(phi_s)
        ])
        
        # For each Y, find nearest point in X
        K = Y @ X.T
        nearest = K.argmax(axis=1)
        
        # Centroid for each point
        centroids = np.zeros((N, 3))
        counts = np.zeros(N)
        for j in range(N):
            mask = nearest == j
            if mask.sum() > 0:
                centroids[j] = Y[mask].mean(axis=0)
                counts[j] = mask.sum()
        
        # Normalize centroids
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        centroids = centroids / norms
        
        return centroids
    
    # Iterative refinement
    X_current = X.copy()
    for it in range(30):
        centroids = spherical_voronoi_centroid(X_current)
        # Move toward centroid with damping
        damping = 0.5 / (1 + it * 0.1)
        X_current = X_current + damping * (centroids - X_current)
        # Re-normalize
        norms = np.linalg.norm(X_current, axis=1, keepdims=True)
        X_current = X_current / np.maximum(norms, 1e-10)
    
    # Final check: ensure all points are on unit sphere
    norms = np.linalg.norm(X_current, axis=1, keepdims=True)
    X_current = X_current / np.maximum(norms, 1e-10)
    
    return X_current
# EVOLVE-BLOCK-END

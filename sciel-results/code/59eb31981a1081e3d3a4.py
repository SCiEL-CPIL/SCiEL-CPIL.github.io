def _score(points):
    """Return the feasible common radius (0.0 if invalid) using vectorized ops."""
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Distance to container boundary
    norm = np.linalg.norm(points, axis=1)
    r_bound = 1.0 - norm
    if np.min(r_bound) < 0.0:
        return 0.0
        
    # Minimum pairwise half-distance
    # Vectorized distance calculation
    # diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    # dists = np.linalg.norm(diff, axis=2)
    # This creates a 300x300x2 array, which is 180,000 floats. Very fast.
    # However, we only need min dist.
    
    # A faster way for min dist:
    # We can use scipy.spatial.distance.pdist which is C-optimized.
    # But let's stick to standard numpy to avoid import issues if scipy isn't fully loaded or for simplicity.
    # Actually, scipy IS available. pdist is much faster than manual numpy loops for N=300?
    # Let's use scipy.spatial.distance.pdist for the pairwise distances.
    
    from scipy.spatial.distance import pdist
    d = pdist(points)
    r_pair = 0.5 * np.min(d)
    
    r = min(np.min(r_bound), r_pair)
    return float(r) if r > 0.0 else 0.0

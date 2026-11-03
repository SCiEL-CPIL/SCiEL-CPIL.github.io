def _clamp_bary(bary):
    """Clamp barycentric coordinates to the valid region u>=0, v>=0, u+v<=1."""
    bary = np.clip(bary, 0.0, None)
    s = bary[:, 0] + bary[:, 1]
    mask = s > 1.0
    if np.any(mask):
        bary[mask] = bary[mask] / s[mask, None]
    
    # Shrink slightly towards the centroid (1/3, 1/3) to ensure strict interior
    # This prevents floating point errors from pushing points outside the boundary
    eps = 1e-6
    centroid = np.array([1.0/3.0, 1.0/3.0])
    bary = centroid + (1.0 - eps) * (bary - centroid)
    
    # Re-clamp just in case the shrink moved it outside (it shouldn't, but safe)
    bary = np.clip(bary, 0.0, None)
    s = bary[:, 0] + bary[:, 1]
    mask = s > 1.0
    if np.any(mask):
        bary[mask] = bary[mask] / s[mask, None]
        
    return bary

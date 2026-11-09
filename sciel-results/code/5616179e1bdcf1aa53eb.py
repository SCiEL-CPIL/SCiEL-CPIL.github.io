# EVOLVE-BLOCK-START
import numpy as np


def equidistant_points_polygon20() -> np.ndarray:
    """
    Construct a strictly convex 20-gon with vertices in cyclic order that
    maximizes the minimum per-vertex equidistance score. Uses an alternating
    two-radius circular construction to make the four nearest distances
    from each vertex as equal as possible.
    """
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    # Alternate between two radii to equalize nearest-neighbor distances
    r1 = 1.0
    r2 = 1.35
    radii = np.array([r1 if i % 2 == 0 else r2 for i in range(n)])
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    polygon = np.stack([x, y], axis=1)
    
    # Normalize: subtract centroid, divide by mean distance to centroid
    centroid = polygon.mean(axis=0)
    polygon = polygon - centroid
    dists_to_centroid = np.linalg.norm(polygon, axis=1)
    mean_dist = dists_to_centroid.mean()
    if mean_dist > 0.01:
        polygon = polygon / mean_dist
    
    return polygon


# EVOLVE-BLOCK-END

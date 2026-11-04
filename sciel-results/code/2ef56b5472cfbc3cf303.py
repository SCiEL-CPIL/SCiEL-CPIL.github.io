import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130
DEADLINE = 130.0  # 20s margin

def _facet_circumradii(points):
    """Return array of circumradii (in radians) for each hull facet."""
    hull = ConvexHull(points)
    radii = []
    for simplex in hull.simplices:
        a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            radii.append(0.0)
            continue
        n = n / n_norm
        # Pick outward direction: dot(n, a) should be positive for outward
        if np.dot(n, a) < 0:
            n = -n
        # Circumradius
        cos_r = np.abs(np.dot(n, a))
        cos_r = np.clip(cos_r, -1, 1)
        r = np.arccos(cos_r)
        radii.append(r)
    return np.array(radii)

def _covering_radius(points):
    radii = _facet_circumradii(points)
    return np.max(radii)

def construct():
    start_time = time.time()
    
    # Initial: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    # Local optimization: iteratively adjust points to reduce max circumradius
    # Use a simple approach: for the worst facet, nudge its vertices
    step_size = 0.01
    for iteration in range(5000):
        if time.time() - start_time > DEADLINE:
            break
        
        radii = _facet_circumradii(pts)
        max_idx = np.argmax(radii)
        if radii[max_idx] < 1e-10:
            break
        
        # Find the facet
        hull = ConvexHull(pts)
        simplex = hull.simplices[max_idx]
        a_idx, b_idx, c_idx = simplex
        
        # Move each vertex slightly away from the circumcenter direction
        a, b, c = pts[a_idx], pts[b_idx], pts[c_idx]
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        n = n / n_norm
        if np.dot(n, a) < 0:
            n = -n
        
        # The circumcenter is n. Move a, b, c slightly in the direction that reduces their distance to n
        # Actually, we want to reduce the circumradius, which means moving a, b, c closer to n
        # But they must stay on the unit sphere.
        # Gradient: to reduce arccos(n·a), we want to increase n·a, so move a toward n.
        for idx in [a_idx, b_idx, c_idx]:
            p = pts[idx]
            # Move p slightly toward n
            p_new = p + step_size * (n - np.dot(p, n) * n)
            p_new = p_new / np.linalg.norm(p_new)
            pts[idx] = p_new
        
        cr = _covering_radius(pts)
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
        else:
            # Revert
            pts = best_pts.copy()
            step_size *= 0.95
    
    # Ensure all points are on unit sphere
    norms = np.linalg.norm(best_pts, axis=1)
    best_pts = best_pts / norms[:, None]
    
    return best_pts

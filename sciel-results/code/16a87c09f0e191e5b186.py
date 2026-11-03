"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _compute_covering_radius(pts):
    """Compute the covering radius by finding the largest circumradius of hull facets."""
    try:
        hull = ConvexHull(pts)
        max_r = 0.0
        for simplex in hull.simplices:
            tri = pts[simplex]
            # Compute circumcenter of triangle on unit sphere
            a, b, c = tri[0], tri[1], tri[2]
            # Using vector formula for circumcenter of triangle
            d = 2.0 * (a[0] * (b[1] * c[2] - b[2] * c[1]) +
                       b[0] * (c[1] * a[2] - c[2] * a[1]) +
                       c[0] * (a[1] * b[2] - a[2] * b[1]))
            if abs(d) < 1e-12:
                continue
            ux = ((a[0]**2 + a[1]**2 + a[2]**2) * (b[1] * c[2] - b[2] * c[1]) -
                  (b[0]**2 + b[1]**2 + b[2]**2) * (a[1] * c[2] - a[2] * c[1]) +
                  (c[0]**2 + c[1]**2 + c[2]**2) * (a[1] * b[2] - a[2] * b[1])) / d
            uy = ((b[0]**2 + b[1]**2 + b[2]**2) * (a[2] * c[0] - a[0] * c[2]) -
                  (a[0]**2 + a[1]**2 + a[2]**2) * (b[2] * c[0] - b[0] * c[2]) +
                  (c[0]**2 + c[1]**2 + c[2]**2) * (b[2] * a[0] - b[0] * a[2])) / d
            uz = ((a[0]**2 + a[1]**2 + a[2]**2) * (b[0] * c[1] - b[1] * c[0]) -
                  (b[0]**2 + b[1]**2 + b[2]**2) * (a[0] * c[1] - a[1] * c[0]) +
                  (c[0]**2 + c[1]**2 + c[2]**2) * (a[0] * b[1] - a[1] * b[0])) / d
            center = np.array([ux, uy, uz])
            norm_c = np.linalg.norm(center)
            if norm_c < 1e-12:
                continue
            center = center / norm_c
            # Check that this circumcenter is on the correct side (outside the triangle)
            # Compute the angular distance from center to each vertex
            dists = np.arccos(np.clip(np.sum(pts[simplex] * center, axis=1), -1.0, 1.0))
            r = np.max(dists)
            if r > max_r:
                max_r = r
        return np.degrees(max_r)
    except Exception:
        return 180.0


def _find_worst_vertex_for_point(pts, point_idx):
    """Find the circumcenter of the largest facet adjacent to the given point."""
    try:
        hull = ConvexHull(pts)
        # Find all facets that include this point
        best_r = 0.0
        best_center = None
        for simplex in hull.simplices:
            if point_idx in simplex:
                tri = pts[simplex]
                a, b, c = tri[0], tri[1], tri[2]
                d = 2.0 * (a[0] * (b[1] * c[2] - b[2] * c[1]) +
                           b[0] * (c[1] * a[2] - c[2] * a[1]) +
                           c[0] * (a[1] * b[2] - a[2] * b[1]))
                if abs(d) < 1e-12:
                    continue
                ux = ((a[0]**2 + a[1]**2 + a[2]**2) * (b[1] * c[2] - b[2] * c[1]) -
                      (b[0]**2 + b[1]**2 + b[2]**2) * (a[1] * c[2] - a[2] * c[1]) +
                      (c[0]**2 + c[1]**2 + c[2]**2) * (a[1] * b[2] - a[2] * b[1])) / d
                uy = ((b[0]**2 + b[1]**2 + b[2]**2) * (a[2] * c[0] - a[0] * c[2]) -
                      (a[0]**2 + a[1]**2 + a[2]**2) * (b[2] * c[0] - b[0] * c[2]) +
                      (c[0]**2 + c[1]**2 + c[2]**2) * (b[2] * a[0] - b[0] * a[2])) / d
                uz = ((a[0]**2 + a[1]**2 + a[2]**2) * (b[0] * c[1] - b[1] * c[0]) -
                      (b[0]**2 + b[1]**2 + b[2]**2) * (a[0] * c[1] - a[1] * c[0]) +
                      (c[0]**2 + c[1]**2 + c[2]**2) * (a[0] * b[1] - a[1] * b[0])) / d
                center = np.array([ux, uy, uz])
                norm_c = np.linalg.norm(center)
                if norm_c < 1e-12:
                    continue
                center = center / norm_c
                dists = np.arccos(np.clip(np.sum(pts[simplex] * center, axis=1), -1.0, 1.0))
                r = np.max(dists)
                if r > best_r:
                    best_r = r
                    best_center = center
        return best_center
    except Exception:
        return None


def _move_toward_worst_vertex(pts, step_size=0.1):
    """Move each point toward the circumcenter of its largest adjacent facet."""
    new_pts = pts.copy()
    for i in range(len(pts)):
        center = _find_worst_vertex_for_point(pts, i)
        if center is not None:
            # Move point i toward the center
            # Use spherical interpolation (slerp)
            dot = np.clip(np.dot(pts[i], center), -1.0, 1.0)
            theta = np.arccos(dot)
            if theta > 1e-12:
                # Slerp toward center
                t = min(step_size, theta)
                sin_theta = np.sin(theta)
                new_pt = (np.sin((1 - t / theta) * theta) / sin_theta) * pts[i] + \
                         (np.sin(t / theta * theta) / sin_theta) * center
                new_pts[i] = new_pt
            else:
                new_pts[i] = center
    return _normalize(new_pts)


def construct():
    # Golden-angle (Fibonacci) spiral points on the unit sphere: a good starting point.
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    best_pts = pts.copy()
    best_radius = _compute_covering_radius(pts)
    
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    # Adaptive step size optimization
    step_size = 0.1
    for iteration in range(2000):
        if time.time() > deadline:
            break
        
        new_pts = _move_toward_worst_vertex(pts, step_size)
        new_radius = _compute_covering_radius(new_pts)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_pts = new_pts.copy()
            pts = new_pts
            # Decrease step size slightly for refinement
            step_size = max(0.01, step_size * 0.95)
        else:
            # If no improvement, try smaller step
            step_size = max(0.01, step_size * 0.8)
            if step_size < 0.01:
                break
    
    # Final polish pass with smaller steps
    if time.time() < deadline - 5.0:
        for polish_iter in range(500):
            if time.time() > deadline:
                break
            new_pts = _move_toward_worst_vertex(best_pts, 0.01)
            new_radius = _compute_covering_radius(new_pts)
            if new_radius < best_radius:
                best_radius = new_radius
                best_pts = new_pts
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END

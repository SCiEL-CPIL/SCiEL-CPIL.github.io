"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import math

TIME_LIMIT = 130.0  # safety margin before 150s kill

def _covering_radius_degrees(points):
    """Compute the covering radius in degrees for a set of points on the unit sphere."""
    try:
        hull = ConvexHull(points)
        max_r = 0.0
        for simplex in hull.simplices:
            # Get the facet vertices
            facet_pts = points[simplex]
            # Compute the circumcenter of the facet (triangle)
            # For a triangle on the unit sphere, the circumradius of the spherical cap
            # is the angle from the origin to the circumcenter of the triangle projected
            # onto the sphere, which equals the angle subtended by the circumradius of the triangle.
            # Actually, the covering radius for a facet is the angular distance from the 
            # facet's circumcenter (on the sphere) to any vertex of the facet.
            # The circumcenter of a triangle in 3D: find the point equidistant from all 3 vertices.
            a, b, c = facet_pts[0], facet_pts[1], facet_pts[2]
            # Circumcenter of triangle in 3D
            d = 2.0 * (a[0] * (b[1] * c[2] - b[2] * c[1]) -
                        a[1] * (b[0] * c[2] - b[2] * c[0]) +
                        a[2] * (b[0] * c[1] - b[1] * c[0]))
            if abs(d) < 1e-15:
                continue
            sa2 = b[0]**2 + b[1]**2 + b[2]**2
            sb2 = c[0]**2 + c[1]**2 + c[2]**2
            sc2 = a[0]**2 + a[1]**2 + a[2]**2
            ux = (sa2 * (b[1] * c[2] - b[2] * c[1]) -
                  sb2 * (a[1] * c[2] - a[2] * c[1]) +
                  sc2 * (a[1] * b[2] - a[2] * b[1])) / d
            uy = (sa2 * (b[2] * c[0] - b[0] * c[2]) -
                  sb2 * (a[2] * c[0] - a[0] * c[2]) +
                  sc2 * (a[2] * b[0] - a[0] * b[2])) / d
            uz = (sa2 * (b[0] * c[1] - b[1] * c[0]) -
                  sb2 * (a[0] * c[1] - a[1] * c[0]) +
                  sc2 * (a[0] * b[1] - a[1] * b[0])) / d
            # The circumcenter of the triangle on the unit sphere is in the direction of (ux, uy, uz)
            # The angular radius is the angle between this direction and any vertex
            uc = np.array([ux, uy, uz])
            uc_norm = np.linalg.norm(uc)
            if uc_norm < 1e-15:
                continue
            uc = uc / uc_norm
            # Angle between uc and a (any vertex)
            dot = np.clip(np.dot(uc, a), -1.0, 1.0)
            angle = math.acos(dot)
            if angle > max_r:
                max_r = angle
        # Also check that no point is inside a closed hemisphere (validity)
        # The covering radius should be < 90 degrees for a valid configuration
        return math.degrees(max_r)
    except Exception:
        return 90.0  # fallback


def _fibonacci_spiral(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms


def _local_search(points, max_iter=200, step_init=0.05, cooling=0.98):
    """Local search: gradient-free optimization of covering radius."""
    points = _normalize(points.copy())
    best = _covering_radius_degrees(points)
    step = step_init
    for it in range(max_iter):
        if time.time() > TIME_LIMIT - 10:
            break
        improved = False
        for i in range(N):
            if time.time() > TIME_LIMIT - 10:
                break
            orig = points[i].copy()
            # Try moving point i in random directions
            for trial in range(8):
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
                new_pt = orig + step * direction
                new_pt /= np.linalg.norm(new_pt)
                points[i] = new_pt
                cr = _covering_radius_degrees(points)
                if cr < best:
                    best = cr
                    improved = True
                    break
            points[i] = orig
        if not improved:
            step *= cooling
            if step < 1e-6:
                break
    return _normalize(points), best


def construct():
    start_time = time.time()
    
    # Start with Fibonacci spiral
    best_points = _fibonacci_spiral(N)
    best_cr = _covering_radius_degrees(best_points)
    
    # Multiple restarts with perturbation + local search
    np.random.seed(42)
    n_restarts = 30
    for restart in range(n_restarts):
        if time.time() > TIME_LIMIT - 20:
            break
        
        # Create perturbed initial configuration
        if restart == 0:
            pts = best_points.copy()
        else:
            pts = _fibonacci_spiral(N)
            # Add random perturbation
            perturbation_scale = 0.1 + 0.3 * (restart / n_restarts)
            noise = np.random.randn(N, 3) * perturbation_scale
            pts = _normalize(pts + noise)
        
        # Local search
        pts, cr = _local_search(pts, max_iter=100, step_init=0.05, cooling=0.97)
        
        if cr < best_cr:
            best_cr = cr
            best_points = pts.copy()
    
    # Final polish with smaller steps
    if time.time() < TIME_LIMIT - 15:
        best_points, best_cr = _local_search(best_points, max_iter=200, step_init=0.02, cooling=0.98)
    
    return _normalize(best_points)
# EVOLVE-BLOCK-END

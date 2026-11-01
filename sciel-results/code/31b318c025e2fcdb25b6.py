"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms


def _covering_radius(pts):
    """Compute the covering radius in radians using convex hull facets."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: dense grid
        n_grid = 200
        theta = np.linspace(0, np.pi, n_grid)
        phi = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
        TH, PH = np.meshgrid(theta, phi)
        gx = np.sin(TH) * np.cos(PH)
        gy = np.sin(TH) * np.sin(PH)
        gz = np.cos(TH)
        grid = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
        dists = np.arccos(np.clip(grid @ pts.T, -1, 1))
        return np.max(np.min(dists, axis=1))
    
    max_rad = 0.0
    for simplex in hull.simplices:
        tri = pts[simplex]
        # Find circumcenter of the triangle on the sphere
        a, b, c = tri[0], tri[1], tri[2]
        # Solve for point x on sphere equidistant from a, b, c
        # x . a = x . b = x . c = cos(r)
        # Use the fact that x is in the plane of a,b,c: x = alpha*a + beta*b + gamma*c
        # and |x| = 1, x.a = x.b = x.c
        # Alternative: find normal to plane, then circumcenter in plane
        edge1 = b - a
        edge2 = c - a
        normal = np.cross(edge1, edge2)
        n_norm = np.linalg.norm(normal)
        if n_norm < 1e-12:
            continue
        normal = normal / n_norm
        
        # Circumcenter in the plane of a,b,c
        # Solve: (x - a) . (b - a) = |b-a|^2/2, (x - a) . (c - a) = |c-a|^2/2
        # x = a + s*(b-a) + t*(c-a)
        # (s*(b-a) + t*(c-a)) . (b-a) = |b-a|^2/2
        # (s*(b-a) + t*(c-a)) . (c-a) = |c-a|^2/2
        d1 = edge1.dot(edge1)
        d2 = edge2.dot(edge2)
        d12 = edge1.dot(edge2)
        rhs1 = d1 / 2.0
        rhs2 = d2 / 2.0
        det = d1 * d2 - d12 * d12
        if abs(det) < 1e-12:
            continue
        s = (rhs1 * d2 - rhs2 * d12) / det
        t = (rhs2 * d1 - rhs1 * d12) / det
        x = a + s * edge1 + t * edge2
        x_norm = np.linalg.norm(x)
        if x_norm < 1e-12:
            continue
        x = x / x_norm
        # Ensure x is on the correct side (same hemisphere as normal or opposite)
        # The circumcenter should be the point on the sphere in the cap
        if x.dot(normal) < 0:
            x = -x
        # Check if x is actually the farthest point in this facet's cap
        # The covering radius for this facet is the angle from x to a (or b or c)
        rad = np.arccos(np.clip(np.dot(x, a), -1, 1))
        if rad > max_rad:
            max_rad = rad
    
    # Also check: the worst point might be at a vertex if the hull is degenerate
    # But for a good covering, the worst point is at a facet circumcenter
    return max_rad


def _fibonacci_sphere(n, offset=0.0):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    v = rng.standard_normal((n, 3))
    return _normalize(v)


def _refine_from_worst(pts, worst_pt, step_size=0.01):
    """Move the nearest point(s) toward the worst point."""
    pts = pts.copy()
    # Find the 3 nearest points to worst_pt
    dists = np.arccos(np.clip(pts @ worst_pt, -1, 1))
    idx = np.argsort(dists)[:3]
    for i in idx:
        # Move point i toward worst_pt
        direction = worst_pt - pts[i]
        norm_dir = np.linalg.norm(direction)
        if norm_dir < 1e-10:
            continue
        direction = direction / norm_dir
        pts[i] = pts[i] + step_size * direction
        pts[i] = _normalize(pts[i].reshape(1, -1)).ravel()
    return _normalize(pts)


def _local_optimize(pts, max_iters=200, step_size=0.005):
    """Local optimization: iteratively find worst point and move nearest sites."""
    pts = _normalize(pts.copy())
    best_radius = _covering_radius(pts)
    
    for it in range(max_iters):
        # Find worst point
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        worst_rad = 0.0
        worst_pt = None
        for simplex in hull.simplices:
            tri = pts[simplex]
            a, b, c = tri[0], tri[1], tri[2]
            edge1 = b - a
            edge2 = c - a
            normal = np.cross(edge1, edge2)
            n_norm = np.linalg.norm(normal)
            if n_norm < 1e-12:
                continue
            normal = normal / n_norm
            
            d1 = edge1.dot(edge1)
            d2 = edge2.dot(edge2)
            d12 = edge1.dot(edge2)
            det = d1 * d2 - d12 * d12
            if abs(det) < 1e-12:
                continue
            s = (d1/2.0 * d2 - d2/2.0 * d12) / det
            t = (d2/2.0 * d1 - d1/2.0 * d12) / det
            x = a + s * edge1 + t * edge2
            x_norm = np.linalg.norm(x)
            if x_norm < 1e-12:
                continue
            x = x / x_norm
            if x.dot(normal) < 0:
                x = -x
            rad = np.arccos(np.clip(np.dot(x, a), -1, 1))
            if rad > worst_rad:
                worst_rad = rad
                worst_pt = x
        
        if worst_pt is None:
            break
        
        if worst_rad < best_radius * 0.999:
            best_radius = worst_rad
            best_pts = pts.copy()
        
        # Try moving
        new_pts = _refine_from_worst(pts, worst_pt, step_size)
        new_rad = _covering_radius(new_pts)
        if new_rad < best_radius:
            pts = new_pts
            best_radius = new_rad
            best_pts = pts.copy()
        else:
            # Try smaller step
            new_pts = _refine_from_worst(pts, worst_pt, step_size * 0.5)
            new_rad = _covering_radius(new_pts)
            if new_rad < best_radius:
                pts = new_pts
                best_radius = new_rad
                best_pts = pts.copy()
            else:
                break
    
    return best_pts, best_radius


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 130s deadline, 20s safety margin
    
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_radius = np.inf
    
    # Multi-start with different initializations
    starts = []
    # Fibonacci spiral
    starts.append(_fibonacci_sphere(N))
    # Slightly perturbed Fibonacci
    for k in range(3):
        base = _fibonacci_sphere(N)
        pert = rng.standard_normal((N, 3)) * 0.1
        starts.append(_normalize(base + pert))
    # Random starts
    for k in range(5):
        starts.append(_random_sphere(N, rng))
    # Golden ratio spiral with different offset
    starts.append(_fibonacci_sphere(N, offset=np.pi/3))
    starts.append(_fibonacci_sphere(N, offset=np.pi/7))
    
    for start_idx, init_pts in enumerate(starts):
        if time.time() > deadline - 10.0:
            break
        
        # Local optimization
        refined_pts, radius = _local_optimize(init_pts, max_iters=150, step_size=0.004)
        
        if radius < best_radius:
            best_radius = radius
            best_pts = refined_pts.copy()
        
        # If we have good time, do a second pass with smaller steps
        if time.time() < deadline - 30.0 and radius < best_radius * 1.01:
            refined_pts2, radius2 = _local_optimize(refined_pts, max_iters=100, step_size=0.001)
            if radius2 < best_radius:
                best_radius = radius2
                best_pts = refined_pts2.copy()
    
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END

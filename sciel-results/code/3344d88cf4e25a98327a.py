"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _spherical_circumradius(v1, v2, v3):
    """
    Compute the spherical circumradius of the triangle formed by v1, v2, v3 on the unit sphere.
    Returns the angle in radians.
    """
    # Directions of perpendicular bisectors in the tangent space?
    # c is parallel to (v1 - v2) x (v2 - v3)
    d1 = v1 - v2
    d2 = v2 - v3
    c = np.cross(d1, d2)
    norm_c = np.linalg.norm(c)
    if norm_c < 1e-10:
        # Degenerate triangle
        return np.pi / 2
    c = c / norm_c
    
    # The circumcenter on the sphere is c or -c.
    # We want the one inside the triangle.
    # Check dot product with v1.
    dot = np.dot(c, v1)
    if dot < 0:
        c = -c
        dot = -dot
    
    # The angular distance is arccos(dot)
    # Clip to avoid numerical issues
    dot = np.clip(dot, -1.0, 1.0)
    return np.arccos(dot)

def _covering_radius(pts):
    """Compute covering radius via convex hull facets using precise spherical circumradius."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
        max_circumradius = 0.0
        for simplex_idx in hull.simplices:
            if len(simplex_idx) == 3:
                simplex = pts[simplex_idx]
                cr = _spherical_circumradius(simplex[0], simplex[1], simplex[2])
                max_circumradius = max(max_circumradius, cr)
            # Other simplex types are less common/irrelevant for max covering radius in this context
        return np.degrees(max_circumradius)
    except Exception:
        # Fallback: sample-based estimate (less precise but robust)
        rng = np.random.RandomState(42)
        samples = rng.randn(5000, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        # Vectorized distance calculation
        # samples: (5000, 3), pts: (50, 3)
        # We want min distance for each sample to any point
        # Dot product: (5000, 50)
        dots = np.clip(np.dot(samples, pts.T), -1, 1)
        angles = np.arccos(dots)
        min_angles = np.min(angles, axis=1)
        return np.degrees(np.max(min_angles))

def _fibonacci_points(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return _normalize(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))

def _random_points(n, seed):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _normalize(pts)

def _icosahedron_based_points(n, seed):
    """Start from icosahedron vertices and subdivide."""
    rng = np.random.RandomState(seed)
    # Icosahedron vertices
    t = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ])
    ico = _normalize(ico)
    # Add random points and keep best n
    all_pts = np.vstack([ico, _random_points(n - len(ico), seed)])
    # Select n points with good spread using greedy farthest-point
    selected = [0]
    for _ in range(n - 1):
        # Vectorized min distance to selected
        # all_pts: (M, 3), selected_pts: (K, 3)
        # Dots: (M, K)
        sel_pts = all_pts[selected]
        dots = np.clip(np.dot(all_pts, sel_pts.T), -1, 1)
        angles = np.arccos(dots)
        min_dists = np.min(angles, axis=1)
        selected.append(np.argmax(min_dists))
    return _normalize(all_pts[selected])

def _local_refine(pts, time_limit):
    """Refine points using gradient-based optimization on covering radius."""
    pts = _normalize(pts).copy()
    best = pts.copy()
    best_cr = _covering_radius(best)
    start_time = time.time()
    
    def objective_flat(x):
        p = _normalize(x.reshape(-1, 3))
        return _covering_radius(p)
    
    # Use Nelder-Mead with restarts
    for restart in range(5):
        if time.time() - start_time > time_limit - 5:
            break
        x0 = best.flatten()
        try:
            res = minimize(objective_flat, x0, method='Nelder-Mead', 
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_cr:
                best_cr = res.fun
                best = _normalize(res.x.reshape(-1, 3))
        except Exception:
            pass
        if time.time() - start_time > time_limit - 5:
            break
    
    return best, best_cr

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin for 150s limit
    
    # Multiple diverse initializations
    initializations = []
    initializations.append(('fib', _fibonacci_points(N)))
    for seed in range(5):
        initializations.append((f'rand_{seed}', _random_points(N, seed)))
    for seed in range(3):
        initializations.append((f'ico_{seed}', _icosahedron_based_points(N, seed)))
    
    best_pts = None
    best_cr = np.inf
    
    # Evaluate each initialization
    for name, pts in initializations:
        if time.time() > deadline - 20:
            break
        cr = _covering_radius(pts)
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
    
    # Refine the best initialization
    if best_pts is not None and time.time() < deadline - 10:
        refined_pts, refined_cr = _local_refine(best_pts, deadline - time.time() - 5)
        if refined_cr < best_cr:
            best_cr = refined_cr
            best_pts = refined_pts
    
    # If we still have time, try refining other good initializations
    if time.time() < deadline - 15:
        for name, pts in initializations:
            if time.time() > deadline - 10:
                break
            cr = _covering_radius(pts)
            if cr < best_cr + 1.0:  # Only refine if close to best
                if time.time() < deadline - 10:
                    refined_pts, refined_cr = _local_refine(pts, deadline - time.time() - 5)
                    if refined_cr < best_cr:
                        best_cr = refined_cr
                        best_pts = refined_pts
    
    if best_pts is None:
        best_pts = _fibonacci_points(N)
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END

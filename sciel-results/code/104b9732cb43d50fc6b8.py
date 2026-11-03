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

def _covering_radius(pts):
    """Compute covering radius via convex hull facets."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
        max_circumradius = 0.0
        for simplex_idx in hull.simplices:
            simplex = pts[simplex_idx]
            # Compute circumradius of this simplex on the sphere
            # For a triangle on unit sphere, circumradius relates to the cap it defines
            # The covering radius for a facet is the angular distance from the facet's circumcenter to its vertices
            if len(simplex) == 3:
                a, b, c = simplex
                # Angles between points
                ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
                bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
                ca = np.arccos(np.clip(np.dot(c, a), -1, 1))
                # Circumradius of spherical triangle
                s = (ab + bc + ca) / 2
                # Use spherical law of cosines / area-based approach
                # R_circ = sqrt(tan(s/2) * tan((s-ab)/2) * tan((s-bc)/2) * tan((s-ca)/2))
                # But simpler: the covering radius is the max angular distance to any point on the sphere
                # For a facet, the worst point is the circumcenter of the spherical triangle
                # Approximate: use the incenter or centroid direction
                centroid = (a + b + c) / 3
                centroid = centroid / np.linalg.norm(centroid)
                # Angular distances from centroid to vertices
                d1 = np.arccos(np.clip(np.dot(centroid, a), -1, 1))
                d2 = np.arccos(np.clip(np.dot(centroid, b), -1, 1))
                d3 = np.arccos(np.clip(np.dot(centroid, c), -1, 1))
                cr = max(d1, d2, d3)
            elif len(simplex) == 2:
                a, b = simplex
                # For an edge, the worst point is the midpoint of the arc
                midpoint = (a + b) / 2
                midpoint = midpoint / np.linalg.norm(midpoint)
                d1 = np.arccos(np.clip(np.dot(midpoint, a), -1, 1))
                cr = d1
            else:
                cr = 0.0
            max_circumradius = max(max_circumradius, cr)
        return np.degrees(max_circumradius)
    except Exception:
        # Fallback: sample-based estimate
        rng = np.random.RandomState(42)
        samples = rng.randn(5000, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        min_dists = np.min([np.arccos(np.clip(np.dot(samples, p), -1, 1)) for p in pts], axis=0)
        return np.degrees(np.max(min_dists))

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
        min_dists = np.min([np.arccos(np.clip(np.dot(all_pts, all_pts[i]), -1, 1)) for i in selected], axis=0)
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

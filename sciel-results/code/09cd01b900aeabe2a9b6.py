import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

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
        return np.pi / 2 # Or max distance, but shouldn't happen
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

def _covering_radius(points):
    """
    Compute the covering radius (max spherical circumradius of hull facets) in degrees.
    """
    try:
        hull = ConvexHull(points)
        max_r = 0.0
        for simplex in hull.simplices:
            v1 = points[simplex[0]]
            v2 = points[simplex[1]]
            v3 = points[simplex[2]]
            r = _spherical_circumradius(v1, v2, v3)
            if r > max_r:
                max_r = r
        return np.degrees(max_r)
    except Exception:
        return 180.0 # Invalid configuration

def _fibonacci_points(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n, seed=42):
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(n, 3))
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms

def _icosahedron_points(n):
    # Start with icosahedron and add random points?
    # Or just use random.
    # Let's try a mix.
    return _random_points(n, seed=123)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    best_points = None
    best_score = float('inf')
    
    # Initializations
    inits = [
        _fibonacci_points(N),
        _random_points(N, seed=1),
        _random_points(N, seed=2),
        _random_points(N, seed=3),
    ]
    
    for init in inits:
        if time.time() > deadline:
            break
            
        # Optimization
        # Nelder-Mead is good for this.
        # We need to flatten the points to a 1D array for scipy.optimize
        x0 = init.flatten()
        
        def objective(x):
            pts = x.reshape(N, 3)
            # Normalize to ensure they stay on sphere (NM might drift slightly)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            if np.any(norms < 1e-6):
                return 180.0
            pts = pts / norms
            return _covering_radius(pts)
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.success or res.fun < best_score:
                best_points = res.x.reshape(N, 3)
                best_score = res.fun
        except Exception:
            pass

    # If we have time, do a second pass of refinement on the best point found
    if best_points is not None and time.time() < deadline - 20.0:
        x0 = best_points.flatten()
        
        def objective(x):
            pts = x.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            if np.any(norms < 1e-6):
                return 180.0
            pts = pts / norms
            return _covering_radius(pts)
            
        try:
            # Smaller simplex for fine tuning
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8, 'initial_simplex': 1e-3})
            if res.fun < best_score:
                best_points = res.x.reshape(N, 3)
                best_score = res.fun
        except Exception:
            pass

    # Final normalization
    if best_points is None:
        return _fibonacci_points(N)
        
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points

# EVOLVE-BLOCK-END

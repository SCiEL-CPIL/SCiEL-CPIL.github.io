import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import time

N = 13

def _calc_score(pts):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    # Check for degenerate hull
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0

    # Calculate all triangle areas
    # n=13, so 13*12*11/6 = 286 triangles
    min_area = np.inf
    n = len(pts)
    for i in range(n):
        pi = pts[i]
        for j in range(i + 1, n):
            pj = pts[j]
            v1 = pj - pi
            for k in range(j + 1, n):
                pk = pts[k]
                v2 = pk - pi
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    
    if min_area == np.inf:
        return 0.0
        
    return min_area / hull_area

def _objective(x):
    """
    Objective function for optimization.
    We fix the first 3 points to form a triangle of area 0.5: (0,0), (1,0), (0,1).
    The remaining 10 points are optimized.
    x is an array of shape (20,) representing 10 points (x,y) pairs.
    """
    # Reshape x to 10 points
    inner_pts = x.reshape(-1, 2)
    
    # Construct full point set
    # Base triangle vertices
    base = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    
    # Combine
    pts = np.vstack([base, inner_pts])
    
    # Optional: Project points to stay within the triangle to ensure hull is the base triangle?
    # If points go outside, the hull area increases.
    # If we want to maximize min_area / hull_area, and we assume the optimal hull is the base triangle,
    # we should constrain points to be inside.
    # However, allowing them to move outside might find a better configuration with a different hull.
    # But for simplicity and robustness, let's start by constraining them to the triangle.
    # If a point is outside, we can clip it or penalize it.
    # Clipping changes the geometry.
    # Let's just let them be. If they go outside, the hull area changes.
    # The ratio is scale invariant, so the absolute position doesn't matter, only relative.
    
    score = _calc_score(pts)
    
    # We want to maximize score, so minimize -score
    return -score

def construct():
    start_time = time.time()
    deadline = 110.0 # 10s margin
    
    # Initial guess: Random points inside the triangle
    # Barycentric coordinates (u, v, w) with u+v+w=1, u,v,w >= 0
    # Point = u*A + v*B + w*C
    # A=(0,0), B=(1,0), C=(0,1)
    # Point = (v, w)
    # So x in [0,1], y in [0,1], x+y <= 1
    
    def random_inside_triangle():
        u = np.random.random(10)
        v = np.random.random(10)
        w = 1 - u - v
        # If w < 0, reflect or re-sample
        # Simple rejection
        while np.any(w < 0):
            u = np.random.random(10)
            v = np.random.random(10)
            w = 1 - u - v
        # Points are (v, w)
        return np.column_stack((v, w))

    # Bounds for the 10 inner points: x in [0,1], y in [0,1]
    # But we need x+y <= 1.
    # Differential Evolution handles box constraints.
    # We can map box constraints to the triangle using a transformation,
    # or just use box [0,1]x[0,1] and accept that some points might be outside the base triangle,
    # which would change the hull.
    # To be safe and efficient, let's use a transformation.
    # Actually, DE is slow with 20 variables.
    # Let's try a smaller search space or a smarter initialization.
    
    # Alternative: Use a grid or known symmetric patterns.
    # For n=13, a 4x4 grid (16 points) is too many.
    # A triangular grid of side 4 has 1+3+5+7 = 16 points.
    # Side 3 has 1+3+5 = 9 points.
    # So 13 points is between side 3 and 4.
    
    # Let's use Differential Evolution with a reasonable number of iterations.
    # 20 variables.
    # Bounds: [0,1] for x and y.
    bounds = [(0, 1)] * 20
    
    # Use a seeded RNG for reproducibility
    rng = np.random.default_rng(42)
    
    best_result = None
    best_score = -1
    
    try:
        # Differential Evolution
        # maxiter needs to be small enough to fit in time.
        # popsize=15 (default) * 20 vars * maxiter
        # Let's try maxiter=100, popsize=20
        result = differential_evolution(
            _objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=20,
            tol=1e-4,
            polish=True # Use Nelder-Mead for final polish
        )
        
        if result.success or result.fun < best_score:
            best_result = result.x
            best_score = -result.fun
            
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        if best_result is not None:
             return _reconstruct_points(best_result)
        else:
             return _fallback_points()

    # If we have time, do a local refinement on the best result
    if best_result is not None:
        # Run Nelder-Mead again for higher precision
        try:
            res2 = minimize(
                _objective, 
                best_result, 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res2.fun < best_score:
                best_result = res2.x
                best_score = -res2.fun
        except Exception:
            pass
            
    if best_result is not None:
        return _reconstruct_points(best_result)
    else:
        return _fallback_points()

def _reconstruct_points(x_inner):
    inner_pts = x_inner.reshape(-1, 2)
    base = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    pts = np.vstack([base, inner_pts])
    return pts

def _fallback_points():
    # Return a simple grid or random points if optimization fails
    # A 4x4 grid clipped to triangle?
    # Or just the base triangle + 10 random points inside
    rng = np.random.default_rng(0)
    u = rng.random(10)
    v = rng.random(10)
    w = 1 - u - v
    # Ensure inside
    mask = w < 0
    while np.any(mask):
        u[mask] = rng.random(np.sum(mask))
        v[mask] = rng.random(np.sum(mask))
        w = 1 - u - v
        mask = w < 0
    inner = np.column_stack((v, w))
    base = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    return np.vstack([base, inner])

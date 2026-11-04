import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

N = 13

def _compute_score(points):
    """
    Compute the score: min triangle area / convex hull area.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Calculate convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0

    # Calculate all triangle areas
    min_area = float('inf')
    
    # Vectorized-ish approach for speed
    # Precompute all pairs? No, 286 is small.
    # But let's use numpy operations on the whole set if possible?
    # Actually, simple loops are fine for 286 triples.
    
    for i in range(n - 2):
        p1 = points[i]
        for j in range(i + 1, n - 1):
            p2 = points[j]
            v1 = p2 - p1
            for k in range(j + 1, n):
                p3 = points[k]
                v2 = p3 - p1
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
                    if min_area < 1e-10:
                        return 0.0
    
    if min_area < 1e-10:
        return 0.0
        
    return min_area / hull_area

def _objective(params):
    """
    Objective function for optimization.
    Minimizes -score.
    """
    points = params.reshape(-1, 2)
    
    # Check for degeneracy
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-6:
            return 1e6 # Penalty for non-convex or degenerate
    except Exception:
        return 1e6
        
    # Calculate areas
    n = len(points)
    min_area = float('inf')
    
    for i in range(n - 2):
        p1 = points[i]
        for j in range(i + 1, n - 1):
            p2 = points[j]
            v1 = p2 - p1
            for k in range(j + 1, n):
                p3 = points[k]
                v2 = p3 - p1
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
                    if min_area < 1e-10:
                        return 1e6

    if min_area < 1e-10:
        return 1e6
        
    score = min_area / hull_area
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    best_score = 0.0
    best_points = None
    
    # Strategy: 
    # 1. Differential Evolution (Global) to find a good region.
    # 2. Local Polish (L-BFGS-B) to refine.
    # 
    # Since the problem is affine invariant, we can fix the scale/position to some extent,
    # but DE handles the full space. 
    # Bounds: Let's search in [-1, 1] x [-1, 1]. The normalization handles the rest.
    
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    # We will run DE for a portion of the time, then polish.
    # DE is stochastic. We'll run it with a time limit if possible, or fixed iterations.
    # scipy's DE doesn't have a direct time limit, but we can estimate.
    # Let's try to run DE for ~60-80 seconds, then polish with remaining time.
    
    # To make DE faster, we can reduce popsize or maxiter if it's too slow.
    # popsize=20, maxiter=100 might be reasonable.
    
    try:
        # Run Differential Evolution
        # tol=1e-6, maxiter=100, popsize=20
        # We need to be careful with the time.
        # Let's set a smaller maxiter and see.
        
        # Since we can't easily interrupt DE mid-run without a callback that raises,
        # we'll estimate. 
        # Each objective eval is ~1ms? 20 pop * 100 iter * 20 evals/iter? No, DE is 1 eval per individual per gen.
        # So 20 * 100 = 2000 evals. ~2-5 seconds. 
        # We can afford more.
        
        # Let's try maxiter=300, popsize=30.
        # 30 * 300 = 9000 evals. ~10-20 seconds.
        
        res = differential_evolution(
            _objective, 
            bounds, 
            seed=42,
            maxiter=300,
            popsize=30,
            tol=1e-6,
            polish=False # We will polish manually to control time
        )
        
        current_params = res.x
        current_score = -res.fun
        
        # Polish with L-BFGS-B
        # We have ~120s left.
        # L-BFGS-B is fast.
        
        # We might want to run multiple polishes from slightly perturbed points?
        # Or just one good polish.
        
        # Let's do a sequence of polishes with increasing tightness or just one robust one.
        
        # L-BFGS-B needs bounds.
        # Since we are in [-1, 1] mostly, keep bounds.
        
        res2 = minimize(
            _objective, 
            current_params, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-12}
        )
        
        if res2.success or res2.fun < _objective(current_params):
            current_params = res2.x
            current_score = -res2.fun
            
        # Check time
        if time.time() < deadline - 5.0:
            # Try a second DE run with different seed?
            # Or just another polish?
            # Let's try another DE run to see if we find a better global.
            
            res3 = differential_evolution(
                _objective, 
                bounds, 
                seed=123,
                maxiter=200,
                popsize=20,
                tol=1e-6,
                polish=False
            )
            
            if -res3.fun > current_score:
                current_params = res3.x
                current_score = -res3.fun
                
                # Polish this one too
                res4 = minimize(
                    _objective, 
                    current_params, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-12}
                )
                
                if res4.success or res4.fun < _objective(current_params):
                    current_params = res4.x
                    current_score = -res4.fun
        
        # Final check
        final_points = current_params.reshape(-1, 2)
        final_score = _compute_score(final_points)
        
        if final_score > best_score:
            best_score = final_score
            best_points = final_points.copy()
            
    except Exception as e:
        pass # Fallback to below
    
    if best_points is None:
        # Fallback: Regular polygon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        x = 0.5 * np.cos(angles)
        y = 0.5 * np.sin(angles)
        best_points = np.column_stack((x, y))
        
    return best_points

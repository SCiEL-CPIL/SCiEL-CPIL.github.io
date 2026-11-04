"""
Strategy: Direct optimization of the 28 coordinates using Differential Evolution
with a soft-min objective function to handle the non-smooth "min" operator.
This avoids the bias of ring-based parameterizations and allows the points
to settle into an arbitrary optimal configuration.
"""

import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution

N = 14

def _compute_min_triangle_ratio(pts):
    """
    Compute min_triangle_area / convex_hull_area.
    """
    if len(pts) < 3:
        return 0.0
    
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-12:
            return 0.0
    except Exception:
        return 0.0

    n = len(pts)
    min_area = float('inf')
    
    # Vectorized calculation of min triangle area
    # Triangles (i, j, k) with i < j < k
    for i in range(n - 2):
        pi = pts[i]
        for j in range(i + 1, n - 1):
            pj = pts[j]
            pk = pts[j+1:]
            
            v1 = pj - pi
            v2 = pk - pi
            crosses = np.abs(v1[0]*v2[:,1] - v1[1]*v2[:,0])
            areas = 0.5 * crosses
            
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
                
    return min_area / hull_area

def _soft_min_objective(params, beta=100.0):
    """
    Objective function for optimization.
    We want to maximize the score, so we minimize the negative soft-min score.
    Using a soft-min (log-sum-exp) approximates the min function smoothly.
    """
    pts = params.reshape(N, 2)
    
    # Compute all triangle areas
    n = len(pts)
    areas = []
    
    # To keep memory and time reasonable, we compute areas in batches or just all at once
    # 14 points -> C(14,3) = 364 triangles. This is small.
    for i in range(n - 2):
        pi = pts[i]
        for j in range(i + 1, n - 1):
            pj = pts[j]
            pk = pts[j+1:]
            v1 = pj - pi
            v2 = pk - pi
            crosses = np.abs(v1[0]*v2[:,1] - v1[1]*v2[:,0])
            areas.append(0.5 * crosses)
            
    if not areas:
        return 1e9
        
    areas = np.concatenate(areas)
    
    # Check for degenerate hull
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-12:
            return 1e9
    except Exception:
        return 1e9

    # Normalize areas by hull area
    norm_areas = areas / hull_area
    
    # Soft min: -log(sum(exp(-beta * x))) / beta
    # This is a smooth approximation of min(x)
    # For large beta, it approaches min(x)
    # We want to maximize min(x), so we minimize -soft_min(x)
    # soft_min(x) = -1/beta * log(sum(exp(-beta * x)))
    # -soft_min(x) = 1/beta * log(sum(exp(-beta * x)))
    
    # Numerical stability: subtract max
    max_val = np.max(norm_areas)
    exp_terms = np.exp(-beta * (norm_areas - max_val))
    log_sum = np.log(np.sum(exp_terms))
    
    soft_min_val = max_val - (1.0 / beta) * log_sum
    
    return -soft_min_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_score = -float('inf')
    best_pts = None
    
    # Bounds for DE: We can normalize the hull to be within [0, 1]x[0, 1] or similar
    # Since the score is scale-invariant, we can fix the scale.
    # Let's constrain points to a unit square or circle for DE.
    # Actually, DE works best with bounded variables.
    # Let's use a box [-1, 1] x [-1, 1]. The hull area will be < 4.
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    # Strategy: Run DE for a fixed number of iterations or time limit.
    # DE can be slow. We will use a moderate population size and maxiter.
    # We will also try multiple random seeds to escape local optima.
    
    # To improve speed, we can use a smaller beta initially and increase it?
    # Or just use a fixed high beta.
    # Let's try beta=50 first.
    
    # We will run DE with a time check inside the callback if possible, 
    # but DE doesn't easily support time-based stopping in scipy.
    # We can run a limited number of iterations.
    
    # Let's estimate time per evaluation.
    # _soft_min_objective is ~364 cross products + log-sum-exp. Very fast.
    # DE with popsize=15 (15*28 vars) and maxiter=50 -> ~750 evaluations.
    # Should be fast enough to run multiple times.
    
    rng = np.random.default_rng(42)
    
    # Define a wrapper for DE that checks time
    def de_callback(xk, convergence):
        if time.time() > deadline - 10.0:
            raise RuntimeError("Time limit reached")
        return False

    # Run DE with different random seeds
    for seed in [42, 123, 456, 789, 101]:
        if time.time() > deadline - 20.0:
            break
            
        try:
            res = differential_evolution(
                _soft_min_objective, 
                bounds, 
                seed=seed,
                maxiter=100,
                popsize=15,
                tol=1e-4,
                callback=de_callback,
                polish=False  # We will polish manually to save time/control
            )
            
            if res.success or res.fun < 1e8:
                pts = res.x.reshape(N, 2)
                score = _compute_min_triangle_ratio(pts)
                
                if score > best_score:
                    best_score = score
                    best_pts = pts.copy()
                    
        except RuntimeError:
            # Time limit reached, break
            break
        except Exception as e:
            pass

    # Polish the best result found using Nelder-Mead on the hard min (or high beta soft min)
    if best_pts is not None and time.time() < deadline - 5.0:
        def _hard_negative_score(params):
            pts = params.reshape(N, 2)
            s = _compute_min_triangle_ratio(pts)
            if s <= 0:
                return 1e9
            return -s

        from scipy.optimize import minimize
        try:
            res = minimize(
                _hard_negative_score, 
                best_pts.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res.fun < 0:
                final_pts = res.x.reshape(N, 2)
                final_score = _compute_min_triangle_ratio(final_pts)
                if final_score > best_score:
                    best_score = final_score
                    best_pts = final_pts
        except Exception:
            pass

    # If no valid points found (shouldn't happen), return a regular polygon
    if best_pts is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_pts

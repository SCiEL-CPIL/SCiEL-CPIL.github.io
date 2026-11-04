"""
Strategy: Full-dimensional Direct Global Search (Differential Evolution)
Instead of assuming a 2-ring symmetric structure (7-7 split), this program
searches the full 28-dimensional space of point coordinates.
It uses scipy.optimize.differential_evolution for global exploration.
The objective function is the negative of the minimum triangle area ratio.
To handle the non-differentiable min() and potential numerical noise,
we rely on the robustness of DE and a tight deadline guard.
"""

import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution

N = 14

def _score(pts):
    """
    Compute the score: min triangle area / convex hull area.
    Returns 0 if points are collinear or hull area is 0.
    """
    if len(pts) < 3:
        return 0.0
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume # In 2D, volume is area
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    # Calculate Minimum Triangle Area
    # Vectorized calculation for all triples
    n = len(pts)
    min_area = float('inf')
    
    # Efficient vectorized computation
    # We iterate over pairs (i, j) and vectorize over k
    for i in range(n - 2):
        pi = pts[i]
        for j in range(i + 1, n - 1):
            pj = pts[j]
            pk = pts[j+1:]
            
            # Cross product of (pj - pi) and (pk - pi)
            v1x = pj[0] - pi[0]
            v1y = pj[1] - pi[1]
            v2x = pk[:, 0] - pi[0]
            v2y = pk[:, 1] - pi[1]
            
            crosses = np.abs(v1x * v2y - v1y * v2x)
            areas = 0.5 * crosses
            
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
                
    return min_area / hull_area

def _objective(params):
    """
    Objective function for minimization.
    Returns negative score (higher score is better, so we minimize -score).
    """
    pts = params.reshape((N, 2))
    score = _score(pts)
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s safety margin
    
    # Initial guess: A regular 14-gon on unit circle
    # This provides a valid starting point and a baseline score
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    initial_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    initial_score = _score(initial_pts)
    best_score = initial_score
    best_params = initial_pts.flatten()
    
    # Define bounds for each coordinate
    # We search in [-1, 1] x [-1, 1]
    bounds = [(-1.0, 1.0) for _ in range(2 * N)]
    
    # Run Differential Evolution
    # DE is stochastic and global. 
    # maxiter is capped, but we must also respect wall-clock time.
    # We use a callback to check time if possible, but DE doesn't support 
    # mid-iteration termination easily without wrapping. 
    # Instead, we limit maxiter and maxfev conservatively.
    
    # Estimate cost: 
    # _score is O(N^3) roughly, but vectorized inner loop makes it fast.
    # N=14, triples = 364. Vectorized ops are fast.
    # DE population size = 15 * 28 = 420 evaluations per generation.
    # If each eval takes ~1ms, 420ms/gen. 
    # 100 generations = 42s. 
    # Let's allow up to 150 generations, but monitor time.
    
    # We will run DE with a reasonable number of iterations.
    # If it times out, we return the best found so far (initial or from previous runs).
    # Note: differential_evolution does not have a built-in time limit.
    # We will run it, but if it's too slow, we might need to reduce popsize or maxiter.
    # Let's try maxiter=50, popsize=10 (smaller population for speed).
    
    try:
        # Use a smaller population for speed, but enough for diversity
        res = differential_evolution(
            _objective, 
            bounds, 
            maxiter=50, 
            popsize=10, 
            tol=0.01, 
            seed=42,
            polish=False # Skip local polish to save time, DE is global
        )
        
        if res.success or (not res.success and res.fun < -best_score):
            candidate_pts = res.x.reshape((N, 2))
            candidate_score = _score(candidate_pts)
            if candidate_score > best_score:
                best_score = candidate_score
                best_params = res.x
                
    except Exception:
        pass

    # If time remains, try a second round with a different seed or tighter bounds?
    # Or just return the best found.
    # Given the 120s limit and DE's variability, one run is often sufficient for a jump.
    # Let's check time.
    
    if time.time() < deadline - 10.0:
        # Try a second DE run with different parameters to escape local optima
        # Use a larger population but fewer iterations?
        try:
            res2 = differential_evolution(
                _objective, 
                bounds, 
                maxiter=30, 
                popsize=15, 
                tol=0.01, 
                seed=123,
                polish=False
            )
            
            if res2.success or (not res2.success and res2.fun < -best_score):
                candidate_pts2 = res2.x.reshape((N, 2))
                candidate_score2 = _score(candidate_pts2)
                if candidate_score2 > best_score:
                    best_score = candidate_score2
                    best_params = res2.x
        except Exception:
            pass

    # Final check: ensure we return valid points
    final_pts = best_params.reshape((N, 2))
    
    # Safety: if the optimization resulted in degenerate points (e.g. all same),
    # fallback to the initial regular polygon.
    if _score(final_pts) < 1e-6:
        final_pts = initial_pts
        
    return final_pts

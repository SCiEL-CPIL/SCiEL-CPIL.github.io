"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution

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

def _negative_score(params):
    """
    Objective function for optimization.
    params: flattened array of shape (2*N,) representing N points in 2D.
    """
    pts = params.reshape(-1, 2)
    
    # Prevent degenerate configurations (e.g., all points at origin)
    if np.allclose(pts, 0):
        return 1e9
    
    # The score is scale invariant, so the optimizer might try to scale points up/down.
    # To prevent this, we normalize the RMS radius to 1 before scoring.
    rms = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
    if rms < 1e-9:
        return 1e9
    pts = pts / rms
    
    s = _score(pts)
    return -s

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Define bounds for DE. 
    # Since we normalize RMS radius inside _negative_score, the absolute scale doesn't matter,
    # but providing reasonable bounds helps DE navigate the space.
    # Bounds [-1, 1] are sufficient given normalization.
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    best_result = None
    best_score = -np.inf
    
    # Multi-start Differential Evolution
    # DE is a population-based global optimizer.
    # We will run multiple independent DE instances with different random seeds.
    # Each instance uses a subset of the time budget.
    
    # Number of restarts. 
    # Each DE run with popsize=20, maxiter=50 takes roughly 1-2 seconds (approx 1000 evals).
    # 110s budget allows for ~50-100 restarts if we keep it tight, or fewer with higher iterations.
    # Let's try 20 restarts with moderate iterations.
    
    num_restarts = 20
    maxiter_per_run = 100
    popsize = 20
    
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        try:
            # Differential Evolution
            # Strategy: 'best1bin' is default. 'rand1bin' is more exploratory.
            # Let's stick to default or 'best1bin' with high mutation/crossover.
            result = differential_evolution(
                _negative_score, 
                bounds, 
                maxiter=maxiter_per_run,
                popsize=popsize,
                seed=i,
                tol=1e-8,
                init='sobol', # Sobol sequence for better initial spread
                mutation=(0.5, 1.5), # Higher mutation for exploration
                recombination=0.8
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x
                
        except Exception:
            pass
            
    # If no result found (shouldn't happen), fallback to regular polygon
    if best_result is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_result = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
        
    pts = best_result.reshape(-1, 2)
    
    # Final normalization to unit RMS radius for consistency
    rms = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
    if rms > 1e-6:
        pts = pts / rms
        
    return pts
# EVOLVE-BLOCK-END

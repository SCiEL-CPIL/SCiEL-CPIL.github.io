"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import itertools
import time

N = 13

# EVOLVE-BLOCK-START

def _compute_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    Returns 0.0 if degenerate.
    """
    points = np.asarray(points)
    n = len(points)
    
    # Check for duplicate points or collinearity issues
    if n < 3:
        return 0.0

    # 1. Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0

    # 2. Calculate Minimum Triangle Area
    # There are C(13, 3) = 286 triangles
    min_area = float('inf')
    
    # Vectorized calculation for all triples
    # Get all indices
    idx = np.arange(n)
    i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
    
    # Filter valid triples i < j < k
    mask = (i < j) & (j < k)
    i = i[mask]
    j = j[mask]
    k = k[mask]
    
    # Get points
    p_i = points[i]
    p_j = points[j]
    p_k = points[k]
    
    # Area = 0.5 * | (x_j - x_i)*(y_k - y_i) - (y_j - y_i)*(x_k - x_i) |
    # We can compute 2*Area directly to avoid the 0.5 factor until the end
    # 2*Area = | cross(p_j - p_i, p_k - p_i) |
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    
    min_area = np.min(areas)
    
    if min_area <= 1e-12:
        return 0.0
        
    return min_area / hull_area

def _negative_score(points_flat):
    """
    Objective function for minimization.
    Reshapes flat array to (N, 2) and returns negative score.
    """
    points = points_flat.reshape(N, 2)
    score = _compute_score(points)
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initial best solution
    best_score = -np.inf
    best_points = None
    
    # Strategy 1: Differential Evolution for global search
    # Bounds: [0, 1] x [0, 1]
    bounds = [(0, 1)] * (N * 2)
    
    # Use a seed for reproducibility in the first run, but allow multiple runs if time permits
    # DE is stochastic, so we run it with a specific seed first.
    
    # Run Differential Evolution
    # maxiter and popsize need to be balanced against time
    # N=13, dim=26. 
    # Let's try a moderate population size.
    
    try:
        result = differential_evolution(
            _negative_score, 
            bounds, 
            seed=42, 
            maxiter=50, 
            popsize=10, 
            tol=1e-4, 
            polish=True
        )
        
        if result.success or (result.fun < best_score):
            best_points = result.x.reshape(N, 2)
            best_score = -result.fun
            
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 30:
        # If we are close to the deadline, skip local polish
        if best_points is not None:
            return best_points
        else:
            # Fallback to random
            rng = np.random.default_rng(42)
            return rng.random((N, 2))
    
    # Strategy 2: Local Polish using L-BFGS-B or Nelder-Mead
    # Start from the best point found by DE
    if best_points is not None:
        try:
            # Use Nelder-Mead for local refinement as it handles non-smoothness (min function) better than gradient-based methods
            # L-BFGS-B assumes differentiability, which min() is not.
            result_local = minimize(
                _negative_score, 
                best_points.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            local_score = -result_local.fun
            if local_score > best_score:
                best_points = result_local.x.reshape(N, 2)
                best_score = local_score
                
        except Exception:
            pass
    
    # Strategy 3: Try a few random restarts if time permits
    if time.time() < deadline - 10:
        rng = np.random.default_rng(123)
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            # Random initialization within [0,1]
            x0 = rng.random((N, 2))
            
            # Quick local search
            try:
                result_rand = minimize(
                    _negative_score, 
                    x0.flatten(), 
                    method='Nelder-Mead', 
                    options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                rand_score = -result_rand.fun
                if rand_score > best_score:
                    best_points = result_rand.x.reshape(N, 2)
                    best_score = rand_score
            except Exception:
                pass
    
    if best_points is None:
        # Final fallback
        rng = np.random.default_rng(0)
        best_points = rng.random((N, 2))
        
    return best_points

# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

def _min_triangle_area(points):
    """
    Compute the minimum area of any triangle formed by 3 points.
    """
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    
    for i in range(N):
        for j in range(i + 1, N):
            if j + 1 >= N:
                continue
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas = 0.5 * np.abs(cross)
            
            if len(areas) > 0:
                local_min = np.min(areas)
                if local_min < min_area:
                    min_area = local_min
                    
    return min_area

def _score(points):
    """
    Compute the minimum triangle area divided by the convex hull area.
    """
    try:
        # Check for degeneracy: all points must not be collinear
        # ConvexHull requires at least 3 non-collinear points
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    
    if hull_area < 1e-10:
        return 0.0
        
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _objective(points_flat):
    """
    Objective function for Differential Evolution.
    We want to maximize score, so we minimize -score.
    To avoid division by zero or unstable hulls, we add a penalty if the hull is bad.
    """
    points = points_flat.reshape(N, 2)
    
    # Simple sanity check
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
        
    score = _score(points)
    
    if score < 1e-6:
        return 1e10
        
    return -score

def construct():
    """
    Generate 14 points in a planar convex region normalized to unit area
    to maximize the minimum area over all triples.
    
    Strategy:
    1. Use Differential Evolution to search the global landscape of point configurations.
       DE is robust against local minima compared to Nelder-Mead.
    2. The search space is bounded. We can assume points lie within [-1, 1]x[-1, 1] 
       or similar, as the problem is scale invariant. 
    3. Normalize the best result found to unit area.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Define bounds for DE. 
    # Since the problem is scale invariant, the absolute size doesn't matter, 
    # only the relative positions. 
    # A regular 14-gon inscribed in a unit circle has coordinates in [-1, 1].
    # We'll search in [-1.5, 1.5] to allow for some expansion/contraction if needed,
    # though DE will likely find a configuration close to the unit scale.
    bounds = [(-1.5, 1.5)] * (2 * N)
    
    # Initial population seed: Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    reg_points = np.column_stack([np.cos(angles), np.sin(angles)])
    x0 = reg_points.flatten()
    
    # Differential Evolution parameters
    # maxiter: number of generations
    # popsize: size of the population
    # tol: convergence tolerance
    # strategy: 'best1bin' is default, 'best1exp' might be more robust
    # mutation: (low, high)
    # recombination: C
    
    best_result = None
    best_score_val = 0.0
    
    # Run DE
    try:
        # We use a moderate population and iterations to fit in the time budget.
        # DE is stochastic, so it might take longer than NM.
        # popsize=15 (default is 15), maxiter=100
        # We will monitor time.
        
        # Helper to check time inside DE callback? DE doesn't have a simple callback for stopping in older SciPy versions easily, 
        # but we can just run it and hope it finishes, or reduce maxiter.
        # Let's try maxiter=50, popsize=10 to be fast, then polish.
        
        res = differential_evolution(
            _objective, 
            bounds,
            seed=42,
            maxiter=50,
            popsize=10,
            tol=1e-5,
            mutation=(0.5, 1.5),
            recombination=0.7,
            polish=True,  # This uses L-BFGS-B at the end, which is good for fine-tuning
            init='latinhypercube'
        )
        
        if res.success or res.fun < 1e9:
            current_score = -res.fun
            if current_score > best_score_val:
                best_score_val = current_score
                best_result = res.x
    except Exception:
        pass
    
    # If DE didn't find anything good, fall back to regular polygon
    if best_result is None:
        best_result = x0
        best_score_val = _score(best_result.reshape(N, 2))
    else:
        # Verify the score of the DE result
        test_score = _score(best_result.reshape(N, 2))
        if test_score > best_score_val:
             best_score_val = test_score
        else:
             # DE might have returned a point that _score evaluates differently due to precision?
             # Or polish failed. Let's stick to the best we have.
             # Actually, _objective returns -score. So -res.fun should be score.
             pass

    # Polish with Nelder-Mead for local refinement
    if time.time() < deadline - 20:
        try:
            # Reshape for NM
            x0_nm = best_result
            res_nm = minimize(
                _objective, 
                x0_nm, 
                method='Nelder-Mead', 
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if res_nm.fun < _objective(best_result):
                best_result = res_nm.x
                best_score_val = -res_nm.fun
        except Exception:
            pass

    # Final Polish with Powell if time remains
    if time.time() < deadline - 10:
        try:
            res_powell = minimize(
                _objective, 
                best_result, 
                method='Powell', 
                options={'maxiter': 100, 'xtol': 1e-9, 'ftol': 1e-12}
            )
            if res_powell.fun < _objective(best_result):
                best_result = res_powell.x
                best_score_val = -res_powell.fun
        except Exception:
            pass

    # Normalize to unit area
    final_points = best_result.reshape(N, 2)
    
    try:
        hull = ConvexHull(final_points)
        hull_area = hull.volume
    except Exception:
        # Fallback to regular polygon if hull fails
        final_points = reg_points.copy()
        hull_area = 1.0 # Will be recalculated
        
    if hull_area > 1e-10:
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale
    else:
        # Degenerate case, return regular polygon normalized
        final_points = reg_points.copy()
        hull = ConvexHull(final_points)
        hull_area = hull.volume
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale

    return final_points
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import time


def _compute_score(points):
    """Compute R = (d_min/d_max)^2 for a set of points."""
    d = pdist(points)
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2


def _surrogate_objective(flat_points, p=40):
    """
    Smooth power-mean surrogate: minimize -log of power-mean of distances.
    This provides gradient-like information for DE.
    """
    n = 16
    points = flat_points.reshape(n, 2)
    d = pdist(points)
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0 or d_min == 0:
        return 1e10
    # Power mean of order p approximates min as p->inf
    # We want to maximize d_min/d_max, so minimize -(d_min/d_max)^p
    ratio = d_min / d_max
    if ratio <= 0:
        return 1e10
    return -np.log(ratio)


def _local_refine(points, deadline, seed=0):
    """Multi-stage local refinement: Powell → Nelder-Mead → Powell."""
    n = 16
    current_score = _compute_score(points)
    
    # Stage 1: Powell method
    def _neg_log_score(flat_pts):
        pts = flat_pts.reshape(n, 2)
        s = _compute_score(pts)
        if s <= 0:
            return 1e10
        return -np.log(s)
    
    flat = points.flatten()
    
    # Powell optimization
    result1 = minimize(_neg_log_score, flat, method='Powell', 
                       options={'maxiter': 500, 'xtol': 1e-10, 'ftol': 1e-12})
    if time.time() > deadline - 30:
        return points, current_score
    
    pts1 = result1.x.reshape(n, 2)
    score1 = _compute_score(pts1)
    if score1 > current_score:
        points = pts1
        current_score = score1
    
    # Stage 2: Nelder-Mead
    result2 = minimize(_neg_log_score, points.flatten(), method='Nelder-Mead',
                       options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
    if time.time() > deadline - 15:
        return points, current_score
    
    pts2 = result2.x.reshape(n, 2)
    score2 = _compute_score(pts2)
    if score2 > current_score:
        points = pts2
        current_score = score2
    
    # Stage 3: Final Powell polish
    result3 = minimize(_neg_log_score, points.flatten(), method='Powell',
                       options={'maxiter': 1000, 'xtol': 1e-12, 'ftol': 1e-14})
    
    pts3 = result3.x.reshape(n, 2)
    score3 = _compute_score(pts3)
    if score3 > current_score:
        points = pts3
        current_score = score3
    
    return points, current_score


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    # Define bounds for DE: normalize so that one pair is at distance 1
    # Fix point 0 at origin, point 1 at (1,0) to break translation/rotation symmetry
    # Remaining 14 points are free
    # But for DE, we'll work in full space and normalize later
    
    # Use bounds: [-5, 5] for each coordinate
    bounds = [(-5, 5)] * (n * 2)
    
    best_points = None
    best_score = -1.0
    
    # Run differential evolution with the surrogate objective
    # Use multiple restarts if time permits
    n_de_runs = 3
    
    for run in range(n_de_runs):
        if time.time() > deadline - 120:
            break
        
        seed_val = 42 + run * 100
        
        try:
            de_result = differential_evolution(
                _surrogate_objective,
                bounds=bounds,
                seed=seed_val,
                maxiter=200,
                tol=1e-8,
                popsize=15,
                mutation=(0.3, 1.0),
                recombination=0.7,
                polish=False,  # We'll do our own refinement
                init='latinhypercube'
            )
        except Exception:
            continue
        
        if time.time() > deadline - 120:
            break
        
        candidate_points = de_result.x.reshape(n, 2)
        
        # Normalize: center and scale
        candidate_points = candidate_points - candidate_points.mean(axis=0)
        scale = np.max(np.linalg.norm(candidate_points, axis=1))
        if scale > 1e-10:
            candidate_points = candidate_points / scale
        
        # Local refinement
        refined_points, refined_score = _local_refine(candidate_points, deadline, seed=seed_val)
        
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_points.copy()
    
    # If DE didn't produce good results, try some structured initializations
    if best_points is None or best_score < 0.01:
        # 4x4 grid
        grid = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
        grid = (grid - grid.mean(axis=0)) / grid.std()
        
        refined, score = _local_refine(grid, deadline, seed=999)
        if score > best_score:
            best_score = score
            best_points = refined
    
    # Final check
    if best_points is None:
        # Fallback: simple grid
        best_points = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
        best_points = (best_points - best_points.mean(axis=0)) / best_points.std()
    
    # Ensure valid output shape
    assert best_points.shape == (16, 2), f"Invalid shape: {best_points.shape}"
    assert np.all(np.isfinite(best_points)), "Non-finite values in points"
    
    return best_points


# EVOLVE-BLOCK-END

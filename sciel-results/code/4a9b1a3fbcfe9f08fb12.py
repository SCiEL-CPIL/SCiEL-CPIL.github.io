"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize

def _compute_score(points):
    """Compute the squared ratio of max to min pairwise distance."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    iu = np.triu_indices(N, k=1)
    dist_sq_upper = dist_sq[iu]
    
    if len(dist_sq_upper) == 0:
        return 1e10
    
    min_d_sq = np.min(dist_sq_upper)
    max_d_sq = np.max(dist_sq_upper)
    
    if min_d_sq <= 1e-12:
        return 1e10
    
    return max_d_sq / min_d_sq

def _objective_de(points_flat):
    """
    Objective function for Differential Evolution.
    Takes a flat array of (N, D) points.
    """
    points = points_flat.reshape(N, D)
    return _compute_score(points)

def construct():
    """
    Construct 14 points in 3D space to minimize the squared ratio of
    maximum to minimum pairwise distance.
    
    Uses Differential Evolution (global search) followed by Nelder-Mead (local polish).
    """
    # Bounds: Each coordinate in [-1, 1]
    bounds = [(-1.0, 1.0)] * (N * D)
    
    best_points = None
    best_score = 1e10

    # Phase 1: Global Search with Differential Evolution
    try:
        result = differential_evolution(
            _objective_de,
            bounds,
            maxiter=400,
            popsize=20,
            tol=1e-9,
            seed=42,
            polish=False
        )
        best_points = result.x.reshape(N, D)
        best_score = _compute_score(best_points)
    except Exception:
        # Fallback to random if DE fails
        best_points = np.random.uniform(-1, 1, (N, D))
        best_score = _compute_score(best_points)

    # Phase 2: Local Polish with Nelder-Mead
    try:
        res_nm = minimize(
            _objective_de, 
            best_points.flatten(), 
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        polished_points = res_nm.x.reshape(N, D)
        polished_score = _compute_score(polished_points)
        
        if polished_score < best_score:
            best_points = polished_points
            best_score = polished_score
    except Exception:
        pass

    # Phase 3: Multi-start Polish to escape potential local minima
    # DE provides a good global estimate, but NM can get stuck.
    # We try a few perturbations around the best found solution.
    for i in range(10):
        # Perturb slightly
        noise_scale = 0.1 / (i + 1)
        perturbation = np.random.normal(0, noise_scale, (N, D))
        start_pts = best_points + perturbation
        start_score = _compute_score(start_pts)
        
        # Keep track of best start point too
        if start_score < best_score:
            best_points = start_pts
            best_score = start_score
            
        try:
            res_nm = minimize(
                _objective_de, 
                start_pts.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            polished_points = res_nm.x.reshape(N, D)
            polished_score = _compute_score(polished_points)
            
            if polished_score < best_score:
                best_points = polished_points
                best_score = polished_score
        except Exception:
            continue

    return best_points
# EVOLVE-BLOCK-END

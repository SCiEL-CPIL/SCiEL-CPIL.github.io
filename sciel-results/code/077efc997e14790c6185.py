# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import time

_N = 14

def _score(points):
    """Compute (d_min / d_max)^2 for a set of points."""
    n = len(points)
    if n < 2:
        return 0.0
    dists = pdist(points, metric='euclidean')
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    return (d_min / d_max) ** 2

def _generate_tetrahedral_points(rv, rm, rc):
    """Generate 14 points based on tetrahedral symmetry orbits."""
    sqrt3 = np.sqrt(3)
    
    # Orbit 1: 4 Vertices of regular tetrahedron
    v_coords = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ]) / sqrt3
    v_points = rv * v_coords
    
    # Orbit 2: 6 Edge Midpoints
    m_coords = np.array([
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1]
    ])
    m_points = rm * m_coords
    
    # Orbit 3: 4 Face Centroids
    c_coords = np.array([
        [1, 1, -1],
        [1, -1, 1],
        [-1, 1, 1],
        [-1, -1, -1]
    ]) / sqrt3
    c_points = rc * c_coords
    
    return np.vstack([v_points, m_points, c_points])

def _soft_min_max_score(points, alpha=100.0):
    """
    Smooth surrogate for log((d_min/d_max)^2) using log-sum-exp.
    Returns 2*(softmin(log_dists) - softmax(log_dists))
    """
    dists = pdist(points, metric='euclidean')
    if len(dists) == 0:
        return -np.inf
    
    d_min = np.min(dists)
    d_max = np.max(dists)
    
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max) or d_min <= 0:
        return -np.inf
    
    log_dists = np.log(dists)
    
    # softmin(log_dists, alpha) = -(1/alpha)*logsumexp(-alpha*log_dists)
    neg_alpha_ld = -alpha * log_dists
    neg_alpha_ld_max = np.max(neg_alpha_ld)
    softmin_val = -(1.0/alpha) * (neg_alpha_ld_max + np.log(np.sum(np.exp(neg_alpha_ld - neg_alpha_ld_max))))
    
    # softmax(log_dists, alpha) = (1/alpha)*logsumexp(alpha*log_dists)
    alpha_ld = alpha * log_dists
    alpha_ld_max = np.max(alpha_ld)
    softmax_val = (1.0/alpha) * (alpha_ld_max + np.log(np.sum(np.exp(alpha_ld - alpha_ld_max))))
    
    log_ratio = softmin_val - softmax_val
    
    if not np.isfinite(log_ratio):
        return -np.inf
    
    return 2.0 * log_ratio

def _neg_soft_score_radii(params, alpha=100.0):
    """Negative soft score for optimization."""
    rv, rm, rc = params
    rv = abs(rv)
    rm = abs(rm)
    rc = abs(rc)
    
    if rv == 0 and rm == 0 and rc == 0:
        return 1.0
        
    points = _generate_tetrahedral_points(rv, rm, rc)
    if not np.all(np.isfinite(points)):
        return 1.0
        
    soft_val = _soft_min_max_score(points, alpha)
    if not np.isfinite(soft_val):
        return 1.0
        
    return -soft_val

def _exact_score_radii(params):
    """Exact negative score for final polishing."""
    rv, rm, rc = params
    rv = abs(rv)
    rm = abs(rm)
    rc = abs(rc)
    
    if rv == 0 and rm == 0 and rc == 0:
        return 1.0
        
    points = _generate_tetrahedral_points(rv, rm, rc)
    if not np.all(np.isfinite(points)):
        return 1.0
        
    return -_score(points)

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min / d_max)^2.
    
    Uses a high-symmetry combinatorial template (Tetrahedron Vertices, 
    Edge Midpoints, Face Centroids) and optimizes the 3 radial scaling factors.
    Uses Differential Evolution for global search and BFGS for local refinement.
    """
    n = _N
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    best_score = -np.inf
    best_points = None
    best_params = None
    
    # Bounds for the radii. 
    # The relative scale doesn't matter, but the ratios do.
    # We can normalize one radius, e.g., rv=1, and search rm, rc.
    # However, the current template generates points scaled by rv, rm, rc.
    # If we scale all by k, the distances scale by k, ratio stays same.
    # So we can fix rv=1 and search rm, rc in [0.1, 2.0].
    # But to be safe and allow the optimizer to find the best scale if numerical issues arise,
    # let's search all 3 but constrain them to a box.
    # Actually, since the score is scale-invariant, we can fix rv=1.0.
    # Let's try fixing rv=1.0 and optimizing rm, rc. This reduces dimensionality to 2.
    
    def _neg_soft_score_2d(params, alpha):
        rm, rc = params
        return _neg_soft_score_radii([1.0, rm, rc], alpha)
    
    def _exact_score_2d(params):
        rm, rc = params
        return _exact_score_radii([1.0, rm, rc])
    
    # Phase 1: Global Search with Differential Evolution
    # Use a moderate alpha for the global search to smooth out local minima
    alpha_global = 50.0
    bounds = [(0.1, 3.0), (0.1, 3.0)] # rm, rc
    
    if time.time() < deadline - 300:
        try:
            result_de = differential_evolution(
                lambda x: _neg_soft_score_2d(x, alpha_global),
                bounds=bounds,
                maxiter=200,
                popsize=20,
                tol=1e-8,
                seed=42,
                polish=False # We will polish separately
            )
            if result_de.x is not None:
                rm, rc = result_de.x
                params = np.array([1.0, rm, rc])
                score = -result_de.fun
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_points = _generate_tetrahedral_points(1.0, rm, rc)
        except Exception:
            pass

    # If DE failed or didn't find a good score, use a multi-start local search
    if best_params is None or best_score < 0.5:
        initial_guesses = [
            [1.0, 1.0, 1.0],
            [1.0, 1.2, 0.8],
            [1.0, 0.8, 1.2],
            [1.0, 1.5, 0.5],
            [1.0, 0.5, 1.5],
            [1.0, 1.1, 1.1],
            [1.0, 0.9, 0.9]
        ]
        
        for guess in initial_guesses:
            if time.time() > deadline - 200:
                break
            try:
                # Use BFGS with finite difference gradient on soft objective
                result_local = minimize(
                    lambda p: _neg_soft_score_2d(p, alpha_global),
                    guess[1:], # [rm, rc]
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if result_local.x is not None:
                    rm, rc = result_local.x
                    params = np.array([1.0, rm, rc])
                    score = -result_local.fun
                    if score > best_score:
                        best_score = score
                        best_params = params
                        best_points = _generate_tetrahedral_points(1.0, rm, rc)
            except Exception:
                continue

    # Phase 2: Local Refinement with increasing alpha
    if best_params is not None and time.time() < deadline - 100:
        current_params = best_params.copy()
        alphas = [100.0, 500.0, 1000.0, 5000.0]
        
        for alpha in alphas:
            if time.time() > deadline - 50:
                break
            try:
                result_ref = minimize(
                    lambda p, a=alpha: _neg_soft_score_2d(p, a),
                    current_params[1:],
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if result_ref.x is not None:
                    rm, rc = result_ref.x
                    new_params = np.array([1.0, rm, rc])
                    # Check with exact score to see if it's better
                    exact_score = -_exact_score_radii(new_params)
                    if exact_score > best_score:
                        best_score = exact_score
                        best_params = new_params
                        best_points = _generate_tetrahedral_points(1.0, rm, rc)
                        current_params = new_params
            except Exception:
                continue

    # Phase 3: Final High-Precision Polish with Exact Objective
    if best_params is not None and time.time() < deadline - 10:
        try:
            result_final = minimize(
                _exact_score_2d,
                best_params[1:],
                method='Nelder-Mead',
                bounds=bounds, # Nelder-Mead doesn't support bounds, but we can check
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            if result_final.x is not None:
                # Ensure bounds are respected (NM can go out of bounds)
                rm, rc = np.clip(result_final.x, 0.1, 3.0)
                new_params = np.array([1.0, rm, rc])
                exact_score = -_exact_score_radii(new_params)
                if exact_score > best_score:
                    best_score = exact_score
                    best_params = new_params
                    best_points = _generate_tetrahedral_points(1.0, rm, rc)
        except Exception:
            pass

    # Fallback
    if best_points is None:
        best_points = _generate_tetrahedral_points(1.0, 1.0, 1.0)
    
    # Ensure shape and validity
    if best_points.shape != (n, 3) or not np.all(np.isfinite(best_points)):
        golden_angle = np.pi * (3 - np.sqrt(5))
        thetas = np.arccos(1 - 2 * (np.arange(n) + 0.5) / n)
        phis = golden_angle * np.arange(n)
        x = np.sin(thetas) * np.cos(phis)
        y = np.sin(thetas) * np.sin(phis)
        z = np.cos(thetas)
        best_points = np.column_stack([x, y, z])

    return best_points

# EVOLVE-BLOCK-END

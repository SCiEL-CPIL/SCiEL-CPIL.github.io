"""
Differential Evolution + L-BFGS-B polish for Heilbronn convex regions n=14.
Uses a polar coordinate representation to enforce convexity and reduce search space.
"""
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import time
import itertools

N = 14

# EVOLVE-BLOCK-START

_TRIPLE_INDICES = np.array(list(itertools.combinations(range(N), 3)))
_NUM_TRIPLES = _TRIPLE_INDICES.shape[0]  # 364

def _all_triangle_areas(points):
    """Vectorized computation of all triangle areas."""
    i_idx = _TRIPLE_INDICES[:, 0]
    j_idx = _TRIPLE_INDICES[:, 1]
    k_idx = _TRIPLE_INDICES[:, 2]
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _min_triangle_area(points):
    return np.min(_all_triangle_areas(points))

def _convex_hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def _score(points):
    """Score = min triangle area / convex hull area."""
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    return min_area / hull_area

def _soft_min_triangle_area(points, temperature=0.01):
    """Smooth approximation of min triangle area via log-sum-exp."""
    areas = _all_triangle_areas(points)
    areas = np.maximum(areas, 1e-12)
    neg_T_areas = -areas / temperature
    max_val = np.max(neg_T_areas)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_T_areas - max_val)))
    return -temperature * log_sum_exp

def _soft_score(points, temperature=0.01):
    soft_min_area = _soft_min_triangle_area(points, temperature)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    return soft_min_area / hull_area

def _params_to_points(params):
    """
    Convert polar perturbation parameters to point coordinates.
    params: shape (2*N,) = [delta_r_0, delta_theta_0, ..., delta_r_13, delta_theta_13]
    Base: regular 14-gon, radius 1.
    Constraints:
      - delta_r in [-0.4, 0.4]  => radius in [0.6, 1.4] (positive)
      - delta_theta in [-0.25, 0.25] => angles remain ordered for small perturbations
    """
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    base_radius = np.ones(N)
    
    delta_r = params[0::2]
    delta_theta = params[1::2]
    
    # Clip to valid ranges
    delta_r = np.clip(delta_r, -0.4, 0.4)
    delta_theta = np.clip(delta_theta, -0.25, 0.25)
    
    radii = base_radius + delta_r
    angles = base_angles + delta_theta
    
    # Sort by angle to ensure convex ordering
    sort_idx = np.argsort(angles)
    angles_sorted = angles[sort_idx]
    radii_sorted = radii[sort_idx]
    
    x = radii_sorted * np.cos(angles_sorted)
    y = radii_sorted * np.sin(angles_sorted)
    
    return np.column_stack((x, y))

def _negative_score_params(params):
    """Objective for parameter-space optimization (exact)."""
    points = _params_to_points(params)
    return -_score(points)

def _negative_score_cartesian(flat_points):
    """Objective for Cartesian-space optimization (exact)."""
    points = flat_points.reshape(N, 2)
    return -_score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit

    # Define bounds for differential evolution
    # delta_r in [-0.4, 0.4], delta_theta in [-0.25, 0.25]
    bounds = [(-0.4, 0.4), (-0.25, 0.25)] * N
    
    # Initial guess: regular polygon (all zeros)
    init_params = np.zeros(2 * N)
    
    best_score = -np.inf
    best_points = _params_to_points(init_params)
    best_score = _score(best_points)
    
    # Phase 1: Differential Evolution for global search
    # DE is population-based and good for escaping local minima
    # We use a moderate population size and max iterations to fit within time
    try:
        de_result = differential_evolution(
            _negative_score_params,
            bounds,
            seed=42,
            maxiter=200,
            popsize=15,  # 15 * 14 = 210 evaluations per generation
            tol=1e-6,
            mutation=(0.3, 1.2),
            recombination=0.8,
            polish=True,  # DE includes a final polish step
            init='sobol'  # Sobol sequence for better initial sampling
        )
        
        if de_result.success or de_result.fun < -best_score:
            cur_params = de_result.x
            cur_points = _params_to_points(cur_params)
            cur_score = _score(cur_points)
            if cur_score > best_score:
                best_score = cur_score
                best_points = cur_points
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 10:
        return best_points
    
    # Phase 2: L-BFGS-B polish in parameter space
    # L-BFGS-B is faster than Nelder-Mead for smooth objectives and handles bounds
    try:
        res = minimize(
            _negative_score_params,
            _params_to_points(best_points).flatten() if False else np.zeros(2*N), # Use best_params from DE if available, else init
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        # Actually, we need the params, not points. Let's recover params from best_points is hard.
        # Better: keep track of best_params.
        # Let's re-run DE or just polish the DE result.
        # If DE didn't run or failed, we should polish the initial or a random start.
        # For simplicity, let's assume DE gave us a good start in best_points.
        # We need the corresponding params.
        # Since _params_to_points is not invertible easily, we should store best_params.
        
    except Exception:
        pass
    
    # Let's refactor to keep best_params
    best_params = init_params.copy()
    
    try:
        de_result = differential_evolution(
            _negative_score_params,
            bounds,
            seed=42,
            maxiter=200,
            popsize=15,
            tol=1e-6,
            mutation=(0.3, 1.2),
            recombination=0.8,
            polish=False,
            init='sobol'
        )
        if de_result.fun < -_score(_params_to_points(best_params)):
            best_params = de_result.x
            best_points = _params_to_points(best_params)
            best_score = _score(best_points)
    except Exception:
        pass
    
    if time.time() > deadline - 10:
        return best_points
    
    # Phase 2: L-BFGS-B polish in parameter space
    try:
        res = minimize(
            _negative_score_params,
            best_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if res.fun < -best_score:
            best_params = res.x
            best_points = _params_to_points(best_params)
            best_score = _score(best_points)
    except Exception:
        pass
    
    if time.time() > deadline - 5:
        return best_points
    
    # Phase 3: Final polish in Cartesian space (unconstrained, but we know it should be convex)
    # This allows points to move slightly off the polar grid if it helps
    try:
        res = minimize(
            _negative_score_cartesian,
            best_points.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if res.fun < -best_score:
            best_points = res.x.reshape(N, 2)
            best_score = _score(best_points)
    except Exception:
        pass
    
    return best_points

# EVOLVE-BLOCK-END

"""Rewritten solver: regular 14-gon with angular/radial perturbations, soft-min objective."""
import numpy as np
from scipy.optimize import minimize
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
    Convert perturbation parameters to point coordinates.
    params: array of shape (2*N,) = [delta_r_0, delta_theta_0, delta_r_1, delta_theta_1, ...]
    Base configuration: regular 14-gon with radius 1.
    """
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    base_radius = np.ones(N)
    
    delta_r = params[0::2]
    delta_theta = params[1::2]
    
    # Constrain perturbations to maintain convexity
    # delta_r in [-0.3, 0.3], delta_theta in [-0.3, 0.3]
    delta_r = np.clip(delta_r, -0.3, 0.3)
    delta_theta = np.clip(delta_theta, -0.3, 0.3)
    
    radii = base_radius + delta_r
    angles = base_angles + delta_theta
    
    # Ensure convexity by sorting by angle (if needed) and checking
    # For small perturbations, the order should be preserved
    # But to be safe, sort by angle
    sort_idx = np.argsort(angles)
    angles_sorted = angles[sort_idx]
    radii_sorted = radii[sort_idx]
    
    x = radii_sorted * np.cos(angles_sorted)
    y = radii_sorted * np.sin(angles_sorted)
    
    return np.column_stack((x, y))

def _negative_soft_score_params(params, temperature=0.01):
    """Objective for parameter-space optimization (soft)."""
    points = _params_to_points(params)
    return -_soft_score(points, temperature)

def _negative_score_params(params):
    """Objective for parameter-space optimization (exact)."""
    points = _params_to_points(params)
    return -_score(points)

def _negative_score_cartesian(flat_points):
    """Objective for Cartesian-space optimization (exact)."""
    points = flat_points.reshape(N, 2)
    return -_score(points)

def _generate_initial_params_list():
    """Generate diverse initial parameter configurations."""
    candidates = []
    
    # 1. Zero perturbations (regular polygon)
    candidates.append(np.zeros(2 * N))
    
    # 2-5. Small random perturbations
    for seed in [1, 2, 3, 4]:
        rng = np.random.RandomState(seed)
        candidates.append(rng.randn(2 * N) * 0.05)
    
    # 6-8. Medium random perturbations
    for seed in [5, 6, 7]:
        rng = np.random.RandomState(seed)
        candidates.append(rng.randn(2 * N) * 0.15)
    
    # 9. Alternating radius pattern
    params = np.zeros(2 * N)
    for i in range(N):
        params[2*i] = 0.1 if i % 2 == 0 else -0.1
    candidates.append(params)
    
    # 10. Angular shift pattern
    params = np.zeros(2 * N)
    for i in range(N):
        params[2*i+1] = 0.05 if i % 2 == 0 else -0.05
    candidates.append(params)
    
    return candidates

def _optimize_from_params(init_params, deadline):
    """
    Optimize from a single starting parameter configuration.
    Phase 1: Soft-min guided optimization in parameter space.
    Phase 2: Exact polish in parameter space.
    Phase 3: Final polish in Cartesian space (if time remains).
    """
    best_score = -np.inf
    best_params = init_params.copy()
    best_points = _params_to_points(init_params)
    
    # Phase 1: Soft-min optimization with decreasing temperature
    for temp in [0.1, 0.05, 0.02]:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(
                lambda x: _negative_soft_score_params(x, temp),
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-11}
            )
            if res.x is not None:
                cur_params = res.x
                cur_points = _params_to_points(cur_params)
                cur_score = _score(cur_points)
                if cur_score > best_score:
                    best_score = cur_score
                    best_params = cur_params
                    best_points = cur_points
        except Exception:
            pass
    
    # Phase 2: Exact objective polish in parameter space
    if time.time() < deadline - 3:
        try:
            res = minimize(
                _negative_score_params,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 8000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res.x is not None:
                cur_params = res.x
                cur_points = _params_to_points(cur_params)
                cur_score = _score(cur_points)
                if cur_score > best_score:
                    best_score = cur_score
                    best_params = cur_params
                    best_points = cur_points
        except Exception:
            pass
    
    # Phase 3: Cartesian polish (if time remains)
    if time.time() < deadline:
        try:
            res = minimize(
                _negative_score_cartesian,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-11, 'fatol': 1e-13}
            )
            if res.x is not None:
                cur_points = res.x.reshape(N, 2)
                cur_score = _score(cur_points)
                if cur_score > best_score:
                    best_score = cur_score
                    best_points = cur_points
        except Exception:
            pass
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit (killed at 150s)
    
    param_candidates = _generate_initial_params_list()
    
    best_score = -np.inf
    best_points = None
    
    for init_params in param_candidates:
        if time.time() > deadline - 8:  # Reserve time for final polish
            break
        try:
            pts, score = _optimize_from_params(init_params, deadline)
            if score > best_score:
                best_score = score
                best_points = pts
        except Exception:
            pass
    
    if best_points is None:
        # Fallback: regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Final global polish with remaining time
    if time.time() < deadline - 1:
        try:
            res = minimize(
                _negative_score_cartesian,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 15000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if res.x is not None:
                final_pts = res.x.reshape(N, 2)
                final_score = _score(final_pts)
                if final_score > best_score:
                    best_points = final_pts
        except Exception:
            pass
    
    return best_points

# EVOLVE-BLOCK-END

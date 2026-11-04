"""Heilbronn triangles: maximize the smallest triangle area formed by 11 points in an equilateral triangle."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

# Precompute indices for all triples to speed up area calculation
_IDX_I = None
_IDX_J = None
_IDX_K = None
_N_TRIPLES = None

def _get_indices():
    global _IDX_I, _IDX_J, _IDX_K, _N_TRIPLES
    if _IDX_I is None:
        i, j, k = np.indices((N, N, N))
        mask = (i < j) & (j < k)
        _IDX_I = i[mask]
        _IDX_J = j[mask]
        _IDX_K = k[mask]
        _N_TRIPLES = len(_IDX_I)
    return _IDX_I, _IDX_J, _IDX_K

def _min_triangle_area_vectorized(points):
    """Compute the smallest area among all triangles formed by triples of points, vectorized."""
    i_idx, j_idx, k_idx = _get_indices()
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _all_triangle_areas(points):
    """Compute all triangle areas, vectorized."""
    i_idx, j_idx, k_idx = _get_indices()
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas

def _param_to_points(params):
    """
    Map parameters in [0, 1]^{2N} to points inside the triangle.
    For each point i, we have params s_i, t_i in [0, 1].
    Barycentric coordinates (relative to V0, V1, V2) are:
    lambda_1 = s_i * (1 - t_i)
    lambda_2 = s_i * t_i
    lambda_3 = 1 - s_i
    Point = lambda_1 * V1 + lambda_2 * V2 + lambda_3 * V0
    Since V0 is (0,0), Point = lambda_1 * V1 + lambda_2 * V2
    """
    s = params[0::2]
    t = params[1::2]
    # Clip to avoid boundary issues in gradients if needed, but L-BFGS-B handles bounds
    # We use bounds in minimize, so params are already in [0,1]
    lam1 = s * (1.0 - t)
    lam2 = s * t
    x = lam1 * V1[0] + lam2 * V2[0]
    y = lam1 * V1[1] + lam2 * V2[1]
    return np.column_stack([x, y])

def _points_to_param(points):
    """
    Map points in the triangle to parameters (s, t).
    """
    x = points[:, 0]
    y = points[:, 1]
    # Inverse of param_to_points
    # y = lam1 * 0 + lam2 * sqrt(3)/2 => lam2 = 2*y/sqrt(3)
    # x = lam1 * 1 + lam2 * 0.5 => lam1 = x - 0.5*lam2
    # s = lam1 + lam2
    # t = lam2 / s
    lam2 = 2.0 * y / np.sqrt(3)
    lam1 = x - 0.5 * lam2
    s = lam1 + lam2
    t = np.zeros_like(s)
    mask = s > 1e-12
    t[mask] = lam2[mask] / s[mask]
    # Clip to valid range
    s = np.clip(s, 1e-10, 1.0 - 1e-10)
    t = np.clip(t, 1e-10, 1.0 - 1e-10)
    params = np.zeros(2 * N)
    params[0::2] = s
    params[1::2] = t
    return params

def _soft_min_objective(params, beta):
    """
    Smooth surrogate: negative soft-min of triangle areas.
    soft_min(areas) = -1/beta * log(sum(exp(-beta * areas)))
    As beta -> inf, this approaches min(areas).
    We minimize -soft_min.
    """
    points = _param_to_points(params)
    areas = _all_triangle_areas(points)
    # Numerically stable log-sum-exp
    min_a = np.min(areas)
    # soft_min = min_a - (1/beta) * log(sum(exp(-beta*(areas - min_a))))
    # Objective = -soft_min = -min_a + (1/beta) * log(sum(exp(-beta*(areas - min_a))))
    shifted = -beta * (areas - min_a)
    # To avoid overflow in exp, subtract max (which is 0 since areas>=min_a)
    # Actually shifted <= 0.
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = min_a - log_sum / beta
    return -soft_min_val

def _soft_min_grad(params, beta):
    """
    Gradient of the soft-min objective w.r.t. params.
    """
    points = _param_to_points(params)
    areas = _all_triangle_areas(points)
    
    # Weights for soft-min
    min_a = np.min(areas)
    shifted = -beta * (areas - min_a)
    # shifted is <= 0
    exp_shifted = np.exp(shifted)
    weights = exp_shifted / np.sum(exp_shifted)
    
    # d(soft_min)/d(area_k) = -w_k
    # Objective = -soft_min, so d(Objective)/d(area_k) = w_k
    
    i_idx, j_idx, k_idx = _get_indices()
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    # Compute cross products
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    
    # d(area)/d(point) = 0.5 * sign(cross) * d(cross)/d(point)
    sign_cross = np.sign(cross)
    sign_cross[np.abs(cross) < 1e-15] = 0.0
    
    # Accumulate gradients for each point
    grad_points = np.zeros((N, 2))
    
    # Contribution from pi
    d_cross_pi_x = -(pk[:, 1] - pi[:, 1]) + (pj[:, 1] - pi[:, 1])
    d_cross_pi_y = -(pj[:, 0] - pi[:, 0]) + (pk[:, 0] - pi[:, 0])
    w = weights * sign_cross * 0.5
    np.add.at(grad_points, i_idx, w[:, np.newaxis] * np.column_stack([d_cross_pi_x, d_cross_pi_y]))
    
    # Contribution from pj
    d_cross_pj_x = pk[:, 1] - pi[:, 1]
    d_cross_pj_y = -(pk[:, 0] - pi[:, 0])
    np.add.at(grad_points, j_idx, w[:, np.newaxis] * np.column_stack([d_cross_pj_x, d_cross_pj_y]))
    
    # Contribution from pk
    d_cross_pk_x = -(pj[:, 1] - pi[:, 1])
    d_cross_pk_y = pj[:, 0] - pi[:, 0]
    np.add.at(grad_points, k_idx, w[:, np.newaxis] * np.column_stack([d_cross_pk_x, d_cross_pk_y]))
    
    # Chain rule: d(params)/d(points)
    s = params[0::2]
    t = params[1::2]
    
    dx_ds = (1 - t) * V1[0] + t * V2[0]
    dx_dt = -s * V1[0] + s * V2[0]
    dy_ds = (1 - t) * V1[1] + t * V2[1]
    dy_dt = -s * V1[1] + s * V2[1]
    
    grad_params = np.zeros(2 * N)
    grad_params[0::2] = grad_points[:, 0] * dx_ds + grad_points[:, 1] * dy_ds
    grad_params[1::2] = grad_points[:, 0] * dx_dt + grad_points[:, 1] * dy_dt
    
    # Objective is -soft_min, so gradient is -d(soft_min)/d(params)
    # Wait.
    # Let F = -soft_min.
    # dF/darea_k = w_k.
    # dF/dparams = sum_k (dF/darea_k * darea_k/dparams)
    # The code above calculated grad_points which is d(soft_min)/d(point) ??
    # Let's re-verify signs.
    # soft_min = min_a - 1/beta log(sum exp(-beta(areas - min_a)))
    # d(soft_min)/d(area_k) = -w_k.
    # So d(soft_min)/d(point_m) = sum_k (-w_k) * d(area_k)/d(point_m).
    # My grad_points calculation:
    # w = weights * sign_cross * 0.5
    # This is w_k * d(area_k)/d(point_m) contribution?
    # area_k = 0.5 * |cross_k|.
    # d(area_k)/d(point_m) = 0.5 * sign(cross_k) * d(cross_k)/d(point_m).
    # So the term added to grad_points is w_k * 0.5 * sign(cross_k) * d(cross_k)/d(point_m).
    # This equals w_k * d(area_k)/d(point_m).
    # So grad_points currently holds sum_k w_k * d(area_k)/d(point_m).
    # This is d(sum_k w_k area_k)/d(point_m).
    # But we need d(soft_min)/d(point_m) = - sum_k w_k d(area_k)/d(point_m).
    # So grad_points = - d(soft_min)/d(point_m).
    # Therefore d(soft_min)/d(point_m) = -grad_points.
    # And we want gradient of Objective = -soft_min.
    # d(Objective)/d(point_m) = - d(soft_min)/d(point_m) = grad_points.
    # So grad_points is actually d(Objective)/d(point_m).
    
    # Now chain rule to params:
    # d(Objective)/d(param_j) = sum_m (d(Objective)/d(point_m) * d(point_m)/d(param_j))
    # grad_params[0::2] (ds) = sum_m grad_points[m,0] * dx_ds[m] + grad_points[m,1] * dy_ds[m]
    # This matches the code below.
    
    # So grad_params is the gradient of the Objective (-soft_min).
    return grad_params

def _generate_structured_inits():
    """Generate diverse initial configurations in Cartesian coordinates."""
    inits = []
    
    # 1. Symmetric grid-like
    # Vertices, edge midpoints, center
    pts = [
        [0.05, 0.02], # near V0
        [0.95, 0.02], # near V1
        [0.5, 0.85],  # near V2
        [0.5, 0.05],  # bottom mid
        [0.25, 0.43], # left mid
        [0.75, 0.43], # right mid
        [0.5, 0.29],  # center
        [0.35, 0.15],
        [0.65, 0.15],
        [0.5, 0.55],
        [0.25, 0.25],
    ]
    inits.append(np.array(pts))
    
    # 2. Random
    rng = np.random.default_rng(42)
    for _ in range(5):
        r1 = rng.uniform(0, 1, N)
        r2 = rng.uniform(0, 1, N)
        mask = r1 + r2 > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]
        x = r1 * V1[0] + r2 * V2[0]
        y = r1 * V1[1] + r2 * V2[1]
        inits.append(np.column_stack([x, y]))
        
    return inits

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    best_score = -float('inf')
    best_params = None
    
    inits = _generate_structured_inits()
    
    # We will run optimization from each init
    # Use L-BFGS-B with bounds
    bounds = [(1e-10, 1.0 - 1e-10)] * (2 * N)
    
    # We can use multiple beta values, starting low (smoother) and increasing (sharper)
    # But typically a single high beta works well if initialized well.
    # Let's try a few betas for the first few restarts to explore, then high beta for refinement.
    
    beta_list = [10.0, 50.0, 100.0, 200.0]
    
    for i, init_points in enumerate(inits):
        if time.time() > deadline - 10.0:
            break
            
        init_params = _points_to_param(init_points)
        
        # Select beta based on iteration or just cycle
        beta = beta_list[i % len(beta_list)]
        
        try:
            res = minimize(
                _soft_min_objective,
                init_params,
                args=(beta,),
                method='L-BFGS-B',
                jac=_soft_min_grad,
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < 0: # fun is negative score
                current_score = -res.fun
                if current_score > best_score:
                    best_score = current_score
                    best_params = res.x.copy()
        except Exception as e:
            pass
    
    # Refine the best solution with a very high beta (almost exact min)
    if best_params is not None and time.time() < deadline - 5.0:
        try:
            # Use the best params as start for high beta refinement
            beta_high = 500.0
            res = minimize(
                _soft_min_objective,
                best_params,
                args=(beta_high,),
                method='L-BFGS-B',
                jac=_soft_min_grad,
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun < 0:
                current_score = -res.fun
                if current_score > best_score:
                    best_score = current_score
                    best_params = res.x.copy()
        except Exception:
            pass
    
    # If no optimization succeeded, fallback to best init
    if best_params is None:
        # Use the first structured init
        best_params = _points_to_param(inits[0])
        best_score = _min_triangle_area_vectorized(_param_to_points(best_params))
        
    final_points = _param_to_points(best_params)
    
    # Verify points are inside (they should be due to bounds)
    return final_points

# EVOLVE-BLOCK-START
# The code above is the complete program.
# The construct function is the entry point.
# EVOLVE-BLOCK-END

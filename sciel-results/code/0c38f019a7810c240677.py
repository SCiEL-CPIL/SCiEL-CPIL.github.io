"""Heilbronn triangles: maximize the smallest triangle area formed by 11 points in an equilateral triangle."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2
S3 = np.sqrt(3)

# Precompute indices for all triples
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

def _clip_to_triangle(points):
    """Clip points to be inside the equilateral triangle."""
    points = points.copy()
    x = points[:, 0]
    y = points[:, 1]
    y = np.clip(y, 0, H)
    x_min = y / S3
    x_max = 1 - y / S3
    x = np.clip(x, x_min, x_max)
    x = np.clip(x, 0, 1)
    points[:, 0] = x
    points[:, 1] = y
    return points

def _param_to_points(params):
    """
    Map parameters in [0, 1]^{2N} to points inside the triangle.
    Barycentric coordinates:
    lambda_1 = s_i * (1 - t_i)
    lambda_2 = s_i * t_i
    lambda_3 = 1 - s_i
    """
    s = params[0::2]
    t = params[1::2]
    s = np.clip(s, 1e-10, 1.0 - 1e-10)
    t = np.clip(t, 1e-10, 1.0 - 1e-10)
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
    lam2 = 2.0 * y / np.sqrt(3)
    lam1 = x - 0.5 * lam2
    s = lam1 + lam2
    t = np.zeros_like(s)
    mask = s > 1e-12
    t[mask] = lam2[mask] / s[mask]
    s = np.clip(s, 1e-10, 1.0 - 1e-10)
    t = np.clip(t, 1e-10, 1.0 - 1e-10)
    params = np.zeros(2 * N)
    params[0::2] = s
    params[1::2] = t
    return params

def _soft_min_objective(params, beta):
    """
    Smooth surrogate: negative soft-min of triangle areas.
    """
    points = _param_to_points(params)
    areas = _all_triangle_areas(points)
    min_a = np.min(areas)
    shifted = -beta * (areas - min_a)
    max_shifted = np.max(shifted)
    log_sum = np.log(np.sum(np.exp(shifted - max_shifted))) + max_shifted
    soft_min = min_a - np.log(np.sum(np.exp(shifted - max_shifted))) / beta
    return -soft_min

def _soft_min_grad(params, beta):
    """
    Gradient of the soft-min objective w.r.t. params.
    Uses log-sum-exp formulation for numerical stability.
    """
    points = _param_to_points(params)
    areas = _all_triangle_areas(points)
    
    neg_beta_areas = -beta * areas
    max_neg = np.max(neg_beta_areas)
    log_sum = np.log(np.sum(np.exp(neg_beta_areas - max_neg))) + max_neg
    soft_min = -log_sum / beta
    
    exp_neg = np.exp(neg_beta_areas - max_neg)
    sum_exp = np.sum(exp_neg)
    weights = exp_neg / sum_exp  # These are the softmax weights
    
    i_idx, j_idx, k_idx = _get_indices()
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    
    sign_cross = np.sign(cross)
    sign_cross[np.abs(cross) < 1e-15] = 1.0
    
    grad_points = np.zeros((N, 2))
    
    d_cross_pi_x = pj[:, 1] - pk[:, 1]
    d_cross_pi_y = pk[:, 0] - pj[:, 0]
    
    d_cross_pj_x = pk[:, 1] - pi[:, 1]
    d_cross_pj_y = -(pk[:, 0] - pi[:, 0])
    
    d_cross_pk_x = -(pj[:, 1] - pi[:, 1])
    d_cross_pk_y = pj[:, 0] - pi[:, 0]
    
    w = weights * sign_cross * 0.5
    
    np.add.at(grad_points, i_idx, w[:, np.newaxis] * np.column_stack([d_cross_pi_x, d_cross_pi_y]))
    np.add.at(grad_points, j_idx, w[:, np.newaxis] * np.column_stack([d_cross_pj_x, d_cross_pj_y]))
    np.add.at(grad_points, k_idx, w[:, np.newaxis] * np.column_stack([d_cross_pk_x, d_cross_pk_y]))
    
    s = params[0::2]
    t = params[1::2]
    s = np.clip(s, 1e-10, 1.0 - 1e-10)
    t = np.clip(t, 1e-10, 1.0 - 1e-10)
    
    dx_ds = (1 - t) * V1[0] + t * V2[0]
    dx_dt = -s * V1[0] + s * V2[0]
    dy_ds = (1 - t) * V1[1] + t * V2[1]
    dy_dt = -s * V1[1] + s * V2[1]
    
    grad_params = np.zeros(2 * N)
    grad_params[0::2] = grad_points[:, 0] * dx_ds + grad_points[:, 1] * dy_ds
    grad_params[1::2] = grad_points[:, 0] * dx_dt + grad_points[:, 1] * dy_dt
    
    return -grad_params

def _generate_fps_initialization(rng, n_points=N, n_candidates=200):
    """
    Generate an initial configuration using Greedy Farthest Point Sampling.
    """
    points = np.zeros((n_points, 2))
    points[0] = (V0 + V1 + V2) / 3.0
    for k in range(1, n_points):
        r1 = rng.uniform(0, 1, n_candidates)
        r2 = rng.uniform(0, 1, n_candidates)
        mask = r1 + r2 > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]
        candidates = np.column_stack([
            r1 * V1[0] + r2 * V2[0],
            r1 * V1[1] + r2 * V2[1]
        ])
        diffs = candidates[:, np.newaxis, :] - points[np.newaxis, :k, :]
        dists = np.linalg.norm(diffs, axis=2)
        min_dists = np.min(dists, axis=1)
        best_idx = np.argmax(min_dists)
        points[k] = candidates[best_idx]
    return points

def _structured_inits():
    """Generate diverse structured initial configurations."""
    inits = []
    
    # Symmetric configuration near corners and center
    base_bary = np.array([
        [0.02, 0.02, 0.96],
        [0.96, 0.02, 0.02],
        [0.02, 0.96, 0.02],
        [0.45, 0.02, 0.53],
        [0.02, 0.45, 0.53],
        [0.53, 0.45, 0.02],
        [0.33, 0.33, 0.34],
        [0.25, 0.55, 0.20],
        [0.55, 0.20, 0.25],
        [0.20, 0.25, 0.55],
        [0.45, 0.25, 0.30],
    ])
    # Convert barycentric to cartesian
    b = base_bary[:, 1]
    c = base_bary[:, 2]
    x = b + 0.5 * c
    y = H * c
    inits.append(np.column_stack([x, y]))
    
    # Three rows
    row_bary = np.array([
        [0.05, 0.05, 0.90],
        [0.35, 0.05, 0.60],
        [0.65, 0.05, 0.30],
        [0.95, 0.05, 0.00],
        [0.15, 0.35, 0.50],
        [0.45, 0.35, 0.20],
        [0.75, 0.35, -0.10],
        [0.25, 0.65, 0.10],
        [0.55, 0.65, -0.20],
        [0.35, 0.55, 0.10],
        [0.65, 0.55, -0.20],
    ])
    row_bary[:, 2] = 1 - row_bary[:, 0] - row_bary[:, 1]
    b = row_bary[:, 1]
    c = row_bary[:, 2]
    x = b + 0.5 * c
    y = H * c
    inits.append(np.column_stack([x, y]))
    
    # Concentric pattern
    cent_bary = np.array([
        [0.33, 0.33, 0.34],
        [0.10, 0.40, 0.50],
        [0.50, 0.40, 0.10],
        [0.40, 0.10, 0.50],
        [0.90, 0.10, 0.00],
        [0.10, 0.90, 0.00],
        [0.50, 0.05, 0.45],
        [0.20, 0.55, 0.25],
        [0.55, 0.25, 0.20],
        [0.25, 0.20, 0.55],
        [0.45, 0.30, 0.25],
    ])
    b = cent_bary[:, 1]
    c = cent_bary[:, 2]
    x = b + 0.5 * c
    y = H * c
    inits.append(np.column_stack([x, y]))
    
    # Grid-based
    grid_bary = np.array([
        [0.05, 0.05, 0.90],
        [0.30, 0.05, 0.65],
        [0.55, 0.05, 0.40],
        [0.80, 0.05, 0.15],
        [0.15, 0.30, 0.55],
        [0.40, 0.30, 0.30],
        [0.65, 0.30, 0.05],
        [0.25, 0.55, 0.20],
        [0.50, 0.55, -0.05],
        [0.35, 0.45, 0.20],
        [0.60, 0.45, -0.05],
    ])
    grid_bary[:, 2] = 1 - grid_bary[:, 0] - grid_bary[:, 1]
    b = grid_bary[:, 1]
    c = grid_bary[:, 2]
    x = b + 0.5 * c
    y = H * c
    inits.append(np.column_stack([x, y]))
    
    # More symmetric patterns
    sym_bary = np.array([
        [0.03, 0.03, 0.94],
        [0.94, 0.03, 0.03],
        [0.03, 0.94, 0.03],
        [0.47, 0.03, 0.50],
        [0.03, 0.47, 0.50],
        [0.50, 0.47, 0.03],
        [0.33, 0.33, 0.34],
        [0.22, 0.53, 0.25],
        [0.53, 0.22, 0.25],
        [0.25, 0.22, 0.53],
        [0.42, 0.28, 0.30],
    ])
    b = sym_bary[:, 1]
    c = sym_bary[:, 2]
    x = b + 0.5 * c
    y = H * c
    inits.append(np.column_stack([x, y]))
    
    # Hexagonal-like pattern
    hex_bary = np.array([
        [0.33, 0.33, 0.34],
        [0.15, 0.45, 0.40],
        [0.55, 0.45, 0.00],
        [0.45, 0.15, 0.40],
        [0.15, 0.15, 0.70],
        [0.70, 0.15, 0.15],
        [0.45, 0.70, -0.15],
        [0.25, 0.25, 0.50],
        [0.50, 0.25, 0.25],
        [0.25, 0.50, 0.25],
        [0.50, 0.50, -0.00],
    ])
    hex_bary[:, 2] = 1 - hex_bary[:, 0] - hex_bary[:, 1]
    b = hex_bary[:, 1]
    c = hex_bary[:, 2]
    x = b + 0.5 * c
    y = H * c
    inits.append(np.column_stack([x, y]))
    
    # FPS initializations
    rng = np.random.RandomState(42)
    for seed in [42, 123, 456]:
        rng = np.random.RandomState(seed)
        inits.append(_generate_fps_initialization(rng))
    
    return inits

def _refine_softmin(params, deadline, beta_start=20.0, beta_end=500.0, n_betas=5):
    """Refine using soft-min surrogate with increasing beta (annealing)."""
    best_params = params.copy()
    best_points = _param_to_points(params)
    best_score = _min_triangle_area_vectorized(best_points)
    
    for i in range(n_betas):
        if time.time() > deadline - 10:
            break
        
        beta = beta_start * (beta_end / beta_start) ** (i / max(n_betas - 1, 1))
        
        try:
            res = minimize(
                lambda p, b=beta: _soft_min_objective(p, b),
                best_params.copy(),
                jac=lambda p, b=beta: _soft_min_grad(p, b),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            
            candidate_params = res.x
            candidate_points = _param_to_points(candidate_params)
            candidate_score = _min_triangle_area_vectorized(candidate_points)
            
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = candidate_points.copy()
                best_params = candidate_params.copy()
                
        except Exception:
            continue
    
    return best_points, best_score

def _nelder_refine(points, deadline, n_attempts=4):
    """Refine using Nelder-Mead in Cartesian space."""
    best_points = points.copy()
    best_score = _min_triangle_area_vectorized(points)
    
    rng = np.random.RandomState(42)
    
    for attempt in range(n_attempts):
        if time.time() > deadline - 5:
            break
        
        if attempt == 0:
            start = best_points.copy()
        else:
            scale = 0.05 / (1 + 0.5 * attempt)
            perturbation = rng.randn(N, 2) * scale
            start = best_points + perturbation
            start = _clip_to_triangle(start)
        
        try:
            def objective(flat):
                pts = flat.reshape(N, 2)
                x = pts[:, 0]
                y = pts[:, 1]
                eps = 1e-8
                penalty = 0.0
                penalty += np.sum(np.maximum(0, -y - eps)) * 100
                penalty += np.sum(np.maximum(0, y - S3 * x + eps)) * 100
                penalty += np.sum(np.maximum(0, y - S3 * (1 - x) + eps)) * 100
                min_area = _min_triangle_area_vectorized(pts)
                return -min_area + penalty
            
            res = minimize(
                objective,
                start.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 1500,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True,
                }
            )
            
            if res.fun < 0:
                candidate = res.x.reshape(N, 2)
                candidate = _clip_to_triangle(candidate)
                score = _min_triangle_area_vectorized(candidate)
                if score > best_score:
                    best_score = score
                    best_points = candidate.copy()
                    
        except Exception:
            continue
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_points = None
    best_score = -np.inf
    
    # Generate structured initial configurations
    inits = _structured_inits()
    
    # Phase 1: Refine each initialization with Nelder-Mead
    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline - 30:
            break
        
        init_pts = _clip_to_triangle(init_pts)
        refined_pts, refined_score = _nelder_refine(init_pts, deadline, n_attempts=3)
        
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_pts.copy()
    
    # Phase 2: Soft-min surrogate refinement with annealing
    if best_points is not None and time.time() < deadline - 20:
        params = _points_to_param(best_points)
        refined_pts, refined_score = _refine_softmin(params, deadline, beta_start=20.0, beta_end=300.0, n_betas=5)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_pts.copy()
    
    # Phase 3: Additional random restarts if time permits
    if best_points is not None and time.time() < deadline - 15:
        rng = np.random.RandomState(123)
        for restart in range(4):
            if time.time() > deadline - 10:
                break
            
            random_pts = _generate_fps_initialization(rng)
            random_pts = _clip_to_triangle(random_pts)
            
            refined_pts, refined_score = _nelder_refine(random_pts, deadline, n_attempts=2)
            
            if refined_score > best_score:
                best_score = refined_score
                best_points = refined_pts.copy()
    
    # Phase 4: Final high-precision polish with soft-min at high beta
    if best_points is not None and time.time() < deadline - 8:
        params = _points_to_param(best_points)
        refined_pts, refined_score = _refine_softmin(params, deadline, beta_start=50.0, beta_end=800.0, n_betas=3)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_pts.copy()
    
    # Final fallback
    if best_points is None:
        best_points = _clip_to_triangle(inits[0])
    
    # Ensure all points are valid
    best_points = _clip_to_triangle(best_points)
    
    return best_points

"""Heilbronn triangles: maximize the smallest triangle area formed by 11 points in an equilateral triangle."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

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

def _objective(params):
    """Objective: negative min triangle area."""
    points = _param_to_points(params)
    return -_min_triangle_area_vectorized(points)

def _soft_min_objective(params, beta):
    """
    Smooth surrogate: negative soft-min of triangle areas.
    soft_min = -logsumexp(-beta * areas) / beta
    """
    points = _param_to_points(params)
    areas = _all_triangle_areas(points)
    neg_beta_areas = -beta * areas
    max_neg = np.max(neg_beta_areas)
    log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_beta_areas - max_neg)))
    soft_min = -log_sum_exp / beta
    return -soft_min

def _soft_min_grad(params, beta):
    """
    Gradient of the soft-min objective w.r.t. params.
    d/dp [-soft_min] where soft_min = -logsumexp(-beta*areas)/beta
    = sum_k w_k * d(area_k)/dp, where w_k = softmax(-beta*areas)
    """
    points = _param_to_points(params)
    areas = _all_triangle_areas(points)
    
    # Softmax weights: w_k = exp(-beta * a_k) / sum_j exp(-beta * a_j)
    neg_beta_areas = -beta * areas
    max_neg = np.max(neg_beta_areas)
    exp_neg = np.exp(neg_beta_areas - max_neg)
    weights = exp_neg / np.sum(exp_neg)
    
    i_idx, j_idx, k_idx = _get_indices()
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    
    sign_cross = np.sign(cross)
    sign_cross[np.abs(cross) < 1e-15] = 0.0
    
    grad_points = np.zeros((N, 2))
    
    # area_k = 0.5 * |cross_k|, d(area_k)/d(point) depends on sign of cross
    w = weights * sign_cross * 0.5
    
    # Contribution from pi: d(cross)/d(pi) = -(pk - pi) x (pj - pi) in 2D
    # cross = (pjx-pix)(pky-piy) - (pjy-piy)(pkx-pix)
    # d(cross)/d(pix) = -(pky-piy) + (pjy-piy) = pjy - pky
    # d(cross)/d(piy) = -(pkx-pix) + (pjx-pix) = pjx - pkx
    d_cross_pi_x = pj[:, 1] - pk[:, 1]
    d_cross_pi_y = pj[:, 0] - pk[:, 0]
    np.add.at(grad_points, i_idx, w[:, np.newaxis] * np.column_stack([d_cross_pi_x, d_cross_pi_y]))
    
    # Contribution from pj:
    # d(cross)/d(pjx) = pky - piy
    # d(cross)/d(pjy) = -(pkx - pix)
    d_cross_pj_x = pk[:, 1] - pi[:, 1]
    d_cross_pj_y = -(pk[:, 0] - pi[:, 0])
    np.add.at(grad_points, j_idx, w[:, np.newaxis] * np.column_stack([d_cross_pj_x, d_cross_pj_y]))
    
    # Contribution from pk:
    # d(cross)/d(pkx) = -(pjy - piy)
    # d(cross)/d(pky) = pjx - pix
    d_cross_pk_x = -(pj[:, 1] - pi[:, 1])
    d_cross_pk_y = pj[:, 0] - pi[:, 0]
    np.add.at(grad_points, k_idx, w[:, np.newaxis] * np.column_stack([d_cross_pk_x, d_cross_pk_y]))
    
    s = params[0::2]
    t = params[1::2]
    s = np.clip(s, 1e-10, 1.0 - 1e-10)
    t = np.clip(t, 1e-10, 1.0 - 1e-10)
    
    # Barycentric to Cartesian Jacobian
    # x = lam1*V1x + lam2*V2x = s(1-t)*V1x + s*t*V2x
    # dx/ds = (1-t)*V1x + t*V2x
    # dx/dt = -s*V1x + s*V2x
    dx_ds = (1.0 - t) * V1[0] + t * V2[0]
    dx_dt = s * (V2[0] - V1[0])
    dy_ds = (1.0 - t) * V1[1] + t * V2[1]
    dy_dt = s * (V2[1] - V1[1])
    
    grad_params = np.zeros(2 * N)
    grad_params[0::2] = grad_points[:, 0] * dx_ds + grad_points[:, 1] * dy_ds
    grad_params[1::2] = grad_points[:, 0] * dx_dt + grad_points[:, 1] * dy_dt
    
    # We're maximizing soft_min, so gradient of objective (-soft_min) is -grad_params
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

def _structured_inits(rng):
    """
    Generate structured initializations based on grid or symmetric patterns.
    """
    inits = []
    
    # Grid-like patterns
    for grid_n in [3, 4, 5]:
        points = []
        for i in range(grid_n):
            for j in range(grid_n - i):
                l0 = 1 - (i + j) / (grid_n - 1) if grid_n > 1 else 1
                l1 = i / (grid_n - 1) if grid_n > 1 else 0
                l2 = j / (grid_n - 1) if grid_n > 1 else 0
                pt = l0 * V0 + l1 * V1 + l2 * V2
                points.append(pt)
        
        points = np.array(points)
        if len(points) >= N:
            idx = rng.choice(len(points), N, replace=False)
            inits.append(points[idx])
        elif len(points) > 0:
            while len(points) < N:
                r1 = rng.uniform(0, 1)
                r2 = rng.uniform(0, 1)
                if r1 + r2 > 1:
                    r1 = 1 - r1
                    r2 = 1 - r2
                pt = r1 * V1 + r2 * V2
                points = np.vstack([points, pt])
            inits.append(points)
    
    # Symmetric patterns: 1 center + 10 around
    center = (V0 + V1 + V2) / 3.0
    pts = [center]
    for angle in np.linspace(0, 2*np.pi, 5, endpoint=False):
        r = 0.3
        pt = center + r * np.array([np.cos(angle), np.sin(angle)])
        pts.append(pt)
    for angle in np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5:
        r = 0.6
        pt = center + r * np.array([np.cos(angle), np.sin(angle)])
        pts.append(pt)
    
    inits.append(np.array(pts))
    
    return inits

def _simulated_annealing(params, rng, max_iter=5000, t_start=0.05, t_end=0.0005):
    """
    Simulated annealing to explore the parameter space with adaptive step sizes.
    """
    current_params = params.copy()
    current_score = -_objective(current_params)
    best_params_sa = current_params.copy()
    best_score_sa = current_score
    
    n_steps = max_iter
    cooling_rate = (t_end / t_start) ** (1.0 / max(1, n_steps - 1))
    
    # Adaptive step size
    step_size = 0.08
    accept_count = 0
    total_count = 0
    
    for step in range(n_steps):
        if time.time() > deadline_global:
            break
        
        T = t_start * (cooling_rate ** step)
        
        # Generate neighbor with adaptive step size
        r = rng.random()
        if r < 0.2:
            # Swap two points
            i, j = rng.choice(N, 2, replace=False)
            new_params = current_params.copy()
            new_params[i*2], new_params[j*2] = new_params[j*2], new_params[i*2]
            new_params[i*2+1], new_params[j*2+1] = new_params[j*2+1], new_params[i*2+1]
        elif r < 0.6:
            # Local jitter: Gaussian perturbation on all points
            noise = rng.normal(0, step_size, size=2*N)
            new_params = np.clip(current_params + noise, 1e-10, 1.0 - 1e-10)
        elif r < 0.85:
            # Perturb one or two points significantly
            n_perturb = rng.randint(1, 3)
            idx = rng.choice(N, n_perturb, replace=False)
            new_params = current_params.copy()
            for i in idx:
                new_params[i*2] = np.clip(current_params[i*2] + rng.normal(0, step_size * 2), 1e-10, 1.0 - 1e-10)
                new_params[i*2+1] = np.clip(current_params[i*2+1] + rng.normal(0, step_size * 2), 1e-10, 1.0 - 1e-10)
        else:
            # Large restart-like move: perturb 3-4 points
            n_perturb = rng.randint(3, 5)
            idx = rng.choice(N, n_perturb, replace=False)
            new_params = current_params.copy()
            for i in idx:
                new_params[i*2] = np.clip(rng.uniform(0.05, 0.95), 1e-10, 1.0 - 1e-10)
                new_params[i*2+1] = np.clip(rng.uniform(0.05, 0.95), 1e-10, 1.0 - 1e-10)
        
        new_score = -_objective(new_params)
        delta = new_score - current_score
        total_count += 1
        
        # Accept if better or with probability exp(delta/T)
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-12)):
            current_params = new_params
            current_score = new_score
            accept_count += 1
            if current_score > best_score_sa:
                best_score_sa = current_score
                best_params_sa = current_params.copy()
        
        # Adapt step size every 200 steps
        if total_count % 200 == 0:
            acceptance_rate = accept_count / total_count
            if acceptance_rate > 0.6:
                step_size *= 1.3
            elif acceptance_rate < 0.2:
                step_size *= 0.7
            step_size = np.clip(step_size, 0.01, 0.3)
            accept_count = 0
            total_count = 0
    
    return best_params_sa, best_score_sa

def construct():
    global deadline_global
    start_time = time.time()
    deadline_global = start_time + 140.0
    
    def time_left():
        return deadline_global - time.time()

    best_params = None
    best_score = -np.inf
    
    rng = np.random.RandomState(42)
    
    initial_params = []
    
    # 1. Generate diverse initializations
    n_fps_inits = 8
    for i in range(n_fps_inits):
        if time_left() < 50.0:
            break
        pts = _generate_fps_initialization(rng)
        params = _points_to_param(pts)
        initial_params.append(params)
        
    # Add structured inits
    for pts in _structured_inits(rng):
        if time_left() < 50.0:
            break
        params = _points_to_param(pts)
        initial_params.append(params)

    # Add random initializations
    for i in range(8):
        if time_left() < 50.0:
            break
        s = rng.uniform(0.1, 0.9, size=N)
        t = rng.uniform(0.1, 0.9, size=N)
        params = np.zeros(2 * N)
        params[0::2] = s
        params[1::2] = t
        initial_params.append(params)

    bounds = [(0.0, 1.0)] * (2 * N)
    
    # 2. Optimize each initialization using SLSQP
    for start_p in initial_params:
        if time_left() < 25.0:
            break
            
        try:
            res = minimize(
                _objective, start_p, method='SLSQP', bounds=bounds,
                options={'maxiter': 150, 'ftol': 1e-10}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass

    # 3. Use soft-min gradient descent (L-BFGS-B) with multiple betas
    if best_params is not None and time_left() > 40.0:
        for beta in [30, 60, 120, 250]:
            if time_left() < 20.0:
                break
            try:
                res = minimize(
                    lambda p, b=beta: _soft_min_objective(p, b),
                    best_params,
                    jac=lambda p, b=beta: _soft_min_grad(p, b),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                true_score = -_objective(res.x)
                if true_score > best_score:
                    best_score = true_score
                    best_params = res.x.copy()
            except Exception:
                pass

    # 4. Simulated annealing to escape local minima
    if best_params is not None and time_left() > 35.0:
        try:
            sa_params, sa_score = _simulated_annealing(best_params, rng, max_iter=2500)
            if sa_score > best_score:
                best_score = sa_score
                best_params = sa_params.copy()
        except Exception:
            pass

    # 5. Polish the best solution found with Nelder-Mead
    if best_params is not None and time_left() > 20.0:
        try:
            res = minimize(
                _objective, best_params, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass

    # 6. More soft-min refinement after SA
    if best_params is not None and time_left() > 15.0:
        for beta in [100, 300]:
            if time_left() < 10.0:
                break
            try:
                res = minimize(
                    lambda p, b=beta: _soft_min_objective(p, b),
                    best_params,
                    jac=lambda p, b=beta: _soft_min_grad(p, b),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12}
                )
                true_score = -_objective(res.x)
                if true_score > best_score:
                    best_score = true_score
                    best_params = res.x.copy()
            except Exception:
                pass

    # 7. Final high-precision Nelder-Mead polish
    if best_params is not None and time_left() > 8.0:
        try:
            res = minimize(
                _objective, best_params, method='Nelder-Mead',
                options={'maxiter': 8000, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass

    if best_params is None:
        # Fallback: Centered grid
        h = np.sqrt(3) / 2
        fallback_points = np.array([
            [0.5, 0.0], [0.0, 0.0], [1.0, 0.0],
            [0.25, h/2], [0.75, h/2], [0.5, h],
            [0.5, h/4], [0.25, h/4], [0.75, h/4],
            [0.5, 3*h/4], [0.5, 2*h/4]
        ])
        return fallback_points

    # Convert best params to points
    best_points = _param_to_points(best_params)
    
    return best_points

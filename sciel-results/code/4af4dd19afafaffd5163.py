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
    soft_min(areas) = min_a - (1/beta) * log(sum(exp(-beta * (areas - min_a))))
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
    weights = exp_neg / sum_exp
    
    i_idx, j_idx, k_idx = _get_indices()
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    
    sign_cross = np.sign(cross)
    sign_cross[np.abs(cross) < 1e-15] = 1.0
    
    d_cross_pi_x = pj[:, 1] - pk[:, 1]
    d_cross_pi_y = pk[:, 0] - pj[:, 0]
    
    d_cross_pj_x = pk[:, 1] - pi[:, 1]
    d_cross_pj_y = -(pk[:, 0] - pi[:, 0])
    
    d_cross_pk_x = -(pj[:, 1] - pi[:, 1])
    d_cross_pk_y = pj[:, 0] - pi[:, 0]
    
    w = weights * sign_cross * 0.5
    
    grad_points = np.zeros((N, 2))
    
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

def _squared_area_qp_objective(params, t):
    """
    Objective for trust-constr QP: minimize t (we maximize min area).
    Actually we minimize -t subject to area >= t for all triples.
    We reformulate: minimize t, subject to -area_i + t <= 0 for all i.
    So objective is t, and constraints are -area_i + t <= 0.
    """
    return t

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

def _simulated_annealing(params, rng, max_iter=5000, t_start=0.1, t_end=0.001, deadline=None):
    """
    Simulated annealing to explore the parameter space.
    """
    current_params = params.copy()
    current_score = -_objective(current_params)
    best_params_sa = current_params.copy()
    best_score_sa = current_score
    
    n_steps = max_iter
    cooling_rate = (t_end / t_start) ** (1.0 / max(1, n_steps - 1))
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        T = t_start * (cooling_rate ** step)
        
        r = rng.random()
        if r < 0.3:
            i, j = rng.choice(N, 2, replace=False)
            new_params = current_params.copy()
            new_params[i*2], new_params[j*2] = new_params[j*2], new_params[i*2]
            new_params[i*2+1], new_params[j*2+1] = new_params[j*2+1], new_params[i*2+1]
        elif r < 0.7:
            noise = rng.normal(0, 0.05, size=2*N)
            new_params = np.clip(current_params + noise, 0.0, 1.0)
        else:
            i = rng.randint(N)
            new_params = current_params.copy()
            new_params[i*2] = np.clip(current_params[i*2] + rng.normal(0, 0.2), 0.0, 1.0)
            new_params[i*2+1] = np.clip(current_params[i*2+1] + rng.normal(0, 0.2), 0.0, 1.0)
        
        new_score = -_objective(new_params)
        delta = new_score - current_score
        
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
            current_params = new_params
            current_score = new_score
            if current_score > best_score_sa:
                best_score_sa = current_score
                best_params_sa = current_params.copy()
    
    return best_params_sa, best_score_sa

def _qp_trust_constr(params_init, deadline=None):
    """
    Trust-constr QP: maximize t subject to area_i >= t for all triples.
    Parameters: [s_1, t_1, s_2, t_2, ..., s_N, t_N, tau]
    Objective: minimize -tau
    Constraints: -area_i + tau <= 0 for all i
    """
    n_vars = 2 * N + 1
    
    def qp_objective(x):
        return -x[-1]
    
    def qp_objective_grad(x):
        g = np.zeros(n_vars)
        g[-1] = -1.0
        return g
    
    i_idx, j_idx, k_idx = _get_indices()
    n_constraints = len(i_idx)
    
    def qp_constraints(x):
        # x[:2N] are point params, x[-1] is tau
        pts = _param_to_points(x[:2*N])
        pi = pts[i_idx]
        pj = pts[j_idx]
        pk = pts[k_idx]
        cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
        areas = 0.5 * np.abs(cross)
        tau = x[-1]
        return -(areas - tau)
    
    def qp_constraint_jacob(x):
        # Jacobian of each constraint w.r.t. x
        pts = _param_to_points(x[:2*N])
        pi = pts[i_idx]
        pj = pts[j_idx]
        pk = pts[k_idx]
        cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
        
        sign_cross = np.sign(cross)
        sign_cross[np.abs(cross) < 1e-15] = 1.0
        
        d_cross_pi_x = pj[:, 1] - pk[:, 1]
        d_cross_pi_y = pk[:, 0] - pj[:, 0]
        d_cross_pj_x = pk[:, 1] - pi[:, 1]
        d_cross_pj_y = -(pk[:, 0] - pi[:, 0])
        d_cross_pk_x = -(pj[:, 1] - pi[:, 1])
        d_cross_pk_y = pj[:, 0] - pi[:, 0]
        
        # d(area)/d(point) for area = 0.5*|cross|
        # Use signed gradient
        w = sign_cross * 0.5
        
        grad_points = np.zeros((N, 2))
        np.add.at(grad_points, i_idx, w[:, np.newaxis] * np.column_stack([d_cross_pi_x, d_cross_pi_y]))
        np.add.at(grad_points, j_idx, w[:, np.newaxis] * np.column_stack([d_cross_pj_x, d_cross_pj_y]))
        np.add.at(grad_points, k_idx, w[:, np.newaxis] * np.column_stack([d_cross_pk_x, d_cross_pk_y]))
        
        s = x[:2*N:2]
        t = x[1:2*N:2]
        s = np.clip(s, 1e-10, 1.0 - 1e-10)
        t = np.clip(t, 1e-10, 1.0 - 1e-10)
        
        dx_ds = (1 - t) * V1[0] + t * V2[0]
        dx_dt = -s * V1[0] + s * V2[0]
        dy_ds = (1 - t) * V1[1] + t * V2[1]
        dy_dt = -s * V1[1] + s * V2[1]
        
        grad_params = np.zeros(2 * N)
        grad_params[0::2] = grad_points[:, 0] * dx_ds + grad_points[:, 1] * dy_ds
        grad_params[1::2] = grad_points[:, 0] * dx_dt + grad_points[:, 1] * dy_dt
        
        # Constraint gradient: d(-area + tau)/dx
        # For point params: -grad_params (broadcast to each constraint)
        # For tau: 1
        J = np.zeros((n_constraints, n_vars))
        # Each constraint c_i has gradient: -grad_params for the 2N point vars, 1 for tau
        # But each point var affects multiple constraints, so we need the full sparse structure
        # For simplicity and since n_constraints is 286 and n_vars is 23, build dense J
        for idx in range(n_constraints):
            # The gradient of constraint idx w.r.t. point params is -grad_points contribution
            # This is complex; use finite difference or skip Jacobian
            pass
        
        # Use a simpler approach: return None to let trust-constr use finite differences
        # Or we can build it properly
        J = None
        return J
    
    bounds = [(0.0, 1.0)] * (2 * N) + [(-1.0, 0.5)]
    
    x0 = np.append(params_init, 0.0)
    
    try:
        from scipy.optimize import LinearConstraint
        # We'll use the general constraint with inequality
        constraints = [
            {'type': 'ineq', 'fun': lambda x: -(qp_constraints(x))}
        ]
        
        res = minimize(
            qp_objective, x0, method='trust-constr',
            jac=qp_objective_grad,
            constraints=constraints,
            bounds=bounds,
            options={'maxiter': 300, 'gtol': 1e-6, 'verbose': 0}
        )
        
        if res.success or res.x is not None:
            new_params = res.x[:2*N]
            score = -_objective(new_params)
            return new_params, score
    except Exception:
        pass
    
    return None, -np.inf

def construct():
    start_time = time.time()
    deadline_global = start_time + 140.0
    
    def time_left():
        return deadline_global - time.time()

    best_params = None
    best_score = -np.inf
    
    rng = np.random.RandomState(42)
    
    initial_params = []
    
    n_fps_inits = 10
    for i in range(n_fps_inits):
        if time_left() < 40.0:
            break
        pts = _generate_fps_initialization(rng)
        params = _points_to_param(pts)
        initial_params.append(params)
        
    for pts in _structured_inits(rng):
        if time_left() < 40.0:
            break
        params = _points_to_param(pts)
        initial_params.append(params)

    for i in range(10):
        if time_left() < 40.0:
            break
        s = rng.uniform(0.1, 0.9, size=N)
        t = rng.uniform(0.1, 0.9, size=N)
        params = np.zeros(2 * N)
        params[0::2] = s
        params[1::2] = t
        initial_params.append(params)

    bounds = [(0.0, 1.0)] * (2 * N)
    
    for start_p in initial_params:
        if time_left() < 20.0:
            break
            
        try:
            res = minimize(
                _objective, start_p, method='SLSQP', bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-9}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass

    if best_params is not None and time_left() > 50.0:
        try:
            sa_params, sa_score = _simulated_annealing(best_params, rng, max_iter=3000, deadline=deadline_global)
            if sa_score > best_score:
                best_score = sa_score
                best_params = sa_params.copy()
        except Exception:
            pass

    if best_params is not None and time_left() > 30.0:
        try:
            res = minimize(
                _objective, best_params, method='Nelder-Mead',
                options={'maxiter': 8000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass

    if best_params is not None and time_left() > 25.0:
        for i in range(5):
            if time_left() < 15.0:
                break
            try:
                noise = rng.normal(0, 0.02, size=2*N)
                perturbed = np.clip(best_params + noise, 0.0, 1.0)
                res = minimize(
                    _objective, perturbed, method='SLSQP', bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-9}
                )
                score_res = -res.fun
                if score_res > best_score:
                    best_score = score_res
                    best_params = res.x.copy()
            except Exception:
                pass

    if best_params is not None and time_left() > 20.0:
        for beta in [30, 50, 100, 200, 400]:
            if time_left() < 12.0:
                break
            try:
                res = minimize(
                    lambda p: _soft_min_objective(p, beta),
                    best_params,
                    jac=lambda p: _soft_min_grad(p, beta),
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

    # NEW: Trust-constr QP refinement
    if best_params is not None and time_left() > 25.0:
        try:
            qp_params, qp_score = _qp_trust_constr(best_params, deadline=deadline_global)
            if qp_params is not None and qp_score > best_score:
                best_score = qp_score
                best_params = qp_params.copy()
        except Exception:
            pass

    if best_params is not None and time_left() > 10.0:
        try:
            res = minimize(
                _objective, best_params, method='Nelder-Mead',
                options={'maxiter': 15000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is not None and time_left() > 8.0:
        for i in range(3):
            if time_left() < 5.0:
                break
            try:
                noise = rng.normal(0, 0.005, size=2*N)
                perturbed = np.clip(best_params + noise, 0.0, 1.0)
                res = minimize(
                    _objective, perturbed, method='SLSQP', bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12}
                )
                score_res = -res.fun
                if score_res > best_score:
                    best_score = score_res
                    best_params = res.x.copy()
            except Exception:
                pass

    if best_params is None:
        h = np.sqrt(3) / 2
        fallback_points = np.array([
            [0.5, 0.0], [0.0, 0.0], [1.0, 0.0],
            [0.25, h/2], [0.75, h/2], [0.5, h],
            [0.5, h/4], [0.25, h/4], [0.75, h/4],
            [0.5, 3*h/4], [0.5, 2*h/4]
        ])
        return fallback_points

    best_points = _param_to_points(best_params)
    
    return best_points

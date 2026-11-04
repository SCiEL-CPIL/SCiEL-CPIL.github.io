"""Heilbronn triangles: maximize the smallest triangle area formed by 11 points in an equilateral triangle."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint

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
    Smooth surrogate: negative soft-min of triangle areas via log-sum-exp.
    soft_min = -logsumexp(-beta * areas) / beta
    """
    points = _param_to_points(params)
    areas = _all_triangle_areas(points)
    neg_beta_areas = -beta * areas
    max_neg = np.max(neg_beta_areas)
    log_sum = np.log(np.sum(np.exp(neg_beta_areas - max_neg))) + max_neg
    soft_min = -log_sum / beta
    return -soft_min

def _soft_min_grad(params, beta):
    """
    Gradient of the soft-min objective w.r.t. params using log-sum-exp.
    """
    points = _param_to_points(params)
    areas = _all_triangle_areas(points)
    
    neg_beta_areas = -beta * areas
    max_neg = np.max(neg_beta_areas)
    exp_neg = np.exp(neg_beta_areas - max_neg)
    sum_exp = np.sum(exp_neg)
    weights = exp_neg / sum_exp  # softmax weights
    
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
    
    cooling_rate = (t_end / t_start) ** (1.0 / max(1, max_iter - 1))
    
    for step in range(max_iter):
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

def _squared_area_qp(params_init, deadline=None):
    """
    Trust-constr QP using squared areas to avoid non-smoothness of abs().
    
    Formulation:
      Variables: x = [s_1, t_1, ..., s_N, t_N, tau]  (2N+1 vars)
      Minimize:   -tau
      Subject to: (cross_i)^2 / 4 >= tau^2  for all i
                 i.e., (cross_i)^2 - 4*tau^2 >= 0
                 i.e., -(cross_i)^2 + 4*tau^2 <= 0  (for LinearConstraint with <= 0)
    
    Actually for LinearConstraint we need: A(x) >= lb and A(x) <= ub
    We want: (cross_i)^2 >= 4*tau^2  =>  (cross_i)^2 - 4*tau^2 >= 0
    
    So we set the constraint function to f_i = (cross_i)^2 - 4*tau^2, and require f_i >= 0.
    
    But wait: cross depends on points which depend on params. Let's use a simpler approach:
    Maximize tau subject to area_i >= tau for all i.
    area_i = 0.5 * |cross_i|
    
    To avoid abs, we can use: area_i^2 >= tau^2  AND  area_i >= 0 (always true)
    But area_i^2 = cross_i^2 / 4, so: cross_i^2 / 4 >= tau^2 => cross_i^2 - 4*tau^2 >= 0
    
    This is a smooth constraint! The gradient exists everywhere.
    
    Objective: minimize -tau (i.e., maximize tau)
    Constraint: g_i(x) = cross_i(x)^2 - 4*tau^2 >= 0 for all i
    
    For LinearConstraint: we need lb <= A(x) <= ub
    So A(x) = g_i(x), lb = 0, ub = inf
    """
    n_vars = 2 * N + 1
    i_idx, j_idx, k_idx = _get_indices()
    n_constraints = len(i_idx)
    
    def qp_objective(x):
        return -x[-1]
    
    def qp_objective_grad(x):
        g = np.zeros(n_vars)
        g[-1] = -1.0
        return g
    
    def qp_constraint_fun(x):
        """Compute g_i = cross_i^2 - 4*tau^2 for all i."""
        pts = _param_to_points(x[:2*N])
        pi = pts[i_idx]
        pj = pts[j_idx]
        pk = pts[k_idx]
        cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
        tau = x[-1]
        return cross**2 - 4.0 * tau**2
    
    def qp_constraint_jac(x):
        """
        Jacobian of g_i = cross_i^2 - 4*tau^2 w.r.t. x.
        
        d(cross_i^2)/dx = 2 * cross_i * d(cross_i)/dx
        d(-4*tau^2)/dx = 0 for point params, -8*tau for tau
        
        d(cross)/d(pi_x) = pj_y - pk_y
        d(cross)/d(pi_y) = pk_x - pj_x
        d(cross)/d(pj_x) = pk_y - pi_y
        d(cross)/d(pj_y) = -(pk_x - pi_x)
        d(cross)/d(pk_x) = -(pj_y - pi_y)
        d(cross)/d(pk_y) = pj_x - pi_x
        """
        pts = _param_to_points(x[:2*N])
        pi = pts[i_idx]
        pj = pts[j_idx]
        pk = pts[k_idx]
        cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
        tau = x[-1]
        
        # d(cross^2)/d(point) = 2 * cross * d(cross)/d(point)
        d_cross_pi_x = pj[:, 1] - pk[:, 1]
        d_cross_pi_y = pk[:, 0] - pj[:, 0]
        d_cross_pj_x = pk[:, 1] - pi[:, 1]
        d_cross_pj_y = -(pk[:, 0] - pi[:, 0])
        d_cross_pk_x = -(pj[:, 1] - pi[:, 1])
        d_cross_pk_y = pj[:, 0] - pi[:, 0]
        
        # Gradient of cross^2 w.r.t. each point coordinate
        # For each constraint i, the gradient w.r.t. point p_m is:
        # 2 * cross_i * (contribution of p_m to cross_i)
        
        # Build gradient w.r.t. points first
        grad_points = np.zeros((N, 2))
        
        # 2 * cross * d_cross_pi
        np.add.at(grad_points, i_idx, (2.0 * cross)[:, np.newaxis] * np.column_stack([d_cross_pi_x, d_cross_pi_y]))
        np.add.at(grad_points, j_idx, (2.0 * cross)[:, np.newaxis] * np.column_stack([d_cross_pj_x, d_cross_pj_y]))
        np.add.at(grad_points, k_idx, (2.0 * cross)[:, np.newaxis] * np.column_stack([d_cross_pk_x, d_cross_pk_y]))
        
        # Chain rule: points -> params
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
        
        # Now build the full Jacobian matrix J of shape (n_constraints, n_vars)
        # J[i, 0:2N] = grad_params for constraint i ... but wait, grad_params is per-point, not per-constraint
        # We need J[i, m] = d(g_i)/d(x_m) for each constraint i and variable m
        
        # This requires per-constraint gradients. Let's compute them directly.
        # For constraint i (triple (i_idx[i], j_idx[i], k_idx[i])):
        # d(g_i)/d(point_m) is non-zero only if m is in {i_idx[i], j_idx[i], k_idx[i]}
        
        # Build J as dense matrix
        J = np.zeros((n_constraints, n_vars))
        
        # For each constraint row i:
        # d(g_i)/d(pi) = 2*cross_i * d_cross_pi  (where pi = points[i_idx[i]])
        # d(g_i)/d(pj) = 2*cross_i * d_cross_pj
        # d(g_i)/d(pk) = 2*cross_i * d_cross_pk
        # d(g_i)/d(tau) = -8*tau
        
        # Map point indices to param indices: point m has params at (2m, 2m+1)
        # d(param_2m)/d(point_x) = dx_ds for s, dx_dt for t
        # Actually: point_x = s*(1-t)*V1x + s*t*V2x
        # d(point_x)/d(s) = (1-t)*V1x + t*V2x = dx_ds
        # d(point_x)/d(t) = -s*V1x + s*V2x = dx_dt
        # So d(g_i)/d(s_m) = d(g_i)/d(point_x_m) * dx_ds_m + d(g_i)/d(point_y_m) * dy_ds_m
        #    d(g_i)/d(t_m) = d(g_i)/d(point_x_m) * dx_dt_m + d(g_i)/d(point_y_m) * dy_dt_m
        
        # Precompute per-point gradients for all constraints
        # For point p_m, the total gradient from all constraints involving m
        # But for the Jacobian, we need per-constraint
        
        # Efficient approach: for each constraint i, compute the 3 point gradients
        # and map to param space
        
        # d(g_i)/d(point at i_idx[i]) = 2*cross[i] * [d_cross_pi_x[i], d_cross_pi_y[i]]
        # d(g_i)/d(point at j_idx[i]) = 2*cross[i] * [d_cross_pj_x[i], d_cross_pj_y[i]]
        # d(g_i)/d(point at k_idx[i]) = 2*cross[i] * [d_cross_pk_x[i], d_cross_pk_y[i]]
        
        # For point m, d(g_i)/d(s_m) and d(g_i)/d(t_m):
        # If m == i_idx[i]: d(g_i)/d(s_m) = 2*cross[i] * (d_cross_pi_x[i]*dx_ds[m] + d_cross_pi_y[i]*dy_ds[m])
        #                    d(g_i)/d(t_m) = 2*cross[i] * (d_cross_pi_x[i]*dx_dt[m] + d_cross_pi_y[i]*dy_dt[m])
        # Similarly for j and k.
        
        # Let's compute per-point contributions
        # For each constraint i, add to J[i, 2*m] and J[i, 2*m+1] for m in {i_idx[i], j_idx[i], k_idx[i]}
        
        # Point i_idx[i]:
        m = i_idx
        J[np.arange(n_constraints), 2*m] = 2.0 * cross * (d_cross_pi_x * dx_ds[m] + d_cross_pi_y * dy_ds[m])
        J[np.arange(n_constraints), 2*m+1] = 2.0 * cross * (d_cross_pi_x * dx_dt[m] + d_cross_pi_y * dy_dt[m])
        
        # Point j_idx[i]:
        m = j_idx
        J[np.arange(n_constraints), 2*m] += 2.0 * cross * (d_cross_pj_x * dx_ds[m] + d_cross_pj_y * dy_ds[m])
        J[np.arange(n_constraints), 2*m+1] += 2.0 * cross * (d_cross_pj_x * dx_dt[m] + d_cross_pj_y * dy_dt[m])
        
        # Point k_idx[i]:
        m = k_idx
        J[np.arange(n_constraints), 2*m] += 2.0 * cross * (d_cross_pk_x * dx_ds[m] + d_cross_pk_y * dy_ds[m])
        J[np.arange(n_constraints), 2*m+1] += 2.0 * cross * (d_cross_pk_x * dx_dt[m] + d_cross_pk_y * dy_dt[m])
        
        # d(g_i)/d(tau) = -8*tau for all i
        J[:, -1] = -8.0 * tau
        
        return J
    
    bounds = [(0.0, 1.0)] * (2 * N) + [(-1.0, 0.5)]
    
    x0 = np.append(params_init.copy(), 0.01)
    
    try:
        constraint = LinearConstraint(qp_constraint_jac, lb=0.0, ub=np.inf,
                                       jac_sparsity=None)
        # Actually LinearConstraint expects a function that returns the constraint values
        # and optionally a Jacobian. Let's use the dict form instead for clarity.
        
        constraints = [
            {
                'type': 'ineq',
                'fun': qp_constraint_fun,
                'jac': qp_constraint_jac
            }
        ]
        
        res = minimize(
            qp_objective, x0, method='trust-constr',
            jac=qp_objective_grad,
            constraints=constraints,
            bounds=bounds,
            options={'maxiter': 500, 'gtol': 1e-8, 'verbose': 0}
        )
        
        if res.x is not None:
            new_params = res.x[:2*N]
            score = -_objective(new_params)
            return new_params, score
    except Exception as e:
        pass
    
    return None, -np.inf

def construct():
    start_time = time.time()
    deadline_global = start_time + 140.0  # Safety margin before 150s kill

    def time_left():
        return deadline_global - time.time()

    best_params = None
    best_score = -np.inf
    
    rng = np.random.RandomState(42)
    
    initial_params = []
    
    # Generate diverse initializations
    n_fps_inits = 8
    for i in range(n_fps_inits):
        if time_left() < 60.0:
            break
        pts = _generate_fps_initialization(rng)
        params = _points_to_param(pts)
        initial_params.append(params)
        
    for pts in _structured_inits(rng):
        if time_left() < 60.0:
            break
        params = _points_to_param(pts)
        initial_params.append(params)

    for i in range(8):
        if time_left() < 60.0:
            break
        s = rng.uniform(0.1, 0.9, size=N)
        t = rng.uniform(0.1, 0.9, size=N)
        params = np.zeros(2 * N)
        params[0::2] = s
        params[1::2] = t
        initial_params.append(params)

    bounds = [(0.0, 1.0)] * (2 * N)
    
    # Phase 1: SLSQP from multiple starts
    for start_p in initial_params:
        if time_left() < 50.0:
            break
            
        try:
            res = minimize(
                _objective, start_p, method='SLSQP', bounds=bounds,
                options={'maxiter': 150, 'ftol': 1e-9}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 2: Simulated annealing
    if best_params is not None and time_left() > 55.0:
        try:
            sa_params, sa_score = _simulated_annealing(best_params, rng, max_iter=4000, deadline=deadline_global - 40.0)
            if sa_score > best_score:
                best_score = sa_score
                best_params = sa_params.copy()
        except Exception:
            pass

    # Phase 3: Nelder-Mead refinement
    if best_params is not None and time_left() > 40.0:
        try:
            res = minimize(
                _objective, best_params, method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 4: Perturbation + SLSQP
    if best_params is not None and time_left() > 35.0:
        for i in range(4):
            if time_left() < 25.0:
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

    # Phase 5: Soft-min L-BFGS-B with analytical gradient
    if best_params is not None and time_left() > 30.0:
        for beta in [50, 100, 200, 400, 800]:
            if time_left() < 18.0:
                break
            try:
                res = minimize(
                    lambda p, b=beta: _soft_min_objective(p, b),
                    best_params,
                    jac=lambda p, b=beta: _soft_min_grad(p, b),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-11}
                )
                true_score = -_objective(res.x)
                if true_score > best_score:
                    best_score = true_score
                    best_params = res.x.copy()
            except Exception:
                pass

    # Phase 6: Squared-area QP via trust-constr (smooth, no abs)
    if best_params is not None and time_left() > 25.0:
        try:
            qp_params, qp_score = _squared_area_qp(best_params, deadline=deadline_global - 15.0)
            if qp_params is not None and qp_score > best_score:
                best_score = qp_score
                best_params = qp_params.copy()
        except Exception:
            pass

    # Phase 7: Final high-precision Nelder-Mead polish
    if best_params is not None and time_left() > 15.0:
        try:
            res = minimize(
                _objective, best_params, method='Nelder-Mead',
                options={'maxiter': 20000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 8: Final tiny perturbations + SLSQP
    if best_params is not None and time_left() > 10.0:
        for i in range(3):
            if time_left() < 6.0:
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

    # Fallback
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
    
    # Verify points are inside the triangle (with small tolerance)
    # Barycentric check: all coords should be >= -1e-9
    x = best_points[:, 0]
    y = best_points[:, 1]
    # For triangle (0,0), (1,0), (0.5, sqrt(3)/2):
    # y >= 0, y <= sqrt(3)/2 * min(x, 1-x) ... actually:
    # The triangle is: y >= 0, y <= sqrt(3)*x, y <= sqrt(3)*(1-x)
    # Let's just clip to be safe
    h = np.sqrt(3) / 2
    for i in range(N):
        xi, yi = best_points[i]
        if yi < 0:
            best_points[i, 1] = 0.0
        if yi > h:
            best_points[i, 1] = h
        if xi < 0:
            best_points[i, 0] = 0.0
        if xi > 1:
            best_points[i, 0] = 1.0
        # Check if inside: y <= sqrt(3)*x and y <= sqrt(3)*(1-x)
        max_y_left = np.sqrt(3) * xi
        max_y_right = np.sqrt(3) * (1.0 - xi)
        max_y = min(max_y_left, max_y_right)
        if best_points[i, 1] > max_y:
            best_points[i, 1] = max_y
    
    return best_points

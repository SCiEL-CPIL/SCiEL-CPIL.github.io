"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the smallest triangle area."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0
H = np.sqrt(3) / 2.0

# Precompute indices for all C(11,3) = 165 triples
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_N_TRIPLES = _IDX_TRIPLES.shape[0]

def _bary_to_cart(bary):
    """Convert barycentric coords (u, v) to cartesian.
    Point = u*V1 + v*V2 + (1-u-v)*V0 = u*V1 + v*V2
    """
    x = bary[:, 0] * V1[0] + bary[:, 1] * V2[0]
    y = bary[:, 0] * V1[1] + bary[:, 1] * V2[1]
    return np.column_stack([x, y])

def _clamp_bary(bary):
    """Clamp barycentric coordinates to valid region u>=0, v>=0, u+v<=1."""
    bary = np.clip(bary, 0.0, None)
    s = bary[:, 0] + bary[:, 1]
    mask = s > 1.0
    if np.any(mask):
        bary[mask] = bary[mask] / s[mask, None]
    return bary

def _compute_areas(points):
    """Compute areas of all 165 triangles formed by the points."""
    pi = points[_IDX_TRIPLES[:, 0]]
    pj = points[_IDX_TRIPLES[:, 1]]
    pk = points[_IDX_TRIPLES[:, 2]]
    
    d1 = pj - pi
    d2 = pk - pi
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    areas = 0.5 * np.abs(cross)
    return areas

def _min_triangle_area(points):
    """Exact minimum triangle area."""
    return np.min(_compute_areas(points))

def _soft_min_objective_and_grad(x, beta):
    """
    Computes the soft-min objective and its gradient.
    Objective: (1/beta) * log(sum(exp(-beta * areas)))
    This is a smooth approximation of -min(areas) (which we want to minimize, i.e. maximize min(areas)).
    Wait, let's stick to the convention:
    We want to maximize min(areas).
    Soft min M = -1/beta * log(sum(exp(-beta * areas)))
    We want to maximize M, so we minimize f = -M = 1/beta * log(sum(exp(-beta * areas))).
    
    Gradient of f:
    f = (1/beta) * log(S), where S = sum(exp(-beta * a_t))
    df/da_t = (1/beta) * (1/S) * exp(-beta * a_t) * (-beta) = -exp(-beta * a_t) / S
    Let w_t = exp(-beta * a_t) / S. Then df/da_t = -w_t.
    Note that sum(w_t) = 1.
    
    a_t = 0.5 * |cross_t|
    If cross_t > 0, da_t/dx = 0.5 * d(cross_t)/dx
    If cross_t < 0, da_t/dx = -0.5 * d(cross_t)/dx
    So da_t/dx = 0.5 * sign(cross_t) * d(cross_t)/dx
    
    df/dx = sum_t (df/da_t * da_t/dx) = sum_t (-w_t * 0.5 * sign(cross_t) * d(cross_t)/dx)
    df/dx = -0.5 * sum_t (w_t * sign(cross_t) * d(cross_t)/dx)
    
    cross_t = (p_j - p_i) x (p_k - p_i)
    cross_t = (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
    
    d(cross_t)/dx_i:
    d/dx_i [ (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i) ]
    = -1 * (y_k - y_i) - (y_j - y_i) * (-1)
    = -y_k + y_i + y_j - y_i = y_j - y_k
    
    d(cross_t)/dy_i:
    = (x_j - x_i) * (-1) - 1 * (x_k - x_i)
    = -x_j + x_i - x_k + x_i = 2*x_i - x_j - x_k
    
    d(cross_t)/dx_j:
    = 1 * (y_k - y_i) - 0 = y_k - y_i
    
    d(cross_t)/dy_j:
    = 0 - 1 * (x_k - x_i) = x_i - x_k
    
    d(cross_t)/dx_k:
    = 0 - (y_j - y_i) * 1 = y_i - y_j
    
    d(cross_t)/dy_k:
    = (x_j - x_i) * 1 - 0 = x_j - x_i
    
    We can accumulate gradients for x and y.
    Then convert to barycentric gradient.
    x = u * V1[0] + v * V2[0]
    y = u * V1[1] + v * V2[1]
    V1 = (1, 0), V2 = (0.5, sqrt(3)/2)
    x = u + 0.5v
    y = sqrt(3)/2 v
    
    dx/du = 1, dx/dv = 0.5
    dy/du = 0, dy/dv = sqrt(3)/2
    
    d/dx = (du/dx)(d/du) + (dv/dx)(d/dv)
    d/dy = (du/dy)(d/du) + (dv/dy)(d/dv)
    
    Inverse Jacobian:
    [dx, dy] = [1, 0.5; 0, sqrt(3)/2] [du, dv]
    J = [[1, 0.5], [0, sqrt(3)/2]]
    J^-1 = [[1, -1/sqrt(3)], [0, 2/sqrt(3)]]
    
    So d/du = 1 * d/dx + 0 * d/dy = d/dx
    d/dv = -1/sqrt(3) * d/dx + 2/sqrt(3) * d/dy
    
    Let's verify:
    d(x)/du = 1. d(du)/du = 1.
    d(x)/dv = 0.5. d(du)/dv = 0.
    d(y)/du = 0. d(dv)/du = 0.
    d(y)/dv = sqrt(3)/2. d(dv)/dv = 1.
    
    If we have grad_x and grad_y (gradients wrt x and y coordinates),
    grad_u = grad_x
    grad_v = -1/sqrt(3) * grad_x + 2/sqrt(3) * grad_y
    
    Let's implement this.
    """
    bary = x.reshape(-1, 2)
    bary = _clamp_bary(bary)
    points = _bary_to_cart(bary)
    areas = _compute_areas(points)
    
    # Stable log-sum-exp
    x_log = -beta * areas
    x_max = np.max(x_log)
    if x_max == -np.inf:
        return 0.0, np.zeros_like(x)
    log_sum_exp = x_max + np.log(np.sum(np.exp(x_log - x_max)))
    val = (1.0 / beta) * log_sum_exp
    
    # Weights w_t = exp(-beta * a_t) / S
    # log(w_t) = -beta * a_t - log(S)
    # log(S) = log_sum_exp + beta * max_a ? No.
    # S = sum(exp(x_log)) = exp(x_max) * sum(exp(x_log - x_max))
    # log(S) = x_max + log(sum(exp(x_log - x_max))) = log_sum_exp
    # So w_t = exp(x_log_t - log_sum_exp)
    w = np.exp(x_log - log_sum_exp)
    
    # Gradient accumulation
    # df/dx = -0.5 * sum_t (w_t * sign(cross_t) * d(cross_t)/dx)
    pi = points[_IDX_TRIPLES[:, 0]]
    pj = points[_IDX_TRIPLES[:, 1]]
    pk = points[_IDX_TRIPLES[:, 2]]
    
    d1 = pj - pi
    d2 = pk - pi
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    signs = np.sign(cross)
    # Handle zero cross? If cross is 0, area is 0, gradient is undefined or 0.
    # w_t will be large for small areas. If area is 0, exp(-beta*0)=1.
    # sign(0) is 0 in numpy. This might be okay, effectively ignoring degenerate triangles in gradient?
    # Or maybe we should use a small epsilon?
    # Actually if area is 0, we want to move points apart.
    # If cross is 0, the points are collinear.
    # Let's just use sign.
    
    # d(cross)/dx_i = y_j - y_k
    # d(cross)/dy_i = 2*x_i - x_j - x_k
    # d(cross)/dx_j = y_k - y_i
    # d(cross)/dy_j = x_i - x_k
    # d(cross)/dx_k = y_i - y_j
    # d(cross)/dy_k = x_j - x_i
    
    coeff = -0.5 * w * signs
    
    grad_x = np.zeros((N, 2))
    grad_y = np.zeros((N, 2))
    
    # Use np.add.at for accumulation
    # i index
    idx_i = _IDX_TRIPLES[:, 0]
    idx_j = _IDX_TRIPLES[:, 1]
    idx_k = _IDX_TRIPLES[:, 2]
    
    # d(cross)/dx_i
    dc_dx_i = pj[:, 1] - pk[:, 1]
    np.add.at(grad_x, idx_i, coeff[:, None] * dc_dx_i[:, None])
    
    # d(cross)/dy_i
    dc_dy_i = 2 * pi[:, 0] - pj[:, 0] - pk[:, 0]
    np.add.at(grad_y, idx_i, coeff[:, None] * dc_dy_i[:, None])
    
    # d(cross)/dx_j
    dc_dx_j = pk[:, 1] - pi[:, 1]
    np.add.at(grad_x, idx_j, coeff[:, None] * dc_dx_j[:, None])
    
    # d(cross)/dy_j
    dc_dy_j = pi[:, 0] - pk[:, 0]
    np.add.at(grad_y, idx_j, coeff[:, None] * dc_dy_j[:, None])
    
    # d(cross)/dx_k
    dc_dx_k = pi[:, 1] - pj[:, 1]
    np.add.at(grad_x, idx_k, coeff[:, None] * dc_dx_k[:, None])
    
    # d(cross)/dy_k
    dc_dy_k = pj[:, 0] - pi[:, 0]
    np.add.at(grad_y, idx_k, coeff[:, None] * dc_dy_k[:, None])
    
    # Convert to barycentric gradient
    # grad_u = grad_x
    # grad_v = -1/sqrt(3) * grad_x + 2/sqrt(3) * grad_y
    grad_bary = np.zeros_like(bary)
    grad_bary[:, 0] = grad_x[:, 0]
    grad_bary[:, 1] = -1.0/np.sqrt(3.0) * grad_x[:, 0] + 2.0/np.sqrt(3.0) * grad_y[:, 0]
    
    # Wait, grad_x and grad_y are (N, 2) arrays where column 0 is x-grad and column 1 is y-grad?
    # No, I defined grad_x as (N,2) but only filled column 0?
    # Let's re-clarify.
    # grad_x[i] should be d/dx_i.
    # grad_y[i] should be d/dy_i.
    # In my code:
    # np.add.at(grad_x, idx_i, ...) -> adds to grad_x[idx_i, 0]?
    # No, grad_x is (N,2). idx_i is (M,).
    # np.add.at(grad_x, idx_i, values) where values is (M, 1) adds to column 0.
    # I need separate arrays for dx and dy gradients.
    
    # Let's redo this part in the function body to be safe.
    
    return val, grad_bary.flatten()

def _polish_soft_min(bary_init, beta, time_limit):
    """Polish using SLSQP with soft-min objective and gradient."""
    def obj_and_grad(x):
        val, grad = _soft_min_objective_and_grad(x, beta)
        return val, grad
    
    # Constraints
    constraints = []
    bounds = []
    for i in range(N):
        u_idx = 2 * i
        v_idx = 2 * i + 1
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        
        A_u = np.zeros(2 * N)
        A_u[u_idx] = 1.0
        constraints.append({'type': 'ineq', 'fun': lambda x, A=A_u: A @ x})
        
        A_v = np.zeros(2 * N)
        A_v[v_idx] = 1.0
        constraints.append({'type': 'ineq', 'fun': lambda x, A=A_v: A @ x})
        
        A_sum = np.zeros(2 * N)
        A_sum[u_idx] = -1.0
        A_sum[v_idx] = -1.0
        constraints.append({'type': 'ineq', 'fun': lambda x, A=A_sum: 1.0 + A @ x})
        
    x0 = bary_init.flatten()
    
    try:
        result = minimize(
            obj_and_grad,
            x0,
            method='SLSQP',
            jac=True,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-8}
        )
        
        bary_opt = _clamp_bary(result.x.reshape(-1, 2))
        score_opt = _min_triangle_area(_bary_to_cart(bary_opt))
        
        if time.time() > time_limit:
            return bary_init, _min_triangle_area(_bary_to_cart(bary_init))
            
        return bary_opt, score_opt
        
    except Exception:
        return bary_init, _min_triangle_area(_bary_to_cart(bary_init))

def _simulated_annealing(bary_init, time_limit, rng):
    """
    Run simulated annealing on barycentric coordinates.
    """
    bary = bary_init.copy()
    bary = _clamp_bary(bary)
    
    T_start = 0.05
    T_end = 0.001
    n_iter = 3000
    
    best_bary = bary.copy()
    best_score = _min_triangle_area(_bary_to_cart(bary))
    current_score = best_score
    
    for i in range(n_iter):
        if time.time() > time_limit:
            break
            
        alpha = i / n_iter
        T = T_start * (T_end / T_start) ** alpha
        
        step_size = 0.05 * (1.0 - 0.5 * alpha) 
        perturbation = rng.normal(0, step_size, size=bary.shape)
        new_bary = bary + perturbation
        new_bary = _clamp_bary(new_bary)
        
        new_points = _bary_to_cart(new_bary)
        new_min = _min_triangle_area(new_points)
        
        if new_min > current_score:
            bary = new_bary
            current_score = new_min
            if current_score > best_score:
                best_score = current_score
                best_bary = bary.copy()
        else:
            delta = new_min - current_score
            if rng.uniform() < np.exp(delta / T):
                bary = new_bary
                current_score = new_min
                
    return best_bary, best_score

def _polish_exact(bary_init, time_limit):
    """Polish using Nelder-Mead on the exact objective."""
    def objective(x):
        bary = x.reshape(-1, 2)
        bary = _clamp_bary(bary)
        points = _bary_to_cart(bary)
        return -_min_triangle_area(points)
        
    x0 = bary_init.flatten()
    
    try:
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        
        bary_opt = _clamp_bary(result.x.reshape(-1, 2))
        score_opt = -result.fun
        
        if time.time() > time_limit:
            return bary_init, _min_triangle_area(_bary_to_cart(bary_init))
            
        return bary_opt, score_opt
        
    except Exception:
        return bary_init, _min_triangle_area(_bary_to_cart(bary_init))

def _generate_initializations():
    """Generate diverse initial configurations."""
    inits = []
    
    bary_struct = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.5, 0.5],
        [0.0, 0.5],
        [1.0/3, 1.0/3],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.25, 0.25],
        [0.5, 0.25],
    ])
    inits.append(bary_struct)
    
    rng = np.random.RandomState(42)
    for _ in range(5):
        bary_r = np.zeros((N, 2))
        for _ in range(N):
            while True:
                u = rng.uniform(0, 1)
                v = rng.uniform(0, 1)
                if u + v <= 1:
                    bary_r[_] = [u, v]
                    break
        inits.append(bary_r)
        
    for scale in [0.05, 0.1, 0.2]:
        bary_pert = bary_struct + rng.normal(0, scale, size=bary_struct.shape)
        bary_pert = _clamp_bary(bary_pert)
        inits.append(bary_pert)
        
    return inits

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def time_left():
        return deadline - time.time()
        
    best_bary = None
    best_score = -np.inf
    
    inits = _generate_initializations()
    
    # Phase 1: Simulated Annealing
    rng_sa = np.random.RandomState(0)
    candidates = []
    
    for i, bary0 in enumerate(inits):
        if time_left() < 10.0:
            break
            
        bary_sa, score_sa = _simulated_annealing(bary0, deadline - 5.0, rng_sa)
        
        if score_sa > best_score:
            best_score = score_sa
            best_bary = bary_sa.copy()
            
        candidates.append(bary_sa)
        
    # Phase 2: Soft-Min Polish (Gradient Descent)
    # Try a few different betas
    for beta in [10, 50, 100]:
        if time_left() < 15.0:
            break
        for bary_c in candidates:
            if time_left() < 5.0:
                break
            bary_pol, score_pol = _polish_soft_min(bary_c, beta, deadline - 5.0)
            if score_pol > best_score:
                best_score = score_pol
                best_bary = bary_pol.copy()
                
    # Phase 3: Exact Polish (Nelder-Mead)
    if best_bary is not None:
        if time_left() > 10.0:
            bary_pol, score_pol = _polish_exact(best_bary, deadline - 2.0)
            if score_pol > best_score:
                best_score = score_pol
                best_bary = bary_pol.copy()
                
    # Phase 4: If time remains, restart from best with perturbations
    if time_left() > 20.0 and best_bary is not None:
        for seed in range(5):
            if time_left() < 10.0:
                break
            rng_run = np.random.RandomState(1000 + seed)
            bary_pert = best_bary + rng_run.normal(0, 0.05, size=best_bary.shape)
            bary_pert = _clamp_bary(bary_pert)
            
            bary_sa, score_sa = _simulated_annealing(bary_pert, deadline - 5.0, rng_run)
            if score_sa > best_score:
                best_score = score_sa
                best_bary = bary_sa.copy()
                
            if time_left() > 10.0:
                bary_pol, score_pol = _polish_soft_min(bary_sa, 50, deadline - 5.0)
                if score_pol > best_score:
                    best_score = score_pol
                    best_bary = bary_pol.copy()
                    
                if time_left() > 5.0:
                    bary_pol2, score_pol2 = _polish_exact(bary_pol, deadline - 2.0)
                    if score_pol2 > best_score:
                        best_score = score_pol2
                        best_bary = bary_pol2.copy()

    if best_bary is None:
        best_bary = inits[0]
        
    best_points = _bary_to_cart(best_bary)
    best_points = _clip_to_triangle(best_points)
    
    return best_points

def _clip_to_triangle(points):
    """Clip points to be inside the equilateral triangle."""
    points = points.copy()
    x = points[:, 0]
    y = points[:, 1]
    
    y = np.clip(y, 0, H)
    x_min = y / np.sqrt(3)
    x_max = 1 - y / np.sqrt(3)
    x = np.clip(x, x_min, x_max)
    x = np.clip(x, 0, 1)
    
    points[:, 0] = x
    points[:, 1] = y
    return points

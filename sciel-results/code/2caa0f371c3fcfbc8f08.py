"""
Heilbronn triangles: maximize the smallest triangle area formed by 11 points in an equilateral triangle.
Combined strategy:
1. Soft-min (log-sum-exp) surrogate objective for smooth gradients.
2. L-BFGS-B optimizer with annealing of the soft-min sharpness parameter beta.
3. FPS-based diverse initializations.
4. Final polishing with raw min objective.
"""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

# Precompute indices for all triples (i, j, k) with i < j < k
# Number of triples is C(11, 3) = 165
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_IDX_I = _IDX_TRIPLES[:, 0]
_IDX_J = _IDX_TRIPLES[:, 1]
_IDX_K = _IDX_TRIPLES[:, 2]


def _min_triangle_area_vectorized(points):
    """
    Compute the smallest area among all triangles formed by triples of points, vectorized.
    points: (N, 2) array
    Returns: float min area
    """
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    
    # Cross product magnitude for 2D vectors
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    
    if len(areas) == 0:
        return 0.0
        
    return np.min(areas)

def _all_triangle_areas(points):
    """
    Compute all triangle areas, vectorized.
    """
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas

def _param_to_points(params):
    """
    Map parameters in [0, 1]^{2N} to points inside the triangle.
    Barycentric coordinates:
    lambda_0 = 1 - s_i
    lambda_1 = s_i * (1 - t_i)
    lambda_2 = s_i * t_i
    Point = lambda_1 * V1 + lambda_2 * V2 (since V0 is origin)
    """
    s = params[0::2]
    t = params[1::2]
    
    # Clip to avoid numerical issues at boundaries
    s = np.clip(s, 1e-12, 1.0 - 1e-12)
    t = np.clip(t, 1e-12, 1.0 - 1e-12)
    
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
    
    # Inverse barycentric transformation
    # y = lam1 * V1[1] + lam2 * V2[1] = 0 + lam2 * (sqrt(3)/2) => lam2 = 2*y/sqrt(3)
    lam2 = 2.0 * y / np.sqrt(3)
    # x = lam1 * V1[0] + lam2 * V2[0] = lam1 * 1 + lam2 * 0.5 => lam1 = x - 0.5*lam2
    lam1 = x - 0.5 * lam2
    
    # s = lam1 + lam2, t = lam2 / s
    s = lam1 + lam2
    t = np.zeros_like(s)
    mask = s > 1e-12
    t[mask] = lam2[mask] / s[mask]
    
    s = np.clip(s, 1e-12, 1.0 - 1e-12)
    t = np.clip(t, 1e-12, 1.0 - 1e-12)
    
    params = np.zeros(2 * N)
    params[0::2] = s
    params[1::2] = t
    return params


def _soft_min_objective_and_grad(params, beta):
    """
    Objective: negative soft-min of triangle areas.
    Gradient: computed via chain rule.
    
    soft_min(A) = -1/beta * log(sum(exp(-beta * A_i)))
    We want to maximize min(A), so we minimize -soft_min(A).
    f(params) = -soft_min(A) = 1/beta * log(sum(exp(-beta * A_i)))
    
    grad f = 1/beta * 1/S * sum(-beta * exp(-beta * A_i) * grad A_i)
           = - 1/S * sum(w_i * grad A_i)
    where w_i = exp(-beta * A_i), S = sum(w_i)
    """
    points = _param_to_points(params)
    areas = _all_triangle_areas(points)
    
    # Numerical stability for log-sum-exp
    min_a = np.min(areas)
    # Shift areas to prevent underflow/overflow in exp
    # We compute sum(exp(-beta * (A_i - min_a)))
    shifted = -beta * (areas - min_a)
    # Max shifted is 0 (for min area), so exp is <= 1. Safe.
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    # Objective value
    # -soft_min = min_a - (1/beta) * log(sum(exp(-beta(A_i - min_a))))
    # Wait, the standard soft min is -1/beta log(sum exp(-beta A_i)).
    # Let's stick to the definition:
    # soft_min = min_a - (1/beta) * log( sum( exp(-beta (A_i - min_a)) ) )
    # We minimize -soft_min.
    obj = -(min_a - np.log(sum_exp) / beta)
    
    # Gradient
    # Weights w_i proportional to exp(-beta (A_i - min_a))
    weights = exp_shifted / sum_exp
    
    # Compute grad A_i w.r.t points
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    
    # A_i = 0.5 * |cross_i|
    # cross_i = (pj_x - pi_x)*(pk_y - pi_y) - (pj_y - pi_y)*(pk_x - pi_x)
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    
    # sign of cross to handle absolute value
    # d|cross|/dparams = sign(cross) * dcross/dparams
    sign_cross = np.sign(cross)
    # Avoid division by zero or undefined gradient when cross=0
    sign_cross[np.abs(cross) < 1e-15] = 0.0
    
    # Gradient of cross w.r.t points
    # d(cross)/d(pi) = (-(pk_y - pi_y) - (pj_y - pi_y), -(pj_x - pi_x) + (pk_x - pi_x)) ?
    # Let's re-derive carefully.
    # cross = (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
    # d/dx_i: -(y_k - y_i) - (y_j - y_i) = y_i - y_k - y_j + y_i = 2y_i - y_j - y_k
    # d/dy_i: (x_j - x_i) - (x_k - x_i) = x_j - x_i - x_k + x_i = x_j - x_k
    # d/dx_j: (y_k - y_i)
    # d/dy_j: -(x_k - x_i)
    # d/dx_k: -(y_j - y_i)
    # d/dy_k: (x_j - x_i)
    
    # Actually, simpler:
    # d(cross)/d(p_i) = p_j - p_k (rotated 90 deg? No.)
    # Let's use the explicit components derived in inspiration, they were correct.
    # d_cross_pi_x = -(pk[:, 1] - pi[:, 1]) + (pj[:, 1] - pi[:, 1])
    # d_cross_pi_y = -(pj[:, 0] - pi[:, 0]) + (pk[:, 0] - pi[:, 0])
    # d_cross_pj_x = pk[:, 1] - pi[:, 1]
    # d_cross_pj_y = -(pk[:, 0] - pi[:, 0])
    # d_cross_pk_x = -(pj[:, 1] - pi[:, 1])
    # d_cross_pk_y = pj[:, 0] - pi[:, 0]
    
    d_cross_pi = np.column_stack([
        -(pk[:, 1] - pi[:, 1]) + (pj[:, 1] - pi[:, 1]),
        -(pj[:, 0] - pi[:, 0]) + (pk[:, 0] - pi[:, 0])
    ])
    d_cross_pj = np.column_stack([
        pk[:, 1] - pi[:, 1],
        -(pk[:, 0] - pi[:, 0])
    ])
    d_cross_pk = np.column_stack([
        -(pj[:, 1] - pi[:, 1]),
        pj[:, 0] - pi[:, 0]
    ])
    
    # Gradient of Area A_i = 0.5 * |cross_i|
    # grad_A_i = 0.5 * sign(cross_i) * d_cross_i
    
    # Accumulate gradient for each point
    grad_points = np.zeros((N, 2))
    
    # Weighted sum of gradients
    w = weights * sign_cross * 0.5
    
    np.add.at(grad_points, _IDX_I, w[:, np.newaxis] * d_cross_pi)
    np.add.at(grad_points, _IDX_J, w[:, np.newaxis] * d_cross_pj)
    np.add.at(grad_points, _IDX_K, w[:, np.newaxis] * d_cross_pk)
    
    # Chain rule: grad_params = J^T * grad_points
    # params[2i] = s_i, params[2i+1] = t_i
    # points[i] = [lam1 * V1_x + lam2 * V2_x, lam1 * V1_y + lam2 * V2_y]
    # lam1 = s(1-t), lam2 = st
    # dx/ds = (1-t)V1_x + tV2_x
    # dx/dt = -sV1_x + sV2_x
    # dy/ds = (1-t)V1_y + tV2_y
    # dy/dt = -sV1_y + sV2_y
    
    s = params[0::2]
    t = params[1::2]
    
    dx_ds = (1 - t) * V1[0] + t * V2[0]
    dx_dt = -s * V1[0] + s * V2[0]
    dy_ds = (1 - t) * V1[1] + t * V2[1]
    dy_dt = -s * V1[1] + s * V2[1]
    
    grad_params = np.zeros(2 * N)
    grad_params[0::2] = grad_points[:, 0] * dx_ds + grad_points[:, 1] * dy_ds
    grad_params[1::2] = grad_points[:, 0] * dx_dt + grad_points[:, 1] * dy_dt
    
    # Objective is -soft_min, so we minimize it.
    # The gradient of the objective function to be minimized is -grad(soft_min).
    # My derivation above:
    # obj = -soft_min
    # grad obj = - grad soft_min
    # grad soft_min = sum(w_i * grad A_i) ... wait.
    # soft_min = -1/beta log(sum exp(-beta A_i))
    # d(soft_min)/dA_i = -1/beta * 1/S * (-beta exp(-beta A_i)) = w_i
    # So grad soft_min = sum(w_i * grad A_i)
    # Therefore grad obj = - sum(w_i * grad A_i)
    # My grad_points accumulated sum(w_i * grad A_i) (with sign handled).
    # So I need to negate grad_params.
    
    return obj, -grad_params

def _raw_objective(params):
    """Objective for final polishing: negative min area."""
    points = _param_to_points(params)
    return -_min_triangle_area_vectorized(points)

def _generate_fps_initialization(rng, n_points=N, n_candidates=300):
    """
    Generate an initial configuration using Greedy Farthest Point Sampling.
    """
    points = np.zeros((n_points, 2))
    # Start with centroid
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
        
        # Vectorized distance calculation
        # candidates: (M, 2), points[:k]: (k, 2)
        diffs = candidates[:, np.newaxis, :] - points[np.newaxis, :k, :]
        dists = np.linalg.norm(diffs, axis=2)
        min_dists = np.min(dists, axis=1)
        
        best_idx = np.argmax(min_dists)
        points[k] = candidates[best_idx]
        
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    def time_left():
        return deadline - time.time()

    best_params = None
    best_score = -np.inf
    
    rng = np.random.RandomState(42)
    
    # Generate initial configurations
    initial_params = []
    
    # 1. FPS initializations
    for i in range(5):
        if time_left() < 30.0: break
        pts = _generate_fps_initialization(rng)
        params = _points_to_param(pts)
        initial_params.append(params)
        
    # 2. Random initializations
    for i in range(3):
        if time_left() < 30.0: break
        s = rng.uniform(0.1, 0.9, size=N)
        t = rng.uniform(0.1, 0.9, size=N)
        params = np.zeros(2 * N)
        params[0::2] = s
        params[1::2] = t
        initial_params.append(params)
        
    # 3. Differential Evolution Global Search
    if time_left() > 40.0:
        try:
            de_result = differential_evolution(
                _raw_objective,
                bounds=[(0.0, 1.0)] * (2 * N),
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-8,
                polish=False
            )
            de_params = de_result.x
            initial_params.append(de_params)
        except Exception:
            pass

    bounds = [(0.0, 1.0)] * (2 * N)
    
    # Annealing schedule for beta
    # Start low, end high
    betas = [5.0, 10.0, 20.0, 50.0, 100.0]
    
    current_best_params = initial_params[0].copy()
    
    for start_p in initial_params:
        if time_left() < 20.0: break
        
        current_params = start_p.copy()
        
        # Annealing loop
        for beta in betas:
            if time_left() < 10.0: break
            
            # Define objective and gradient for current beta
            def obj_grad(params):
                return _soft_min_objective_and_grad(params, beta)
            
            try:
                res = minimize(
                    obj_grad, current_params,
                    method='L-BFGS-B',
                    jac=True,
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                current_params = res.x
            except Exception as e:
                pass
        
        # Final polishing with raw min objective
        if time_left() < 5.0: break
        
        try:
            res = minimize(
                _raw_objective, current_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            current_params = res.x
        except Exception:
            pass
        
        score = -_raw_objective(current_params)
        if score > best_score:
            best_score = score
            best_params = current_params.copy()
            
    if best_params is None:
        # Fallback
        return np.array([
            [0.5, 0.0], [0.0, 0.0], [1.0, 0.0],
            [0.25, np.sqrt(3)/4], [0.75, np.sqrt(3)/4], [0.5, np.sqrt(3)/2],
            [0.5, np.sqrt(3)/8], [0.25, np.sqrt(3)/8], [0.75, np.sqrt(3)/8],
            [0.5, 3*np.sqrt(3)/8], [0.5, np.sqrt(3)/4]
        ])
        
    best_points = _param_to_points(best_params)
    
    return best_points

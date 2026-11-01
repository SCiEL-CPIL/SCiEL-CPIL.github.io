import numpy as np
import scipy.spatial
import scipy.optimize
import time
from itertools import combinations

N = 14

def _get_triangle_indices(n):
    indices = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=np.int32)

_TRI_INDICES = _get_triangle_indices(N)
_I = _TRI_INDICES[:, 0]
_J = _TRI_INDICES[:, 1]
_K = _TRI_INDICES[:, 2]

def _min_triangle_area(points):
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross).min()

def _hull_area(points):
    try:
        hull = scipy.spatial.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    try:
        if len(points) < 3:
            return 0.0
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        mta = _min_triangle_area(points)
        return mta / ha
    except Exception:
        return 0.0

def _normalize(points):
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return points
        # Scale so hull area = 1; ratio min_area/hull_area is scale-invariant
        return points / np.sqrt(ha)
    except Exception:
        return points

def _regular_polygon(n, radius=1.0, rotation=0.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _compute_areas(points):
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _neg_soft_min_score_and_grad(points_flat, beta=10.0):
    """
    Objective: - (soft_min(areas) / hull_area)
    soft_min(areas) = -1/beta * log(sum(exp(-beta * areas)))
    """
    points = points_flat.reshape(-1, 2)
    
    areas = _compute_areas(points)
    
    # Avoid overflow/underflow in exp
    min_area = np.min(areas)
    if min_area < 1e-15:
        return 0.0, np.zeros_like(points_flat)
        
    # Shift by min area to keep exp stable
    shifted = areas - min_area
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    
    if sum_exp == 0:
        return 0.0, np.zeros_like(points_flat)
        
    weights = exp_vals / sum_exp
    
    # soft_min value
    soft_min_val = min_area - (1.0 / beta) * np.log(sum_exp)
    
    ha = _hull_area(points)
    if ha < 1e-12:
        return 0.0, np.zeros_like(points_flat)
        
    score = soft_min_val / ha
    
    # Gradient of soft_min with respect to points
    # d(soft_min)/d(area_i) = -weights[i]
    # d(area_i)/d(points) = 0.5 * sign(cross_i) * [grad_cross_i]
    
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    
    signs = np.sign(cross)
    
    # Gradients of cross product
    # cross = (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
    # d/dx_i = -(y_k - y_i) + (y_j - y_i) = -y_k + y_i + y_j - y_i = y_j - y_k
    # d/dy_i = -(x_k - x_i) + (x_j - x_i) = -x_k + x_i + x_j - x_i = x_j - x_k
    # d/dx_j = (y_k - y_i)
    # d/dy_j = -(x_k - x_i)
    # d/dx_k = -(y_j - y_i)
    # d/dy_k = (x_j - x_i)
    
    d_cross_di_x = v1[:, 1] - v2[:, 1] # y_j - y_k
    d_cross_di_y = v1[:, 0] - v2[:, 0] # x_j - x_k
    d_cross_dj_x = v2[:, 1]           # y_k - y_i
    d_cross_dj_y = -v2[:, 0]          # -(x_k - x_i)
    d_cross_dk_x = -v1[:, 1]          # -(y_j - y_i)
    d_cross_dk_y = v1[:, 0]           # x_j - x_i
    
    # Weighted gradients
    # d(soft_min)/d(point_i) = sum_k weights_k * d(area_k)/d(point_i)
    # d(area_k)/d(point_i) = 0.5 * sign(cross_k) * d(cross_k)/d(point_i)
    
    w = weights * signs * 0.5
    
    grad_points = np.zeros_like(points)
    
    dA_di = w[:, None] * np.column_stack([d_cross_di_x, d_cross_di_y])
    dA_dj = w[:, None] * np.column_stack([d_cross_dj_x, d_cross_dj_y])
    dA_dk = w[:, None] * np.column_stack([d_cross_dk_x, d_cross_dk_y])
    
    np.add.at(grad_points, _I, dA_di)
    np.add.at(grad_points, _J, dA_dj)
    np.add.at(grad_points, _K, dA_dk)
    
    # Gradient of score = soft_min / ha
    # We ignore the gradient of ha for simplicity (it's complex and non-smooth).
    # This is a common approximation. The scale invariance means the direction
    # is what matters most.
    grad_score = grad_points / ha
    
    return -score, -grad_score.flatten()

def _neg_sum_k_smallest_score_and_grad(points_flat, K=5):
    """
    Objective: - (sum of K smallest areas / hull_area)
    This is non-smooth but effective for focusing on the worst triangles.
    """
    points = points_flat.reshape(-1, 2)
    areas = _compute_areas(points)
    
    # Get indices of K smallest
    # argsort is O(N log N), but N is small (364)
    idx = np.argsort(areas)[:K]
    min_areas_sum = np.sum(areas[idx])
    
    ha = _hull_area(points)
    if ha < 1e-12:
        return 0.0, np.zeros_like(points_flat)
        
    score = min_areas_sum / ha
    
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    signs = np.sign(cross)
    
    d_cross_di_x = v1[:, 1] - v2[:, 1]
    d_cross_di_y = v1[:, 0] - v2[:, 0]
    d_cross_dj_x = v2[:, 1]
    d_cross_dj_y = -v2[:, 0]
    d_cross_dk_x = -v1[:, 1]
    d_cross_dk_y = v1[:, 0]
    
    # We only sum gradients for the K smallest
    # Create a weight vector that is 1 for the K smallest, 0 otherwise
    w = np.zeros(len(areas))
    w[idx] = 1.0
    
    # d(area)/d(point) = 0.5 * sign * d(cross)/d(point)
    # Sum over active set
    w_active = w * signs * 0.5
    
    grad_points = np.zeros_like(points)
    
    dA_di = w_active[:, None] * np.column_stack([d_cross_di_x, d_cross_di_y])
    dA_dj = w_active[:, None] * np.column_stack([d_cross_dj_x, d_cross_dj_y])
    dA_dk = w_active[:, None] * np.column_stack([d_cross_dk_x, d_cross_dk_y])
    
    np.add.at(grad_points, _I, dA_di)
    np.add.at(grad_points, _J, dA_dj)
    np.add.at(grad_points, _K, dA_dk)
    
    grad_score = grad_points / ha
    
    return -score, -grad_score.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_score = 0.0
    best_points = None
    rng = np.random.RandomState(42)

    def _update_best(points):
        nonlocal best_score, best_points
        pts = _normalize(points)
        sc = _score(pts)
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()
        return sc

    def _optimize_phase(init_points, objective_func, maxiter=200, method='L-BFGS-B'):
        if time.time() > deadline:
            return
        init_norm = _normalize(init_points)
        x0 = init_norm.flatten()
        
        try:
            res = scipy.optimize.minimize(
                objective_func,
                x0,
                jac=True,
                method=method,
                options={'maxiter': maxiter, 'gtol': 1e-8, 'ftol': 1e-12}
            )
            x = res.x
            _update_best(x.reshape(-1, 2))
            return x.reshape(-1, 2)
        except Exception:
            return init_norm

    # 1. Initialization Strategies
    inits = []
    
    # Regular polygon
    inits.append(_regular_polygon(N, 1.0))
    
    # Jittered regular polygon
    for i in range(5):
        noise = rng.uniform(-0.05, 0.05, (N, 2))
        inits.append(_regular_polygon(N, 1.0) + noise)
        
    # Two-ring configurations (quick grid search for best params)
    # We'll use a few good candidates based on symmetry
    for n1, n2 in [(7, 7), (6, 8), (5, 9)]:
        for r2 in [0.5, 0.6, 0.7]:
            for rot in [0.0, np.pi/n1, np.pi/n2]:
                pts1 = _regular_polygon(n1, 1.0, 0.0)
                pts2 = _regular_polygon(n2, r2, rot)
                inits.append(np.vstack([pts1, pts2]))
                
    # 2. Optimization Pipeline
    # For each initialization, run a multi-stage optimization
    for init in inits:
        if time.time() > deadline - 10:
            break
            
        current_points = init.copy()
        
        # Stage 1: Soft min with low beta (global smoothing)
        def _obj_beta10(x):
            return _neg_soft_min_score_and_grad(x, beta=10.0)
        
        current_points = _optimize_phase(current_points, _obj_beta10, maxiter=100)
        
        if time.time() > deadline - 10:
            continue
            
        # Stage 2: Soft min with higher beta (sharper focus)
        def _obj_beta50(x):
            return _neg_soft_min_score_and_grad(x, beta=50.0)
        
        current_points = _optimize_phase(current_points, _obj_beta50, maxiter=100)
        
        if time.time() > deadline - 10:
            continue
            
        # Stage 3: Sum of K smallest (active set)
        def _obj_k5(x):
            return _neg_sum_k_smallest_score_and_grad(x, K=5)
        
        current_points = _optimize_phase(current_points, _obj_k5, maxiter=50)
        
        if time.time() > deadline - 10:
            continue
            
        # Stage 4: Sum of K=1 (exact min, non-smooth)
        def _obj_k1(x):
            return _neg_sum_k_smallest_score_and_grad(x, K=1)
        
        current_points = _optimize_phase(current_points, _obj_k1, maxiter=50)
        
        # Final Polish with Nelder-Mead on raw score
        if time.time() < deadline - 2:
            try:
                def _neg_raw_score(x):
                    return -_score(x.reshape(-1, 2))
                
                res = scipy.optimize.minimize(
                    _neg_raw_score,
                    current_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                _update_best(res.x.reshape(-1, 2))
            except Exception:
                pass

    if best_points is None:
        best_points = _normalize(_regular_polygon(N, 1.0))
    else:
        best_points = _normalize(best_points)
        
    return best_points

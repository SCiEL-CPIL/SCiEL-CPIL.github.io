import numpy as np
import scipy.spatial
import scipy.optimize
import time
from itertools import combinations

N = 14

# Pre-compute triangle indices for speed
def _get_triangle_indices(n):
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
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
        return points / np.sqrt(ha)
    except Exception:
        return points

def _regular_polygon(n, radius=1.0, rotation=0.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _generate_two_ring_config(n1, n2, r1, r2, rot1, rot2):
    pts1 = _regular_polygon(n1, r1, rot1)
    pts2 = _regular_polygon(n2, r2, rot2)
    return np.vstack([pts1, pts2])

def _neg_smooth_score_and_grad(points_flat, beta=100.0):
    """
    Computes negative smooth minimum triangle area divided by hull area,
    and its gradient.
    """
    points = points_flat.reshape(-1, 2)
    
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    max_area = np.max(areas)
    if max_area < 1e-15:
        return 0.0, np.zeros_like(points_flat)

    # Shift for numerical stability in exp
    shifted = areas - max_area
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    if sum_exp == 0:
        return 0.0, np.zeros_like(points_flat)
    
    weights = exp_vals / sum_exp
    
    # Smooth min approximation: -1/beta * log(sum(exp(-beta * areas)))
    # = -1/beta * ( -beta * max_area + log(sum(exp(-beta * (areas - max_area)))) )
    # = max_area - 1/beta * log(sum_exp)
    # Wait, the standard softmin is min(x) ~ -1/beta log(sum exp(-beta x)).
    # So score = softmin(areas) / ha.
    # We want to maximize score, so minimize -score.
    
    log_sum = np.log(sum_exp)
    smooth_min_val = max_area - log_sum / beta
    
    ha = _hull_area(points)
    if ha < 1e-12:
        return 0.0, np.zeros_like(points_flat)
    
    score = smooth_min_val / ha
    
    # Gradient of score w.r.t points
    # d(score)/d(points) = (1/ha) * d(softmin)/d(points)
    # d(softmin)/d(x_i) = - (1/beta) * (1/sum_exp) * sum_j (d(exp(-beta A_j))/dx_i)
    # = - (1/beta) * (1/sum_exp) * sum_j (-beta * exp(-beta A_j) * dA_j/dx_i)
    # = (1/sum_exp) * sum_j (exp(-beta A_j) * dA_j/dx_i)
    # = sum_j (weights_j * dA_j/dx_i)
    
    # We need dA_j/d(points).
    # A_j = 0.5 * |cross_j|. 
    # Note: The sign of cross matters for the gradient of |cross|.
    # d|c|/dx = sign(c) * dc/dx.
    
    signs = np.sign(cross)
    
    # Derivatives of cross product c = v1_x v2_y - v1_y v2_x
    # v1 = P_j - P_i, v2 = P_k - P_i
    # c = (P_j - P_i)_x (P_k - P_i)_y - (P_j - P_i)_y (P_k - P_i)_x
    
    # d c / d P_i = - (v1_y, -v1_x) - (v2_y, -v2_x) ? 
    # Let's derive carefully.
    # c = (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
    # d c / d x_i = -(y_k - y_i) - (-(y_j - y_i)) = -y_k + y_i + y_j - y_i = y_j - y_k
    # d c / d y_i = -(x_k - x_i) - (-(x_j - x_i)) = -x_k + x_i + x_j - x_i = x_j - x_k
    
    # d c / d x_j = (y_k - y_i)
    # d c / d y_j = -(x_k - x_i)
    
    # d c / d x_k = -(y_j - y_i)
    # d c / d y_k = (x_j - x_i)
    
    # So:
    # grad_c_i = [y_j - y_k, x_j - x_k]
    # grad_c_j = [y_k - y_i, -(x_k - x_i)]
    # grad_c_k = [-(y_j - y_i), x_j - x_i]
    
    # In vector form using v1, v2:
    # v1 = P_j - P_i, v2 = P_k - P_i
    # y_j - y_k = v1_y - v2_y
    # x_j - x_k = v1_x - v2_x
    
    # d c / d P_i = [-(v1_y - v2_y), -(v1_x - v2_x)] = [v2_y - v1_y, v2_x - v1_x]
    # d c / d P_j = [v2_y, -v2_x]
    # d c / d P_k = [-v1_y, v1_x]
    
    # Let's verify with the code from inspiration 1 which seemed to work:
    # d_cross_di_x = -(v1[:, 1] + v2[:, 1]) ? 
    # My derivation: v2_y - v1_y.
    # Inspiration: -(v1_y + v2_y). 
    # Let's re-derive d c / d P_i.
    # c = (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
    # d/dx_i: -(y_k - y_i) - [-(y_j - y_i)] = -y_k + y_i + y_j - y_i = y_j - y_k.
    # v1_y = y_j - y_i. v2_y = y_k - y_i.
    # y_j - y_k = (v1_y + y_i) - (v2_y + y_i) = v1_y - v2_y.
    # So d c / d x_i = v1_y - v2_y.
    # d c / d y_i: -(x_k - x_i) - [-(x_j - x_i)] = -x_k + x_i + x_j - x_i = x_j - x_k.
    # x_j - x_k = v1_x - v2_x.
    # So grad_c_i = [v1_y - v2_y, v1_x - v2_x].
    
    # Let's check inspiration 1 again.
    # d_cross_di_x = -(v1[:, 1] + v2[:, 1])
    # This looks like it might be wrong or I am misinterpreting the indices.
    # Let's trust my derivation.
    
    d_c_di_x = v1[:, 1] - v2[:, 1]
    d_c_di_y = v1[:, 0] - v2[:, 0]
    
    d_c_dj_x = v2[:, 1]
    d_c_dj_y = -v2[:, 0]
    
    d_c_dk_x = -v1[:, 1]
    d_c_dk_y = v1[:, 0]
    
    # d A / d P = 0.5 * sign(cross) * d cross / d P
    # We are summing weights_j * d A_j / d P
    # w_j = weights_j * 0.5 * signs_j
    
    w = weights * signs * 0.5
    
    grad_points = np.zeros_like(points)
    
    dA_di = w[:, None] * np.column_stack([d_c_di_x, d_c_di_y])
    dA_dj = w[:, None] * np.column_stack([d_c_dj_x, d_c_dj_y])
    dA_dk = w[:, None] * np.column_stack([d_c_dk_x, d_c_dk_y])
    
    np.add.at(grad_points, _I, dA_di)
    np.add.at(grad_points, _J, dA_dj)
    np.add.at(grad_points, _K, dA_dk)
    
    # Gradient of score = gradient of softmin / ha
    grad_score = grad_points / ha
    
    # We minimize -score
    return -score, -grad_score.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
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

    def _try_optimize_smooth(init_points, max_total_iter=2000):
        """
        Multi-stage L-BFGS-B optimization using smooth min approximation.
        """
        if time.time() > deadline:
            return
        init_norm = _normalize(init_points)
        x = init_norm.flatten()
        
        # Stages with increasing beta (sharpness)
        # Start with low beta to find the basin, increase to sharpen the min
        betas = [20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
        
        for beta in betas:
            if time.time() > deadline - 2:
                break
            
            def _obj_and_grad(x_flat, beta_val=beta):
                return _neg_smooth_score_and_grad(x_flat, beta_val)
            
            try:
                # Use L-BFGS-B with analytic gradient
                res = scipy.optimize.minimize(
                    _obj_and_grad,
                    x,
                    jac=True,
                    method='L-BFGS-B',
                    options={
                        'maxiter': max_total_iter // len(betas), 
                        'gtol': 1e-8, 
                        'ftol': 1e-12,
                        'maxfun': 10000
                    }
                )
                x = res.x
                _update_best(x.reshape(-1, 2))
            except Exception:
                pass
        
        # Final polish with exact score using Nelder-Mead on the result
        if time.time() < deadline - 1:
            try:
                def _neg_exact_score(x_flat):
                    return -_score(x_flat.reshape(-1, 2))
                
                res = scipy.optimize.minimize(
                    _neg_exact_score,
                    x,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                _update_best(res.x.reshape(-1, 2))
            except Exception:
                pass

    # 1. Explore two-ring configurations extensively
    # These provide good initial basins for the 28D optimization
    splits = [(7, 7), (6, 8), (8, 6), (5, 9), (9, 5), (4, 10), (10, 4)]
    
    for n1, n2 in splits:
        if time.time() > deadline - 20: # Leave time for refinement
            break
        
        def _two_ring_score(params):
            r_ratio = params[0]
            rot1 = params[1]
            rot2 = params[2]
            
            r1 = 1.0
            r2 = r_ratio * r1
            
            if r2 <= 0.01:
                return -1e6
                
            pts = _generate_two_ring_config(n1, n2, r1, r2, rot1, rot2)
            sc = _score(_normalize(pts))
            return -sc

        # Try a few starting points for the 3-parameter optimization
        for r_init in [0.3, 0.5, 0.7, 0.9]:
            for rot_init in [0.0, np.pi/n1, np.pi/n2]:
                if time.time() > deadline - 10:
                    continue
                x0 = np.array([r_init, 0.0, rot_init])
                try:
                    res = scipy.optimize.minimize(
                        _two_ring_score,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10}
                    )
                    r_ratio, rot1, rot2 = res.x
                    pts = _generate_two_ring_config(n1, n2, 1.0, r_ratio, rot1, rot2)
                    sc = _update_best(pts)
                    
                    # If this configuration is competitive, refine it with full 28D optimization
                    if sc > best_score * 0.9 and time.time() < deadline - 15:
                        _try_optimize_smooth(pts, max_total_iter=1500)
                        
                except Exception:
                    pass

    # 2. Explore regular polygon and perturbations
    # Regular 14-gon
    reg = _regular_polygon(N, 1.0)
    _update_best(reg)
    if time.time() < deadline - 15:
        _try_optimize_smooth(reg, max_total_iter=1500)
        
    # Perturb regular polygon
    for i in range(5):
        if time.time() > deadline - 10:
            break
        noise = rng.uniform(-0.1, 0.1, (N, 2))
        init = _regular_polygon(N, 1.0) + noise
        _try_optimize_smooth(init, max_total_iter=1000)

    # 3. Random restarts
    for i in range(5):
        if time.time() > deadline - 5:
            break
        # Generate points on a disk
        r = np.sqrt(rng.uniform(0, 1, N))
        theta = rng.uniform(0, 2 * np.pi, N)
        init = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        _try_optimize_smooth(init, max_total_iter=500)

    # Final check
    if best_points is None:
        best_points = _normalize(_regular_polygon(N, 1.0))
    else:
        best_points = _normalize(best_points)
        
    return best_points

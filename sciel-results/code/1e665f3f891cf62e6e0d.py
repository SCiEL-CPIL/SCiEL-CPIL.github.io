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
        # Scale so that hull area = 1
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

def _generate_three_ring_config(n1, n2, n3, r1, r2, r3, rot1, rot2, rot3):
    pts1 = _regular_polygon(n1, r1, rot1)
    pts2 = _regular_polygon(n2, r2, rot2)
    pts3 = _regular_polygon(n3, r3, rot3)
    return np.vstack([pts1, pts2, pts3])

def _neg_smooth_score_and_grad(points_flat, beta=100.0):
    """
    Computes the negative smooth minimum triangle area divided by hull area,
    along with its gradient.
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

    shifted = areas - max_area
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    if sum_exp == 0:
        return 0.0, np.zeros_like(points_flat)
    weights = exp_vals / sum_exp
    
    log_sum = -beta * max_area + np.log(sum_exp)
    smooth_min = -log_sum / beta
    
    ha = _hull_area(points)
    if ha < 1e-12:
        return 0.0, np.zeros_like(points_flat)
    
    score = smooth_min / ha
    
    signs = np.sign(cross)
    
    # Gradients of cross product w.r.t. coordinates of i, j, k
    # cross = (xj-xi)(yk-yi) - (yj-yi)(xk-xi)
    # d/dxi = -(yk-yi) - (yj-yi)*0 ... wait.
    # Let A = (xj-xi), B = (yk-yi), C = (yj-yi), D = (xk-xi)
    # cross = A*B - C*D
    # d/dxi: A'=-1, B'=0, C'=0, D'=-1 => -B - (-1)*D = -B + D = -(yk-yi) + (xk-xi) ?? No.
    # Let's re-derive carefully.
    # cross = (xj - xi)(yk - yi) - (yj - yi)(xk - xi)
    # d/dxi: -1*(yk - yi) - (yj - yi)*(-1) = -(yk - yi) + (yj - yi) = yj - yk
    # d/dxj: 1*(yk - yi) - 0 = yk - yi
    # d/dxk: 0 - (yj - yi)*1 = -(yj - yi) = yi - yj
    
    # d/dyi: (xj - xi)*(-1) - (-1)*(xk - xi) = -(xj - xi) + (xk - xi) = xk - xj
    # d/dyj: 0 - 1*(xk - xi) = -(xk - xi) = xi - xk
    # d/dyk: (xj - xi)*1 - 0 = xj - xi

    d_cross_di_x = (yj - yi) - (yk - yi) # yj - yk
    d_cross_di_y = -(xj - xi) + (xk - xi) # xk - xj
    
    # Wait, the variables in the code are:
    # v1 = pj - pi => v1[0]=xj-xi, v1[1]=yj-yi
    # v2 = pk - pi => v2[0]=xk-xi, v2[1]=yk-yi
    # cross = v1[0]*v2[1] - v1[1]*v2[0]
    
    # d/dpi_x = -v2[1] - (-v2[0])? No.
    # d/dpi_x (v1[0]*v2[1]) = -v2[1]
    # d/dpi_x (v1[1]*v2[0]) = -v2[0]
    # d/dpi_x cross = -v2[1] - (-v2[0]) = v2[0] - v2[1] ?? No.
    # d/dpi_x (A*B - C*D) where A=xj-xi, B=yk-yi, C=yj-yi, D=xk-xi
    # dA/dxi = -1, dB/dxi=0, dC/dxi=0, dD/dxi=-1
    # d/dxi = -1*B - C*(-1) = -B + C = -(yk-yi) + (yj-yi) = yj - yk
    
    # d/dpi_y: dA/dyi=0, dB/dyi=-1, dC/dyi=1, dD/dyi=0
    # d/dyi = A*(-1) - 1*D = -A - D = -(xj-xi) - (xk-xi) = 2xi - xj - xk ??
    # Let's check: A*B - C*D.
    # d/dyi (A*B) = A*(-1) = -A
    # d/dyi (C*D) = 1*D = D
    # d/dyi cross = -A - D = -(xj-xi) - (xk-xi) = 2xi - xj - xk.
    
    # d/dpj_x: dA/dxj=1, others 0. => B = yk-yi
    # d/dpj_y: dC/dyj=1. => -D = -(xk-xi)
    
    # d/dpk_x: dD/dxk=1. => -C = -(yj-yi)
    # d/dpk_y: dB/dyk=1. => A = xj-xi

    # Vectorized:
    d_cross_di_x = (pj[:, 1] - pi[:, 1]) - (pk[:, 1] - pi[:, 1]) # yj - yk
    d_cross_di_y = -(pj[:, 0] - pi[:, 0]) - (pk[:, 0] - pi[:, 0]) # -(xj-xi) - (xk-xi) = 2xi - xj - xk
    
    d_cross_dj_x = pk[:, 1] - pi[:, 1] # yk - yi
    d_cross_dj_y = -(pk[:, 0] - pi[:, 0]) # -(xk - xi)
    
    d_cross_dk_x = -(pj[:, 1] - pi[:, 1]) # -(yj - yi)
    d_cross_dk_y = pj[:, 0] - pi[:, 0] # xj - xi
    
    w = weights * signs * 0.5
    
    grad_points = np.zeros_like(points)
    
    dA_di = w[:, None] * np.column_stack([d_cross_di_x, d_cross_di_y])
    dA_dj = w[:, None] * np.column_stack([d_cross_dj_x, d_cross_dj_y])
    dA_dk = w[:, None] * np.column_stack([d_cross_dk_x, d_cross_dk_y])
    
    np.add.at(grad_points, _I, dA_di)
    np.add.at(grad_points, _J, dA_dj)
    np.add.at(grad_points, _K, dA_dk)
    
    grad_score = grad_points / ha
    
    return -score, -grad_score.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
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

    def _refine(init_points, beta_stages=[20, 50, 200, 500], maxiter_per_stage=200):
        """
        Refines points using multi-stage L-BFGS-B with smooth minimum,
        followed by a short Nelder-Mead polish.
        """
        if time.time() > deadline:
            return init_points
        
        x = _normalize(init_points).flatten()
        
        for beta in beta_stages:
            if time.time() > deadline - 5:
                break
            
            def _obj_and_grad(x_flat, beta_val=beta):
                return _neg_smooth_score_and_grad(x_flat, beta_val)
            
            try:
                res = scipy.optimize.minimize(
                    _obj_and_grad,
                    x,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': maxiter_per_stage, 'gtol': 1e-8, 'ftol': 1e-12}
                )
                x = res.x
                _update_best(x.reshape(-1, 2))
            except Exception:
                pass
        
        # Final polish with Nelder-Mead on the exact score
        if time.time() < deadline - 2:
            try:
                def _neg_exact_score(x_flat):
                    return -_score(x_flat.reshape(-1, 2))
                
                res = scipy.optimize.minimize(
                    _neg_exact_score,
                    x,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                _update_best(res.x.reshape(-1, 2))
                x = res.x
            except Exception:
                pass
                
        return x.reshape(-1, 2)

    # 1. Two-Ring Configurations
    # Splits: (7,7) is most symmetric. (6,8), (8,6) are next.
    splits = [(7, 7), (6, 8), (8, 6), (5, 9), (9, 5)]
    
    for n1, n2 in splits:
        if time.time() > deadline - 20:
            break
            
        # Optimize the two-ring parameters first to get a good start for the 28D optimizer
        # Params: r_ratio, rot1, rot2
        # We fix r1=1.
        
        def _two_ring_score(params):
            r_ratio = params[0]
            rot1 = params[1]
            rot2 = params[2]
            r1 = 1.0
            r2 = r_ratio * r1
            if r2 <= 0.01:
                return 1e6
            pts = _generate_two_ring_config(n1, n2, r1, r2, rot1, rot2)
            return -_score(_normalize(pts))

        best_two_ring_pts = None
        best_two_ring_sc = -np.inf
        
        # Try a few starting points for the 3D optimization
        for r_init in [0.5, 0.6, 0.7]:
            for rot_init in [0.0, np.pi/n1]:
                if time.time() > deadline - 10:
                    break
                x0 = np.array([r_init, 0.0, rot_init])
                try:
                    res = scipy.optimize.minimize(
                        _two_ring_score,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
                    )
                    r_ratio, rot1, rot2 = res.x
                    pts = _generate_two_ring_config(n1, n2, 1.0, r_ratio, rot1, rot2)
                    sc = _score(_normalize(pts))
                    if sc > best_two_ring_sc:
                        best_two_ring_sc = sc
                        best_two_ring_pts = pts
                except Exception:
                    pass
        
        if best_two_ring_pts is not None:
            _refine(best_two_ring_pts)

    # 2. Three-Ring Configurations
    # Splits: (5, 4, 5), (6, 2, 6), (7, 1, 6)
    three_ring_splits = [(5, 4, 5), (6, 2, 6)]
    
    for n1, n2, n3 in three_ring_splits:
        if time.time() > deadline - 15:
            break
            
        def _three_ring_score(params):
            r1, r2, r3 = params[0], params[1], params[2]
            rot1, rot2, rot3 = params[3], params[4], params[5]
            if min(r1, r2, r3) <= 0.01:
                return 1e6
            pts = _generate_three_ring_config(n1, n2, n3, r1, r2, r3, rot1, rot2, rot3)
            return -_score(_normalize(pts))
        
        best_three_ring_pts = None
        best_three_ring_sc = -np.inf
        
        # Initial guess: radii 1.0, 0.6, 0.3
        # Rotations: 0, pi/n2, 0
        x0 = np.array([1.0, 0.6, 0.3, 0.0, np.pi/n2, 0.0])
        try:
            res = scipy.optimize.minimize(
                _three_ring_score,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            pts = _generate_three_ring_config(n1, n2, n3, res.x[0], res.x[1], res.x[2], res.x[3], res.x[4], res.x[5])
            sc = _score(_normalize(pts))
            if sc > best_three_ring_sc:
                best_three_ring_sc = sc
                best_three_ring_pts = pts
        except Exception:
            pass
        
        if best_three_ring_pts is not None:
            _refine(best_three_ring_pts)

    # 3. Regular Polygon Perturbations
    # Start with regular 14-gon and add small noise, then refine
    reg = _regular_polygon(N, 1.0)
    for i in range(5):
        if time.time() > deadline - 10:
            break
        noise = rng.uniform(-0.05, 0.05, (N, 2))
        init = reg + noise
        _refine(init)

    # 4. Random Restarts
    for i in range(3):
        if time.time() > deadline - 5:
            break
        r = np.sqrt(rng.uniform(0, 1, N))
        theta = rng.uniform(0, 2 * np.pi, N)
        init = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        _refine(init)

    # Final check
    if best_points is None:
        best_points = _normalize(_regular_polygon(N, 1.0))
    else:
        best_points = _normalize(best_points)
        
    return best_points

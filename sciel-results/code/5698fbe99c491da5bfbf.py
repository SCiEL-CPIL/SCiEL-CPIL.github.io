import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

N = 13

_TRIPLE_IDX = np.array([
    (i, j, k)
    for i in range(N)
    for j in range(i + 1, N)
    for k in range(j + 1, N)
], dtype=np.int32)

def _triangle_areas_vectorized(points):
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _hull_area(points):
    try:
        if len(points) < 3:
            return 0.0
        if np.allclose(points, points[0]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = np.min(areas)
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area

def _normalize(points):
    """Center and normalize points."""
    centroid = np.mean(points, axis=0)
    pts = points - centroid
    # Normalize so hull area = 1 for consistent comparison
    hull_area = _hull_area(pts)
    if hull_area > 1e-12:
        pts = pts / np.sqrt(hull_area)
    return pts

def _objective_full(x_flat):
    """Full objective: minimize negative score."""
    points = x_flat.reshape(N, 2)
    score = _score(points)
    if score <= 0:
        return 1e10
    return -score

def _smooth_objective(x_flat, p=30):
    """Smooth surrogate using power mean."""
    points = x_flat.reshape(N, 2)
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    if np.allclose(points, 0):
        return 1e10
    
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10
    
    areas = _triangle_areas_vectorized(points)
    areas = np.maximum(areas, 1e-15)
    
    # Soft min
    inv_areas = 1.0 / areas
    soft_min = (np.mean(inv_areas ** p)) ** (-1.0 / p)
    
    score = soft_min / hull_area
    if score <= 0:
        return 1e10
    return -score

def _logsumexp_smooth_obj(x_flat, beta=50.0):
    """Log-sum-exp smooth surrogate: -log(mean(exp(-beta/area)))."""
    points = x_flat.reshape(N, 2)
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    if np.allclose(points, 0):
        return 1e10
    
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10
    
    areas = _triangle_areas_vectorized(points)
    areas = np.maximum(areas, 1e-15)
    
    # -log(mean(exp(-beta/area))) approximates min(1/area) scaled
    # We want to maximize min(1/area) / hull_area
    # So minimize -log(mean(exp(-beta/area))) + log(hull_area)
    neg_beta_over_areas = -beta / areas
    # log-sum-exp trick for numerical stability
    m = np.max(neg_beta_over_areas)
    lse = m + np.log(np.mean(np.exp(neg_beta_over_areas - m)))
    # We want to minimize lse + log(hull_area)
    # Since lse ≈ -beta * min(1/area) = -beta/max(areas), minimizing lse maximizes max(areas)
    # Actually: lse = log(mean(exp(-beta/area))) ≈ -beta/min(1/area) = -beta/max(areas)
    # So -lse ≈ beta/max(areas), and we want to maximize max(areas)/hull_area
    # Minimize: lse + log(hull_area)
    return lse + np.log(hull_area)

def _logsumexp_smooth_grad(x_flat, beta=50.0):
    """Gradient of log-sum-exp smooth objective."""
    points = x_flat.reshape(N, 2)
    centroid = np.mean(points, axis=0)
    pts = points - centroid
    
    if np.allclose(pts, 0):
        return 1e10, np.zeros_like(x_flat)
    
    hull_area = _hull_area(pts)
    if hull_area < 1e-12:
        return 1e10, np.zeros_like(x_flat)
    
    areas = _triangle_areas_vectorized(pts)
    areas = np.maximum(areas, 1e-15)
    
    neg_beta_over_areas = -beta / areas
    m = np.max(neg_beta_over_areas)
    exp_vals = np.exp(neg_beta_over_areas - m)
    weights = exp_vals / np.sum(exp_vals)
    
    # d/d(pts) of area for triangle (i,j,k):
    # area = 0.5 * |cross(pj-pi, pk-pi)|
    # For non-degenerate, d(area)/d(pi) = -0.5*cross_sign*(pk-pi)_perp
    # d(area)/d(pj) = 0.5*cross_sign*(pk-pi)_perp  
    # d(area)/d(pk) = -0.5*cross_sign*(pj-pi)_perp
    # where perp(x) = (-x[1], x[0])
    
    grad_areas = np.zeros_like(pts)
    for idx in range(len(_TRIPLE_IDX)):
        i, j, k = _TRIPLE_IDX[idx]
        pi = pts[i]
        pj = pts[j]
        pk = pts[k]
        v1 = pj - pi
        v2 = pk - pi
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        sign = 1.0 if cross >= 0 else -1.0
        
        # d(area)/d(pi) = -0.5 * sign * perp(v2)
        # d(area)/d(pj) =  0.5 * sign * perp(v2)
        # d(area)/d(pk) = -0.5 * sign * perp(v1)
        perp_v2 = np.array([-v2[1], v2[0]])
        perp_v1 = np.array([-v1[1], v1[0]])
        
        grad_areas[i] += -0.5 * sign * perp_v2
        grad_areas[j] +=  0.5 * sign * perp_v2
        grad_areas[k] += -0.5 * sign * perp_v1
    
    # d(lse)/d(pts) = sum(weights * d(-beta/area)/d(pts))
    # d(-beta/area)/d(pts) = beta/area^2 * d(area)/d(pts)
    d_neg_beta_d_area = beta / (areas ** 2)
    grad_lse = np.zeros_like(pts)
    for idx in range(len(_TRIPLE_IDX)):
        w = weights[idx]
        d_neg = d_neg_beta_d_area[idx]
        grad_lse += w * d_neg * grad_areas[_TRIPLE_IDX[idx, 0]] * 0 + w * d_neg * grad_areas[idx]
    
    # Actually, let me redo this more carefully
    # lse = log(sum(exp(-beta/area_t)))
    # d(lse)/d(pts) = sum_t [exp(-beta/area_t) / sum(exp(-beta/area_s))] * d(-beta/area_t)/d(pts)
    # d(-beta/area_t)/d(pts) = beta/area_t^2 * d(area_t)/d(pts)
    
    grad_obj = np.zeros_like(pts)
    for idx in range(len(_TRIPLE_IDX)):
        i, j, k = _TRIPLE_IDX[idx]
        w = weights[idx]
        d_neg = beta / (areas[idx] ** 2)
        
        pi = pts[i]
        pj = pts[j]
        pk = pts[k]
        v1 = pj - pi
        v2 = pk - pi
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        sign = 1.0 if cross >= 0 else -1.0
        
        perp_v2 = np.array([-v2[1], v2[0]])
        perp_v1 = np.array([-v1[1], v1[0]])
        
        d_area_di = -0.5 * sign * perp_v2
        d_area_dj =  0.5 * sign * perp_v2
        d_area_dk = -0.5 * sign * perp_v1
        
        grad_obj[i] += w * d_neg * d_area_di
        grad_obj[j] += w * d_neg * d_area_dj
        grad_obj[k] += w * d_neg * d_area_dk
    
    # Gradient of log(hull_area) w.r.t. pts
    # Approximate: hull area is piecewise linear, use subgradient
    # For simplicity, approximate d(log(hull_area))/d(pts) ≈ pts / (N * hull_area) * 2
    # This is a rough approximation
    grad_log_hull = 2.0 * pts / (N * hull_area)
    
    grad_total = grad_obj + grad_log_hull
    
    # Account for centering: pts = x - mean(x)
    # d(pts)/d(x) = I - (1/N) * ones
    # So grad_x = grad_pts - mean(grad_pts)
    grad_flat = (grad_total - np.mean(grad_total, axis=0)).flatten()
    
    val = m + np.log(np.mean(exp_vals)) + np.log(hull_area)
    return val, grad_flat

def _make_ring_config(n_outer, n_inner, r_inner, angle_offset):
    """Create a two-ring configuration."""
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    angles_inner = np.linspace(angle_offset, 2 * np.pi + angle_offset, n_inner, endpoint=False)
    
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner_pts = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    
    return np.vstack([outer_pts, inner_pts])

def _optimize_from_start(x0, deadline, best_score_ref):
    """Run multi-phase optimization from a starting point."""
    best_x = x0.copy()
    best_s = _score(x0.reshape(N, 2))
    
    # Phase 1: Smooth optimization
    if time.time() < deadline - 10:
        try:
            res = minimize(
                lambda x: _smooth_objective(x, p=30),
                x0,
                method='Powell',
                options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-7}
            )
            if _score(res.x.reshape(N, 2)) > best_s:
                best_s = _score(res.x.reshape(N, 2))
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Hard objective polish
    if time.time() < deadline - 5:
        try:
            res_nm = minimize(
                _objective_full,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
            )
            s_nm = _score(res_nm.x.reshape(N, 2))
            if s_nm > best_s:
                best_s = s_nm
                best_x = res_nm.x.copy()
        except Exception:
            pass
    
    return best_x, best_s

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_points = None
    best_score = 0.0
    
    def _update_best(points):
        nonlocal best_points, best_score
        if points is None or len(points) != N:
            return
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
    
    # Generate diverse initial configurations
    inits = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]).flatten())
    
    # 2. Two-ring configurations with various parameters
    for n_out in [6, 7, 8]:
        n_in = N - n_out
        for r_in in [0.3, 0.4, 0.5, 0.6, 0.7]:
            for offset in [0, np.pi / n_out, np.pi / (2 * n_out)]:
                pts = _make_ring_config(n_out, n_in, r_in, offset)
                inits.append(pts.flatten())
    
    # 3. Three-ring configurations
    for n1, n2, n3 in [(5, 4, 4), (6, 5, 2), (7, 3, 3), (4, 4, 5)]:
        for r2 in [0.5, 0.6, 0.7]:
            for r3 in [0.2, 0.3, 0.4]:
                a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
                a2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + np.pi / n1
                a3 = np.linspace(0, 2 * np.pi, n3, endpoint=False) + np.pi / n2
                pts = np.vstack([
                    np.column_stack([np.cos(a1), np.sin(a1)]),
                    np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)]),
                    np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)])
                ])
                inits.append(pts.flatten())
    
    # 4. Random starts
    rng = np.random.default_rng(12345)
    for _ in range(15):
        r = np.sqrt(rng.random(N))
        theta = rng.random(N) * 2 * np.pi
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        inits.append(pts.flatten())
    
    # 5. Perturbed regular polygon (slight radial/angular noise)
    for noise_scale in [0.02, 0.05, 0.1]:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = 1.0 + noise_scale * (rng.random(N) - 0.5)
        angles_perturbed = angles + noise_scale * (rng.random(N) - 0.5)
        pts = np.column_stack([radii * np.cos(angles_perturbed), radii * np.sin(angles_perturbed)])
        inits.append(pts.flatten())
    
    # 6. Grid-like configurations (approximating optimal packing)
    # 13 points: try 4x4 grid minus corners, or similar
    for grid_type in ['4x4_minus', '3x5']:
        if grid_type == '4x4_minus':
            # 4x4 grid, remove 3 corners
            xs = np.array([0, 1, 2, 3])
            ys = np.array([0, 1, 2, 3])
            gx, gy = np.meshgrid(xs, ys)
            pts_grid = np.column_stack([gx.ravel(), gy.ravel()])
            # Remove 3 corners
            pts_grid = pts_grid[3:]  # remove first 3
            if len(pts_grid) == N:
                # Center and normalize
                pts_grid = pts_grid - np.mean(pts_grid, axis=0)
                inits.append(pts_grid.flatten())
        elif grid_type == '3x5':
            xs = np.array([0, 1, 2])
            ys = np.array([0, 1, 2, 3, 4])
            gx, gy = np.meshgrid(xs, ys)
            pts_grid = np.column_stack([gx.ravel(), gy.ravel()])
            if len(pts_grid) == N:
                pts_grid = pts_grid - np.mean(pts_grid, axis=0)
                inits.append(pts_grid.flatten())
    
    # Shuffle inits to diversify
    rng2 = np.random.default_rng(999)
    indices = rng2.permutation(len(inits))
    inits = [inits[i] for i in indices]
    
    # Optimize from each start
    for i, x0 in enumerate(inits):
        if time.time() > deadline - 8:
            break
        
        t_start = time.time()
        x_opt, s_opt = _optimize_from_start(x0, deadline, best_score)
        _update_best(x_opt.reshape(N, 2))
        
        elapsed = time.time() - t_start
        if i % 5 == 0:
            print(f"Init {i}: score={best_score:.6f}, elapsed={elapsed:.1f}s", flush=True)
    
    # Final high-precision polish on best
    if best_points is not None and time.time() < deadline - 3:
        try:
            # First, L-BFGS-B polish with tighter beta for sharper min
            def lbfgs_final_func(x):
                val, grad = _logsumexp_smooth_grad(x, beta=100.0)
                return val, grad
            
            res_lbfgs_final = minimize(
                lbfgs_final_func,
                best_points.flatten(),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            _update_best(res_lbfgs_final.x.reshape(N, 2))
        except Exception:
            pass
        
        # Then Nelder-Mead with very tight tolerance
        if best_points is not None and time.time() < deadline - 1:
            try:
                res_final = minimize(
                    _objective_full,
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
                )
                _update_best(res_final.x.reshape(N, 2))
            except Exception:
                pass
    
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points

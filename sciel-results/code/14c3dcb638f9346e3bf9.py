import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

N = 13
_DIM = 2 * N

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

def _smooth_objective(x_flat, p=100):
    """Smooth surrogate using log-sum-exp soft-min of areas, divided by hull area."""
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
    
    # Log-sum-exp based soft-min: -log(sum(exp(-p * areas))) / p
    neg_p_areas = -p * areas
    max_neg = np.max(neg_p_areas)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_p_areas - max_neg)))
    soft_min = -log_sum / p
    
    score = soft_min / hull_area
    if score <= 0:
        return 1e10
    return -score

def _smooth_objective_grad(x_flat, p=100):
    """Smooth objective with numerical gradient for L-BFGS-B."""
    points = x_flat.reshape(N, 2)
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    if np.allclose(points, 0):
        return 1e10, np.zeros(_DIM)
    
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10, np.zeros(_DIM)
    
    areas = _triangle_areas_vectorized(points)
    areas = np.maximum(areas, 1e-15)
    
    # Log-sum-exp based soft-min
    neg_p_areas = -p * areas
    max_neg = np.max(neg_p_areas)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_p_areas - max_neg)))
    soft_min = -log_sum / p
    
    score = soft_min / hull_area
    if score <= 0:
        return 1e10, np.zeros(_DIM)
    
    # Numerical gradient
    eps = 1e-7
    grad = np.zeros(_DIM)
    for i in range(_DIM):
        x_plus = x_flat.copy()
        x_plus[i] += eps
        val_plus = _smooth_objective(x_plus, p)
        x_minus = x_flat.copy()
        x_minus[i] -= eps
        val_minus = _smooth_objective(x_minus, p)
        grad[i] = (val_plus - val_minus) / (2 * eps)
    
    return -score, -grad

def _objective_full(x_flat):
    """Full objective: minimize negative score."""
    points = x_flat.reshape(N, 2)
    points = points - np.mean(points, axis=0)
    score = _score(points)
    if score <= 0:
        return 1e10
    return -score

def _make_ring_config(n_outer, n_inner, r_inner, angle_offset):
    """Create a two-ring configuration."""
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    angles_inner = np.linspace(angle_offset, 2 * np.pi + angle_offset, n_inner, endpoint=False)
    
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner_pts = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    
    return np.vstack([outer_pts, inner_pts])

def _gradient_descent_polish(points, deadline, max_iter=2000, lr=0.01):
    """Gradient descent on the smooth objective with adaptive learning rate."""
    best_x = points.flatten().copy()
    best_s = _score(points)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        val, grad = _smooth_objective_grad(best_x, p=100)
        if not np.isfinite(val) or np.any(np.isnan(grad)):
            break
        
        # Adaptive step size
        norm_grad = np.linalg.norm(grad)
        if norm_grad < 1e-10:
            break
        
        step = lr / (1.0 + norm_grad)
        trial_x = best_x - step * grad
        
        trial_pts = trial_x.reshape(N, 2)
        trial_s = _score(trial_pts)
        
        if trial_s > best_s + 1e-12:
            best_x = trial_x
            best_s = trial_s
            lr = min(lr * 1.1, 0.1)  # Slightly increase if improving
        else:
            lr *= 0.8  # Decrease if not improving
    
    return best_x.reshape(N, 2), best_s

def _local_polish(points, deadline):
    """Multi-phase local optimization from a starting point."""
    best_x = points.flatten().copy()
    best_s = _score(points)
    
    # Phase 1: L-BFGS-B on smooth objective with gradient
    if time.time() < deadline - 15:
        try:
            res = minimize(
                _smooth_objective_grad,
                best_x,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            s = _score(res.x.reshape(N, 2))
            if s > best_s:
                best_s = s
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead on full objective
    if time.time() < deadline - 8:
        try:
            res_nm = minimize(
                _objective_full,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            s_nm = _score(res_nm.x.reshape(N, 2))
            if s_nm > best_s:
                best_s = s_nm
                best_x = res_nm.x.copy()
        except Exception:
            pass
    
    # Phase 3: Gradient descent polish
    if time.time() < deadline - 3:
        try:
            gd_pts, gd_s = _gradient_descent_polish(best_x.reshape(N, 2), deadline - 1)
            if gd_s > best_s:
                best_s = gd_s
                best_x = gd_pts.flatten().copy()
        except Exception:
            pass
    
    return best_x.reshape(N, 2), best_s

def _run_differential_evolution(deadline):
    """Run differential evolution for global search."""
    # Bounds: points in [-2, 2] x [-2, 2]
    bounds = [(-2.0, 2.0)] * _DIM
    
    def de_obj(x):
        points = x.reshape(N, 2)
        points = points - np.mean(points, axis=0)
        score = _score(points)
        return -score if score > 0 else 1e10
    
    try:
        result = differential_evolution(
            de_obj,
            bounds,
            maxiter=200,
            popsize=15,
            tol=1e-8,
            mutation=(0.3, 1.0),
            recombination=0.8,
            seed=42,
            polish=False,
            update_interval=10,
            callback=lambda xk, convergence: time.time() > deadline - 30
        )
        return result.x.reshape(N, 2), _score(result.x.reshape(N, 2))
    except Exception:
        return None, 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
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
    
    # Phase 1: Differential Evolution for global search (30s)
    if time.time() < deadline - 100:
        de_pts, de_s = _run_differential_evolution(deadline)
        if de_pts is not None and de_s > best_score:
            _update_best(de_pts)
            print(f"DE: score={best_score:.6f}", flush=True)
    
    # Phase 2: Generate diverse initial configurations
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
    for _ in range(20):
        r = np.sqrt(rng.random(N))
        theta = rng.random(N) * 2 * np.pi
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        inits.append(pts.flatten())
    
    # 5. Perturbed regular polygon
    for noise_scale in [0.02, 0.05, 0.1, 0.15]:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = 1.0 + noise_scale * (rng.random(N) - 0.5)
        angles_perturbed = angles + noise_scale * (rng.random(N) - 0.5)
        pts = np.column_stack([radii * np.cos(angles_perturbed), radii * np.sin(angles_perturbed)])
        inits.append(pts.flatten())
    
    # 6. Grid-like configurations
    for grid_type in ['4x4_minus', '3x5']:
        if grid_type == '4x4_minus':
            xs = np.array([0, 1, 2, 3])
            ys = np.array([0, 1, 2, 3])
            gx, gy = np.meshgrid(xs, ys)
            pts_grid = np.column_stack([gx.ravel(), gy.ravel()])
            pts_grid = pts_grid[3:]
            if len(pts_grid) == N:
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
    
    # Phase 3: Local optimization from each start
    for i, x0 in enumerate(inits):
        if time.time() > deadline - 20:
            break
        
        x_opt, s_opt = _local_polish(x0.reshape(N, 2), deadline - 10)
        _update_best(x_opt)
        
        if i % 5 == 0:
            print(f"Init {i}: score={best_score:.6f}, elapsed={time.time()-start_time:.1f}s", flush=True)
    
    # Phase 4: Final high-precision polish on best
    if best_points is not None and time.time() < deadline - 5:
        try:
            res_final = minimize(
                _objective_full,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
            )
            _update_best(res_final.x.reshape(N, 2))
        except Exception:
            pass
        
        # Gradient descent final polish
        if best_points is not None and time.time() < deadline - 2:
            try:
                gd_pts, gd_s = _gradient_descent_polish(best_points, deadline - 1)
                _update_best(gd_pts)
            except Exception:
                pass
    
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points

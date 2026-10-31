import numpy as np
import time
import warnings
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

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

def _logsumexp_smooth_obj(x_flat, alpha=80.0):
    """Smooth surrogate using log-sum-exp for soft-min of triangle areas."""
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
    
    # -log(sum(exp(-alpha * areas))) / alpha approximates min(areas)
    # We want to maximize min(areas)/hull_area, so minimize -min(areas)/hull_area
    # soft_min = -log(sum(exp(-alpha*areas)))/alpha
    neg_alpha_areas = -alpha * areas
    # Use log-sum-exp trick for numerical stability
    max_neg = np.max(neg_alpha_areas)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_alpha_areas - max_neg)))
    soft_min = -log_sum / alpha
    
    score = soft_min / hull_area
    if score <= 0:
        return 1e10
    return -score

def _objective_full(x_flat):
    """Full objective: minimize negative score."""
    points = x_flat.reshape(N, 2)
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

def _local_polish(x0, deadline, max_time_per_method=8.0):
    """Multi-stage local optimization from a starting point."""
    best_x = x0.copy()
    best_s = _score(x0.reshape(N, 2))
    
    def _try_optimize(obj_func, method, options):
        nonlocal best_x, best_s
        if time.time() > deadline - 2:
            return
        try:
            res = minimize(obj_func, best_x, method=method, options=options)
            if res.x is not None:
                s = _score(res.x.reshape(N, 2))
                if s > best_s:
                    best_s = s
                    best_x = res.x.copy()
        except Exception:
            pass
    
    # Stage 1: Smooth objective with Powell
    if time.time() < deadline - 4:
        _try_optimize(
            lambda x: _logsumexp_smooth_obj(x, alpha=80.0),
            'Powell',
            {'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8}
        )
    
    # Stage 2: Smooth objective with different alpha
    if time.time() < deadline - 3:
        _try_optimize(
            lambda x: _logsumexp_smooth_obj(x, alpha=150.0),
            'Powell',
            {'maxiter': 150, 'xatol': 1e-9, 'fatol': 1e-9}
        )
    
    # Stage 3: Full objective with Nelder-Mead
    if time.time() < deadline - 2:
        _try_optimize(
            _objective_full,
            'Nelder-Mead',
            {'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
        )
    
    # Stage 4: Final high-precision Nelder-Mead
    if time.time() < deadline - 1:
        _try_optimize(
            _objective_full,
            'Nelder-Mead',
            {'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
        )
    
    return best_x, best_s

def _coordinate_descent_polish(x0, deadline, max_iter=200):
    """Coordinate descent: optimize each point's x and y separately."""
    best_x = x0.copy()
    best_s = _score(best_x.reshape(N, 2))
    
    for iteration in range(max_iter):
        if time.time() > deadline - 2:
            break
        
        improved = False
        for i in range(N):
            for dim in [0, 1]:
                # Optimize point i's coordinate dim
                def _obj_1d(val):
                    x_temp = best_x.copy()
                    x_temp[i * 2 + dim] = val
                    s = _score(x_temp.reshape(N, 2))
                    return -s if s > 0 else 1e10
                
                current_val = best_x[i * 2 + dim]
                # Try a small range around current value
                res = minimize(
                    _obj_1d,
                    [current_val],
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-9, 'fatol': 1e-9}
                )
                if res.x is not None and len(res.x) > 0:
                    new_val = res.x[0]
                    x_temp = best_x.copy()
                    x_temp[i * 2 + dim] = new_val
                    s_temp = _score(x_temp.reshape(N, 2))
                    if s_temp > best_s:
                        best_s = s_temp
                        best_x = x_temp.copy()
                        improved = True
        
        if not improved:
            break
    
    return best_x, best_s

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
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
    
    # 2. Two-ring configurations
    for n_out in [5, 6, 7, 8]:
        n_in = N - n_out
        for r_in in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            for offset in [0, np.pi / n_out, np.pi / (2 * n_out), np.pi / 6]:
                pts = _make_ring_config(n_out, n_in, r_in, offset)
                inits.append(pts.flatten())
    
    # 3. Three-ring configurations
    for n1, n2, n3 in [(4, 4, 5), (5, 4, 4), (6, 3, 4), (7, 3, 3), (3, 5, 5)]:
        for r2 in [0.4, 0.5, 0.6]:
            for r3 in [0.15, 0.25, 0.35]:
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
    
    # 5. Differential Evolution for global search
    if time.time() < deadline - 40:
        try:
            bounds = [(-1.5, 1.5)] * (2 * N)
            de_result = differential_evolution(
                _logsumexp_smooth_obj,
                bounds,
                args=(60.0,),
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-5,
                polish=False,
                updating='deferred',
                init='sobol'
            )
            if de_result.x is not None:
                inits.insert(0, de_result.x)
                s_de = _score(de_result.x.reshape(N, 2))
                if s_de > best_score:
                    best_score = s_de
                    best_points = de_result.x.reshape(N, 2).copy()
        except Exception:
            pass
    
    # Shuffle remaining inits
    rng2 = np.random.default_rng(999)
    if len(inits) > 1:
        indices = rng2.permutation(len(inits))
        inits = [inits[i] for i in indices]
    
    # Optimize from each start
    for i, x0 in enumerate(inits):
        if time.time() > deadline - 10:
            break
        
        # Quick local polish
        x_opt, s_opt = _local_polish(x0, deadline)
        _update_best(x_opt.reshape(N, 2))
        
        if i % 5 == 0:
            elapsed = time.time() - start_time
            print(f"Init {i}: best_score={best_score:.6f}, elapsed={elapsed:.1f}s", flush=True)
    
    # Final polish on best with coordinate descent
    if best_points is not None and time.time() < deadline - 5:
        x_cd, s_cd = _coordinate_descent_polish(best_points.flatten(), deadline, max_iter=100)
        _update_best(x_cd.reshape(N, 2))
    
    # Final high-precision polish
    if best_points is not None and time.time() < deadline - 1:
        try:
            res_final = minimize(
                _objective_full,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-13, 'fatol': 1e-13, 'adaptive': True}
            )
            _update_best(res_final.x.reshape(N, 2))
        except Exception:
            pass
    
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points

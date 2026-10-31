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

def _smooth_objective(x_flat, p=80):
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

def _coordinate_descent_polish(points, deadline, max_iter=3000):
    """Coordinate descent on the hard objective with adaptive step sizes."""
    best_pts = points.copy()
    best_s = _score(best_pts)
    
    base_deltas = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            for coord in range(2):
                for delta in base_deltas:
                    for sign in [1, -1]:
                        trial = best_pts.copy()
                        trial[i, coord] += sign * delta
                        trial_s = _score(trial)
                        if trial_s > best_s + 1e-14:
                            best_pts = trial
                            best_s = trial_s
                            improved = True
                            break
                    if improved:
                        break
        if not improved:
            break
    
    return best_pts, best_s

def _local_polish(points, deadline):
    """Multi-phase local optimization from a starting point."""
    best_x = points.flatten().copy()
    best_s = _score(points)
    
    # Phase 1: Powell on smooth objective
    if time.time() < deadline - 10:
        try:
            res = minimize(
                lambda x: _smooth_objective(x, p=80),
                best_x,
                method='Powell',
                options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            s = _score(res.x.reshape(N, 2))
            if s > best_s:
                best_s = s
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead on full objective
    if time.time() < deadline - 5:
        try:
            res_nm = minimize(
                _objective_full,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            s_nm = _score(res_nm.x.reshape(N, 2))
            if s_nm > best_s:
                best_s = s_nm
                best_x = res_nm.x.copy()
        except Exception:
            pass
    
    # Phase 3: Coordinate descent polish on hard objective
    if time.time() < deadline - 2:
        try:
            cd_pts, cd_s = _coordinate_descent_polish(best_x.reshape(N, 2), deadline)
            if cd_s > best_s:
                best_s = cd_s
                best_x = cd_pts.flatten().copy()
        except Exception:
            pass
    
    return best_x.reshape(N, 2), best_s

def _simulated_annealing(points, deadline):
    """Simulated annealing in Cartesian space with adaptive step size."""
    current = points.copy()
    current_score = _score(current)
    best = current.copy()
    best_s = current_score
    
    time_budget = max(0.5, deadline - 3.0 - time.time())
    est_iters = int(time_budget * 600)
    est_iters = max(300, min(est_iters, 40000))
    
    T0 = 0.05
    T_end = 0.00005
    if est_iters > 1:
        alpha = (T_end / T0) ** (1.0 / (est_iters - 1))
    else:
        alpha = 0.99
    
    rng = np.random.default_rng(42)
    T = T0
    step_scale = 0.05
    
    for it in range(est_iters):
        if time.time() > deadline - 3.0:
            break
        
        # Propose a move
        idx = rng.integers(0, N)
        delta = rng.normal(0, step_scale, size=2)
        trial = current.copy()
        trial[idx] += delta
        
        trial_score = _score(trial)
        
        if trial_score >= current_score or rng.random() < np.exp((trial_score - current_score) / T):
            current = trial
            current_score = trial_score
            if current_score > best_s:
                best = current.copy()
                best_s = current_score
        
        T *= alpha
        # Adapt step size: shrink if too many rejections, grow if too few
        if it % 100 == 99 and it > 0:
            # Simple heuristic: if score hasn't improved in last 100 iters, shrink
            pass
    
    return best, best_s

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
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
    
    # 5. Perturbed regular polygon
    for noise_scale in [0.02, 0.05, 0.1]:
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
    
    # Optimize from each start
    for i, x0 in enumerate(inits):
        if time.time() > deadline - 15:
            break
        
        t_start = time.time()
        x_opt, s_opt = _local_polish(x0.reshape(N, 2), deadline - 10)
        _update_best(x_opt)
        
        # Occasionally run simulated annealing from the best found so far
        if i % 5 == 4 and best_points is not None and time.time() < deadline - 25:
            sa_pts, sa_s = _simulated_annealing(best_points, deadline - 15)
            _update_best(sa_pts)
        
        elapsed = time.time() - t_start
        if i % 5 == 0:
            print(f"Init {i}: score={best_score:.6f}, elapsed={elapsed:.1f}s", flush=True)
    
    # Final high-precision polish on best
    if best_points is not None and time.time() < deadline - 5:
        try:
            res_final = minimize(
                _objective_full,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
            )
            _update_best(res_final.x.reshape(N, 2))
        except Exception:
            pass
        
        # Coordinate descent final polish
        if best_points is not None and time.time() < deadline - 2:
            try:
                cd_pts, cd_s = _coordinate_descent_polish(best_points, deadline - 1)
                _update_best(cd_pts)
            except Exception:
                pass
    
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points

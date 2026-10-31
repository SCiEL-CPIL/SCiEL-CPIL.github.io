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

def _smooth_objective(x_flat, p=50):
    """Smooth surrogate using log-sum-exp based soft-min for better gradients."""
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

def _coordinate_descent_polish(points, deadline, max_iter=5000):
    """Coordinate descent on the hard objective: move each point's x and y independently."""
    best_pts = points.copy()
    best_s = _score(best_pts)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            for coord in range(2):
                for delta in [0.01, -0.01, 0.005, -0.005, 0.002, -0.002, 0.001, -0.001]:
                    trial = best_pts.copy()
                    trial[i, coord] += delta
                    trial_s = _score(trial)
                    if trial_s > best_s + 1e-14:
                        best_pts = trial
                        best_s = trial_s
                        improved = True
                        break
        
        if not improved:
            break
    
    return best_pts, best_s

def _local_polish(points, deadline):
    """Multi-phase local optimization from a starting point."""
    best_x = points.flatten().copy()
    best_s = _score(points)
    
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
    
    if time.time() < deadline - 2:
        try:
            cd_pts, cd_s = _coordinate_descent_polish(best_x.reshape(N, 2), deadline)
            if cd_s > best_s:
                best_s = cd_s
                best_x = cd_pts.flatten().copy()
        except Exception:
            pass
    
    return best_x.reshape(N, 2), best_s

def _get_base_configs():
    """Generate a list of base configurations to be affinely transformed."""
    configs = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Hexagonal lattice: center + 6 + 6
    # Center
    center = np.array([[0.0, 0.0]])
    # Ring 1
    angles1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    ring1 = np.column_stack([np.cos(angles1), np.sin(angles1)])
    # Ring 2 (rotated by 30 deg, radius 2)
    angles2 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/6
    ring2 = np.column_stack([2.0 * np.cos(angles2), 2.0 * np.sin(angles2)])
    configs.append(np.vstack([center, ring1, ring2]))
    
    # 3. Square lattice subset
    # Generate grid points and select 13 closest to origin
    grid_x = np.arange(-2, 3)
    grid_y = np.arange(-2, 3)
    gx, gy = np.meshgrid(grid_x, grid_y)
    dists = np.sqrt(gx**2 + gy**2)
    flat_dists = dists.flatten()
    flat_idx = np.argsort(flat_dists)[:N]
    grid_pts = np.column_stack([gx.flatten(), gy.flatten()])
    configs.append(grid_pts[flat_idx])
    
    # 4. Two-ring: 7 outer, 6 inner
    angles_out = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    angles_in = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/7
    outer = np.column_stack([np.cos(angles_out), np.sin(angles_out)])
    inner = np.column_stack([0.5 * np.cos(angles_in), 0.5 * np.sin(angles_in)])
    configs.append(np.vstack([outer, inner]))
    
    # 5. Two-ring: 8 outer, 5 inner
    angles_out = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    angles_in = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi/8
    outer = np.column_stack([np.cos(angles_out), np.sin(angles_out)])
    inner = np.column_stack([0.5 * np.cos(angles_in), 0.5 * np.sin(angles_in)])
    configs.append(np.vstack([outer, inner]))
    
    return configs

def _affine_objective(params, base_config):
    """
    Objective for affine transform optimization.
    params: [a, b, c, d] for matrix [[a, b], [c, d]]
    """
    a, b, c, d = params
    M = np.array([[a, b], [c, d]])
    points = base_config @ M.T
    # Center
    points = points - np.mean(points, axis=0)
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
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
    
    bases = _get_base_configs()
    
    # === Phase 1: Affine Global Search for each base ===
    # We have limited time, so we do a quick DE for each base
    # or a single DE over all bases if we could parameterize the base choice,
    # but discrete base choice is hard for DE. So we loop.
    
    # To save time, we only run DE on the most promising bases if time is tight.
    # Let's run DE on all 5 bases with a small budget each.
    
    n_bases = len(bases)
    time_per_base = (deadline - 30.0) / n_bases # Leave 30s for local polish
    
    for i, base in enumerate(bases):
        if time.time() > deadline - 30.0:
            break
            
        t_start = time.time()
        t_budget = min(time_per_base, deadline - 30.0 - time.time())
        if t_budget < 5.0:
            continue
            
        try:
            bounds = [(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
            # Use a smaller population and fewer iterations to fit in budget
            max_iter = int(t_budget * 20) # Heuristic
            max_iter = max(20, min(max_iter, 100))
            
            de_result = differential_evolution(
                lambda x: _affine_objective(x, base),
                bounds=bounds,
                seed=42 + i,
                maxiter=max_iter,
                popsize=20,
                tol=1e-6,
                mutation=(0.4, 1.0),
                recombination=0.7,
                polish=False,
                updating='deferred'
            )
            
            if de_result.fun < 1e9:
                a, b, c, d = de_result.x
                M = np.array([[a, b], [c, d]])
                pts = base @ M.T
                pts = pts - np.mean(pts, axis=0)
                _update_best(pts)
                
        except Exception:
            pass
    
    # === Phase 2: Local Polish ===
    # If we have a best point, polish it.
    # Also, try polishing the original bases in case affine search failed or was weak.
    
    inits_to_polish = []
    if best_points is not None:
        inits_to_polish.append(best_points)
    
    for base in bases:
        inits_to_polish.append(base)
    
    # Shuffle to ensure diversity
    rng = np.random.default_rng(42)
    rng.shuffle(inits_to_polish)
    
    # Limit number of polishes to fit in time
    max_polishes = 10
    for i, init_pts in enumerate(inits_to_polish[:max_polishes]):
        if time.time() > deadline - 10.0:
            break
        
        x_opt, s_opt = _local_polish(init_pts, deadline)
        _update_best(x_opt)
        
        if i % 3 == 0:
            elapsed = time.time() - start_time
            print(f"Polish {i}/{min(len(inits_to_polish), max_polishes)}: best_score={best_score:.6f}, elapsed={elapsed:.1f}s", flush=True)
    
    # === Phase 3: Final High-Precision Polish ===
    if best_points is not None and time.time() < deadline - 3:
        try:
            # One more deep polish
            x_opt, s_opt = _local_polish(best_points, deadline)
            _update_best(x_opt)
        except Exception:
            pass
    
    # Fallback
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points

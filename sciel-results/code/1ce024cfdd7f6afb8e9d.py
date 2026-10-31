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
    
    # Adaptive p: use higher p when areas are more spread out (sharper min)
    spread = np.max(areas) - np.min(areas)
    if spread > 1e-10:
        p_eff = p * (1.0 + 0.5 * np.tanh(spread * 10))
    else:
        p_eff = p
    
    neg_p_areas = -p_eff * areas
    max_neg = np.max(neg_p_areas)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_p_areas - max_neg)))
    soft_min = -log_sum / p_eff
    
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

def _coordinate_descent_polish(points, deadline, max_iter=200):
    """Coordinate descent on the hard objective: move each point's x and y independently.
    Uses in-place updates and adaptive step sizes for efficiency."""
    best_pts = points.copy()
    best_s = _score(best_pts)
    
    # Adaptive step sizes
    steps = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    
    for it in range(max_iter):
        if time.time() > deadline - 1:
            break
        
        improved = False
        # Shuffle point order each iteration to avoid bias
        order = np.random.default_rng(it).permutation(N)
        
        for i in order:
            for coord in range(2):
                orig_val = best_pts[i, coord]
                for delta in steps:
                    # Try positive delta
                    best_pts[i, coord] = orig_val + delta
                    trial_s = _score(best_pts)
                    if trial_s > best_s + 1e-15:
                        best_s = trial_s
                        improved = True
                        break
                    # Try negative delta
                    best_pts[i, coord] = orig_val - delta
                    trial_s = _score(best_pts)
                    if trial_s > best_s + 1e-15:
                        best_s = trial_s
                        improved = True
                        break
                # Reset if no improvement found
                if not improved:
                    best_pts[i, coord] = orig_val
        
        if not improved:
            # Try smaller steps
            steps = [s * 0.5 for s in steps]
            if steps[0] < 1e-7:
                break
    
    return best_pts, best_s

def _local_polish(points, deadline):
    """Multi-phase local optimization from a starting point."""
    best_x = points.flatten().copy()
    best_s = _score(points)
    
    # Phase 1: Powell with smooth surrogate
    if time.time() < deadline - 15:
        try:
            res = minimize(
                lambda x: _smooth_objective(x, p=60),
                best_x,
                method='Powell',
                options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-9}
            )
            s = _score(res.x.reshape(N, 2))
            if s > best_s:
                best_s = s
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead on exact objective
    if time.time() < deadline - 8:
        try:
            res_nm = minimize(
                _objective_full,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-11, 'fatol': 1e-11, 'adaptive': True}
            )
            s_nm = _score(res_nm.x.reshape(N, 2))
            if s_nm > best_s:
                best_s = s_nm
                best_x = res_nm.x.copy()
        except Exception:
            pass
    
    # Phase 3: Coordinate descent polish
    if time.time() < deadline - 3:
        try:
            cd_pts, cd_s = _coordinate_descent_polish(best_x.reshape(N, 2), deadline)
            if cd_s > best_s:
                best_s = cd_s
                best_x = cd_pts.flatten().copy()
        except Exception:
            pass
    
    # Phase 4: Final tight polish with very small steps
    if time.time() < deadline - 1:
        try:
            final_pts = best_x.reshape(N, 2).copy()
            final_s = best_s
            for _ in range(50):
                if time.time() > deadline - 0.5:
                    break
                improved = False
                for i in range(N):
                    for coord in range(2):
                        for delta in [0.0001, -0.0001, 0.00005, -0.00005]:
                            trial = final_pts.copy()
                            trial[i, coord] += delta
                            trial_s = _score(trial)
                            if trial_s > final_s + 1e-16:
                                final_pts = trial
                                final_s = trial_s
                                improved = True
                                break
                if not improved:
                    break
            if final_s > best_s:
                best_s = final_s
                best_x = final_pts.flatten().copy()
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
    center = np.array([[0.0, 0.0]])
    angles1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    ring1 = np.column_stack([np.cos(angles1), np.sin(angles1)])
    angles2 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/6
    ring2 = np.column_stack([2.0 * np.cos(angles2), 2.0 * np.sin(angles2)])
    configs.append(np.vstack([center, ring1, ring2]))
    
    # 3. Square lattice subset
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
    
    # 6. Golden angle spiral (Fibonacci distribution)
    n = N
    idx = np.arange(n)
    r = np.sqrt(idx / n)
    theta = idx * 2 * np.pi * 0.618033988749895  # golden angle
    configs.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))
    
    # 7. Three-ring: 1 center, 4 middle, 8 outer
    c = np.array([[0.0, 0.0]])
    ang_mid = np.linspace(0, 2*np.pi, 4, endpoint=False)
    mid = np.column_stack([0.5*np.cos(ang_mid), 0.5*np.sin(ang_mid)])
    ang_out = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
    out = np.column_stack([1.0*np.cos(ang_out), 1.0*np.sin(ang_out)])
    configs.append(np.vstack([c, mid, out]))
    
    # 8. Distorted 13-gon with alternating radii
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radii = np.ones(N)
    radii[::2] = 0.85  # every other point slightly inward
    configs.append(np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]))
    
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
    deadline = start_time + 147.0  # 3s safety margin
    
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
    n_bases = len(bases)
    # Allocate more time to affine search, less to polish
    affine_time_budget = min(45.0, (deadline - 60.0))
    time_per_base = affine_time_budget / n_bases
    
    for i, base in enumerate(bases):
        if time.time() > deadline - 60.0:
            break
            
        t_budget = min(time_per_base, deadline - 60.0 - time.time())
        if t_budget < 3.0:
            continue
            
        try:
            bounds = [(-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0)]
            max_iter = int(t_budget * 15)
            max_iter = max(15, min(max_iter, 80))
            
            de_result = differential_evolution(
                lambda x: _affine_objective(x, base),
                bounds=bounds,
                seed=42 + i,
                maxiter=max_iter,
                popsize=15,
                tol=1e-5,
                mutation=(0.5, 1.0),
                recombination=0.8,
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
    inits_to_polish = []
    if best_points is not None:
        inits_to_polish.append(best_points)
    
    for base in bases:
        inits_to_polish.append(base)
    
    # Prioritize best points first, then diverse bases
    rng = np.random.default_rng(42)
    # Keep best first, shuffle the rest
    if len(inits_to_polish) > 1:
        rest = inits_to_polish[1:]
        rng.shuffle(rest)
        inits_to_polish = [inits_to_polish[0]] + rest
    
    # Limit number of polishes to fit in time
    max_polishes = min(len(inits_to_polish), 12)
    for i, init_pts in enumerate(inits_to_polish[:max_polishes]):
        if time.time() > deadline - 12.0:
            break
        
        x_opt, s_opt = _local_polish(init_pts, deadline)
        _update_best(x_opt)
        
        if i % 2 == 0:
            elapsed = time.time() - start_time
            print(f"Polish {i}/{max_polishes}: best_score={best_score:.6f}, elapsed={elapsed:.1f}s", flush=True)
    
    # === Phase 3: Final High-Precision Polish ===
    if best_points is not None and time.time() < deadline - 5:
        try:
            x_opt, s_opt = _local_polish(best_points, deadline)
            _update_best(x_opt)
        except Exception:
            pass
    
    # Fallback
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points

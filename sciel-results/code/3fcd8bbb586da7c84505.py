"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist

def _compute_score(points):
    """Compute the maximum radius for given points."""
    n = len(points)
    if n < 2:
        return 0.0
    dists = cdist(points, points)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    min_pair_r = min_pair / 2.0
    norms = np.linalg.norm(points, axis=1)
    min_boundary = 1.0 - np.max(norms)
    return min(min_pair_r, min_boundary)

def _soft_min_surrogate(points, beta=100.0):
    """Smooth surrogate for min of pairwise distances/2 and boundary distances."""
    n = len(points)
    if n < 2:
        return 0.0
    dists = cdist(points, points)
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    flat_vals = pair_vals.flatten()
    flat_vals = flat_vals[~np.isinf(flat_vals)]
    norms = np.linalg.norm(points, axis=1)
    boundary_vals = 1.0 - norms
    all_vals = np.concatenate([flat_vals, boundary_vals])
    m = np.min(all_vals)
    exp_vals = np.exp(-beta * (all_vals - m))
    soft_min = m - np.log(np.sum(exp_vals)) / beta
    return soft_min

def _objective(x, beta=100.0):
    """Objective: negative soft-min radius + boundary penalty."""
    pts = x.reshape(N, 2)
    norms = np.linalg.norm(pts, axis=1)
    boundary_violation = np.maximum(0, norms - 1.0)
    penalty = 100.0 * np.sum(boundary_violation)
    radius = _soft_min_surrogate(pts, beta)
    return -radius + penalty

def _ring_initialization(ring_counts=None, ring_radii=None, ring_angles=None):
    """Generate initial configuration with concentric rings."""
    if ring_counts is None:
        ring_counts = [1, 6, 12, 18, 13]
    if ring_radii is None:
        ring_radii = [0.0, 0.25, 0.50, 0.75, 0.92]
    if ring_angles is None:
        ring_angles = [0.0, 0.0, np.pi/12, 0.0, np.pi/13]
    
    points = []
    for ring_idx, (count, radius, angle_offset) in enumerate(zip(ring_counts, ring_radii, ring_angles)):
        if count == 0:
            continue
        if count == 1 and radius < 0.01:
            points.append([0.0, 0.0])
        else:
            for i in range(count):
                angle = 2 * np.pi * i / count + angle_offset
                points.append([radius * np.cos(angle), radius * np.sin(angle)])
    
    return np.array(points[:N])

def _hexagonal_initial():
    """Generate initial hexagonal lattice packing."""
    spacing = 0.25
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points, radius = points[mask], radius[mask]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98

def _polish_smooth(points, start_time, deadline):
    """High-precision polish using smooth surrogate with increasing beta."""
    current = points.copy()
    current_score = _compute_score(current)
    betas = [30, 60, 120, 240, 480, 960]
    
    for beta in betas:
        if time.time() > deadline - 3:
            break
        
        def obj(x):
            return _objective(x, beta)
        
        try:
            result = minimize(
                obj, current.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50}
            )
            new_points = result.x.reshape(N, 2)
            if np.all(np.isfinite(new_points)):
                norms = np.linalg.norm(new_points, axis=1)
                if np.max(norms) <= 1.0 + 1e-12:
                    mask = norms > 1.0
                    if np.any(mask):
                        new_points[mask] /= norms[mask, np.newaxis] * 1.0001
                    new_score = _compute_score(new_points)
                    if new_score > current_score:
                        current = new_points
                        current_score = new_score
        except Exception:
            pass
    
    return current, current_score

def _polish_exact(points, start_time, deadline):
    """Final polish using direct optimization of the exact objective."""
    current = points.copy()
    current_score = _compute_score(current)
    penalty_weights = [10, 50, 200, 1000]
    
    for pw in penalty_weights:
        if time.time() > deadline - 2:
            break
        
        def exact_obj(x):
            pts = x.reshape(N, 2)
            dists = cdist(pts, pts)
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists)
            min_pair_r = min_pair / 2.0
            norms = np.linalg.norm(pts, axis=1)
            min_boundary = 1.0 - np.max(norms)
            r = min(min_pair_r, min_boundary)
            violation = 0.0
            if min_pair_r < r:
                violation += (r - min_pair_r)
            if min_boundary < r:
                violation += (r - min_boundary)
            return -r + pw * violation**2
        
        try:
            result = minimize(
                exact_obj, current.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            new_points = result.x.reshape(N, 2)
            if np.all(np.isfinite(new_points)):
                norms = np.linalg.norm(new_points, axis=1)
                if np.max(norms) <= 1.0 + 1e-12:
                    mask = norms > 1.0
                    if np.any(mask):
                        new_points[mask] /= norms[mask, np.newaxis] * 1.0001
                    new_score = _compute_score(new_points)
                    if new_score > current_score:
                        current = new_points
                        current_score = new_score
        except Exception:
            pass
    
    return current, current_score

def _parametric_points(radii, angles):
    """Build 50 points from ring parameters."""
    ring_sizes = [1, 6, 12, 18, 13]
    points = []
    points.append([0.0, 0.0])
    for ring_idx in range(1, 5):
        size = ring_sizes[ring_idx]
        r = radii[ring_idx - 1]
        angle_offset = angles[ring_idx - 1]
        for i in range(size):
            angle = 2 * np.pi * i / size + angle_offset
            points.append([r * np.cos(angle), r * np.sin(angle)])
    return np.array(points)

def _parametric_objective(params):
    """Objective for parametric optimization."""
    radii = params[:4]
    angles = params[4:8]
    points = _parametric_points(radii, angles)
    norms = np.linalg.norm(points, axis=1)
    if np.max(norms) > 1.0:
        return 100.0
    radius = _soft_min_surrogate(points, beta=50.0)
    return -radius

def _optimize_parametric(start_time, deadline):
    """Optimize ring radii and angles using differential evolution."""
    bounds = [
        (0.15, 0.40), (0.35, 0.65), (0.55, 0.85), (0.75, 0.98),
        (0.0, 2 * np.pi), (0.0, 2 * np.pi), (0.0, 2 * np.pi), (0.0, 2 * np.pi)
    ]
    x0 = np.array([0.25, 0.50, 0.75, 0.90, 0.0, np.pi/12, 0.0, np.pi/13])
    
    best_points = None
    best_score = -np.inf
    
    for seed in [42, 123, 456]:
        if time.time() > deadline - 25:
            break
        try:
            result_de = differential_evolution(
                _parametric_objective, bounds,
                seed=seed, maxiter=150, tol=1e-10, polish=False, init='sobol'
            )
            if time.time() > deadline - 20:
                break
            best_params = result_de.x
            pts = _parametric_points(best_params[:4], best_params[4:8])
            score = _compute_score(pts)
            if score > best_score:
                best_score = score
                best_points = pts.copy()
        except Exception:
            pass
    
    if best_points is not None and time.time() < deadline - 10:
        best_points, best_score = _polish_smooth(best_points, start_time, deadline)
    
    return best_points, best_score

def _binary_search_polish(points, start_time, deadline):
    """Binary search on target radius to squeeze out last bits."""
    current = points.copy()
    current_score = _compute_score(current)
    t_start = time.time()
    lo, hi = current_score * 0.999, current_score * 1.001
    
    for _ in range(20):
        if time.time() > deadline - 5 or (time.time() - t_start) > 30:
            break
        mid = (lo + hi) / 2
        
        def penalized_obj(x, target_r=mid):
            c_local = x.reshape(N, 2)
            b = 1.0 - np.linalg.norm(c_local, axis=1)
            b_pen = np.sum(np.maximum(0, target_r - b) ** 2)
            diff = c_local[:, np.newaxis, :] - c_local[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            iu = np.triu_indices(N, 1)
            pair_dists = dists[iu] / 2.0
            p_pen = np.sum(np.maximum(0, target_r - pair_dists) ** 2)
            return b_pen + p_pen
        
        try:
            res = minimize(
                penalized_obj, current.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-15}
            )
            if np.all(np.isfinite(res.x)):
                cand = res.x.reshape(N, 2)
                r_cand = _compute_score(cand)
                if r_cand >= mid - 1e-10:
                    lo = mid
                    current = cand.copy()
                    current_score = r_cand
                else:
                    hi = mid
        except Exception:
            break
    
    return current, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Phase 1: Parametric optimization
    param_points, param_score = _optimize_parametric(start_time, deadline)
    
    # Phase 2: Multiple initializations
    ring_configs = [
        ([1, 6, 12, 18, 13], [0.0, 0.25, 0.50, 0.75, 0.92], [0.0, 0.0, np.pi/12, 0.0, np.pi/13]),
        ([1, 6, 12, 18, 13], [0.0, 0.24, 0.48, 0.72, 0.90], [0.0, 0.0, np.pi/12, 0.0, np.pi/13]),
        ([1, 6, 12, 18, 13], [0.0, 0.26, 0.52, 0.78, 0.94], [0.0, 0.0, np.pi/12, 0.0, np.pi/13]),
        ([1, 6, 13, 17, 13], [0.0, 0.25, 0.51, 0.75, 0.93], [0.0, 0.0, np.pi/13, 0.0, np.pi/13]),
    ]
    
    initializations = []
    for counts, radii, angles in ring_configs:
        initializations.append(_ring_initialization(counts, radii, angles))
    initializations.append(_hexagonal_initial())
    
    best_points = None
    best_score = -np.inf
    
    if param_points is not None and param_score > best_score:
        best_points = param_points.copy()
        best_score = param_score
    
    for init_points in initializations:
        if time.time() > deadline - 60:
            break
        score = _compute_score(init_points)
        if score > best_score:
            best_score = score
            best_points = init_points.copy()
    
    # Phase 3: Simulated annealing
    rng = np.random.default_rng(42)
    current_points = best_points.copy()
    current_score = best_score
    T = 0.003
    T_min = 1e-10
    cooling = 0.99985
    max_iter = 500000
    norms = np.linalg.norm(current_points, axis=1)
    
    for iteration in range(max_iter):
        if time.time() > deadline - 40:
            break
        idx = rng.integers(0, N)
        scale = max(T, 1e-7) * 0.12
        delta = rng.normal(0, scale, size=2)
        new_points = current_points.copy()
        new_points[idx] += delta
        norm_idx = np.linalg.norm(new_points[idx])
        if norm_idx > 1.0:
            new_points[idx] = new_points[idx] / norm_idx * 0.9995
        diffs = new_points - new_points[idx]
        dists = np.linalg.norm(diffs, axis=1)
        dists[idx] = np.inf
        min_pair = np.min(dists)
        min_pair_r = min_pair / 2.0
        new_norms = norms.copy()
        new_norms[idx] = np.linalg.norm(new_points[idx])
        new_min_boundary = 1.0 - np.max(new_norms)
        new_score = min(min_pair_r, new_min_boundary)
        
        if new_score > current_score or rng.random() < np.exp((new_score - current_score) / max(T, 1e-12)):
            current_points = new_points
            current_score = new_score
            norms = new_norms
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        
        T *= cooling
        if T < T_min:
            break
    
    # Phase 4: Polish
    if time.time() < deadline - 15:
        best_points, best_score = _polish_smooth(best_points, start_time, deadline)
    
    if time.time() < deadline - 10:
        best_points, best_score = _binary_search_polish(best_points, start_time, deadline)
    
    if time.time() < deadline - 3:
        best_points, best_score = _polish_exact(best_points, start_time, deadline)
    
    return best_points
# EVOLVE-BLOCK-END

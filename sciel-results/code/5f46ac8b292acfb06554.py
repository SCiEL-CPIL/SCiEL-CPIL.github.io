import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

N = 13

def _eval_hierarchical(params):
    """
    Hierarchical parameterization:
    - params[0]: ellipse semi-axis a (major)
    - params[1]: ellipse semi-axis b (minor)
    - params[2]: rotation angle theta
    - params[3:3+N]: angular deviations from regular N-gon (small perturbations)
    
    Points are placed on the ellipse at angles (2*pi*k/N + delta_k),
    mapped through the affine transformation defined by (a, b, theta).
    """
    a = params[0]
    b = params[1]
    theta = params[2]
    deltas = params[3:3 + N]
    
    # Ensure positive semi-axes
    a = max(abs(a), 0.1)
    b = max(abs(b), 0.1)
    
    # Base angles for regular N-gon
    base_angles = 2.0 * np.pi * np.arange(N) / N
    
    # Apply angular deviations
    angles = base_angles + deltas
    
    # Points on unit circle, then scale by ellipse
    # Ellipse: x = a*cos(t), y = b*sin(t), then rotate by theta
    cx = a * np.cos(angles)
    cy = b * np.sin(angles)
    
    # Rotate by theta
    ct = np.cos(theta)
    st = np.sin(theta)
    x = ct * cx - st * cy
    y = st * cx + ct * cy
    
    points = np.column_stack([x, y])
    return points

def _neg_hierarchical_score(params):
    """Objective for hierarchical optimization (maximize score)."""
    points = _eval_hierarchical(params)
    s = _score(points)
    return -s

# Pre-compute triangle indices for N=13
_TRI_INDICES = np.array(list(combinations(range(N), 3)))
_I_IDX = _TRI_INDICES[:, 0]
_J_IDX = _TRI_INDICES[:, 1]
_K_IDX = _TRI_INDICES[:, 2]

def _get_triangle_areas(points):
    """
    Vectorized calculation of areas for all triples.
    points: (N, 2) array
    returns: array of shape (num_triangles,)
    """
    pi = points[_I_IDX]
    pj = points[_J_IDX]
    pk = points[_K_IDX]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return areas

def _hull_area(points):
    """Calculate convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Calculate the raw score: min_triangle_area / hull_area."""
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    areas = _get_triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    min_area = np.min(areas)
    return min_area / hull_area

def _softmin_score(points, alpha):
    """
    Smooth approximation of the score using softmin of triangle areas.
    This provides a differentiable-like landscape for optimization.
    """
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    
    areas = _get_triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    
    # Softmin: -1/alpha * log(sum(exp(-alpha * areas)))
    # This is a smooth approximation of min(areas)
    # Subtract max for numerical stability
    a_min = np.min(areas)
    exp_terms = np.exp(-alpha * (areas - a_min))
    log_sum = np.log(np.sum(exp_terms))
    softmin_val = a_min - np.log(np.sum(exp_terms)) / alpha
    
    return softmin_val / hull_area

def _neg_softmin_score(params, alpha):
    """Objective for softmin-based optimization."""
    points = params.reshape(N, 2)
    s = _softmin_score(points, alpha)
    return -s

def _make_ring_config(n_points, radius, phase_offset=0.0, n_rings=1, ring_radii=None):
    """Generate a ring-based initial configuration."""
    if ring_radii is None:
        if n_rings == 1:
            ring_radii = [radius]
        else:
            ring_radii = np.linspace(0.3, radius, n_rings).tolist()
    
    points = []
    per_ring = n_points // n_rings
    remainder = n_points % n_rings
    
    for ring_idx, r in enumerate(ring_radii):
        count = per_ring + (1 if ring_idx < remainder else 0)
        for i in range(count):
            angle = 2 * np.pi * i / count + phase_offset + ring_idx * 0.3
            points.append([r * np.cos(angle), r * np.sin(angle)])
    
    # Fill remaining points if needed
    while len(points) < n_points:
        idx = len(points)
        angle = 2 * np.pi * idx / n_points + phase_offset
        points.append([radius * 0.5 * np.cos(angle), radius * 0.5 * np.sin(angle)])
    
    return np.array(points[:n_points])

def _optimize_from_start(start_pts, deadline, rng):
    """
    Multi-stage optimization from a starting point:
    1. Softmin-based Nelder-Mead with decreasing alpha
    2. L-BFGS-B polish
    3. Nelder-Mead on hard score
    Returns best points and score found.
    """
    best_pts = start_pts.copy()
    best_score = _score(start_pts)
    
    # Stage 1: Softmin optimization with annealing alpha
    alphas = [100, 50, 20, 10, 5, 2]
    for alpha in alphas:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(
                _neg_softmin_score,
                best_pts.flatten(),
                args=(alpha,),
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True}
            )
            cand_pts = res.x.reshape(N, 2)
            score = _score(cand_pts)
            if score > best_score:
                best_score = score
                best_pts = cand_pts
        except Exception:
            pass
    
    # Stage 2: L-BFGS-B polish with softmin (multiple alpha values)
    for alpha_polish in [2.0, 1.0, 0.5]:
        if time.time() < deadline - 10:
            try:
                res = minimize(
                    _neg_softmin_score,
                    best_pts.flatten(),
                    args=(alpha_polish,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                cand_pts = res.x.reshape(N, 2)
                score = _score(cand_pts)
                if score > best_score:
                    best_score = score
                    best_pts = cand_pts
            except Exception:
                pass
    
    # Stage 3: Nelder-Mead on hard score
    if time.time() < deadline - 5:
        try:
            res = minimize(
                _neg_score,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
            )
            cand_pts = res.x.reshape(N, 2)
            score = _score(cand_pts)
            if score > best_score:
                best_score = score
                best_pts = cand_pts
        except Exception:
            pass
    
    return best_pts, best_score

def _neg_score(params):
    """Objective function for minimization (maximizing score)."""
    points = params.reshape(N, 2)
    s = _score(points)
    return -s

def _coordinate_local_search(points, deadline, rng, initial_step=0.05, min_step=1e-6, max_iter=200):
    """
    Coordinate-wise pattern search on the hard score.
    Tries moving each coordinate in +/- direction, accepting improvements.
    Adapts step size based on success/failure.
    """
    best_pts = points.copy()
    best_score = _score(best_pts)
    step = initial_step
    n_iter = 0
    
    while time.time() < deadline - 5 and step > min_step and n_iter < max_iter:
        improved = False
        for i in range(N):
            for dim in range(2):
                for direction in [1, -1]:
                    trial = best_pts.copy()
                    trial[i, dim] += direction * step
                    trial_score = _score(trial)
                    if trial_score > best_score:
                        best_pts = trial
                        best_score = trial_score
                        improved = True
        n_iter += 1
        if improved:
            step *= 1.2  # expand step on success
        else:
            step *= 0.5  # shrink step on failure
    
    return best_pts, best_score

def _polish_with_local_search(points, deadline, rng):
    """
    Multi-stage polishing: coordinate search + Nelder-Mead + softmin polish.
    """
    best_pts = points.copy()
    best_score = _score(best_pts)
    
    # Stage 1: Coordinate-wise local search
    if time.time() < deadline - 10:
        cand_pts, score = _coordinate_local_search(best_pts, deadline, rng)
        if score > best_score:
            best_score = score
            best_pts = cand_pts
    
    # Stage 2: Nelder-Mead on hard score
    if time.time() < deadline - 8:
        try:
            res = minimize(
                _neg_score,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
            )
            cand_pts = res.x.reshape(N, 2)
            score = _score(cand_pts)
            if score > best_score:
                best_score = score
                best_pts = cand_pts
        except Exception:
            pass
    
    # Stage 3: Fine coordinate search
    if time.time() < deadline - 5:
        cand_pts, score = _coordinate_local_search(best_pts, deadline, rng, initial_step=0.01)
        if score > best_score:
            best_score = score
            best_pts = cand_pts
    
    # Stage 4: Ultra-fine Nelder-Mead
    if time.time() < deadline - 3:
        try:
            res = minimize(
                _neg_score,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            cand_pts = res.x.reshape(N, 2)
            score = _score(cand_pts)
            if score > best_score:
                best_score = score
                best_pts = cand_pts
        except Exception:
            pass
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 150s kill
    
    best_score = 0.0
    best_points = None
    rng = np.random.RandomState(42)
    
    # === PHASE 1: Hierarchical parameterization via DE ===
    # Parameters: [a, b, theta, delta_0, ..., delta_12]
    # a, b: ellipse semi-axes (positive)
    # theta: rotation angle
    # deltas: angular deviations from regular N-gon
    
    # DE bounds: a in [0.5, 2.0], b in [0.5, 2.0], theta in [-pi, pi], deltas in [-0.3, 0.3]
    bounds_de = [(0.5, 2.0), (0.5, 2.0), (-np.pi, np.pi)] + [(-0.3, 0.3)] * N
    
    # Run DE on hierarchical parameterization
    de_results = []
    if time.time() < deadline - 40:
        try:
            de_res = differential_evolution(
                _neg_hierarchical_score,
                bounds=bounds_de,
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=True,
                updating='deferred',
                init='latinhypercube'
            )
            if de_res.success or de_res.fun < 0:
                de_points = _eval_hierarchical(de_res.x)
                de_score = _score(de_points)
                de_results.append((de_score, de_points))
                if de_score > best_score:
                    best_score = de_score
                    best_points = de_points
        except Exception:
            pass
    
    # Additional DE run with tighter delta bounds
    if time.time() < deadline - 30:
        try:
            bounds_tight = [(0.8, 1.5), (0.8, 1.5), (-np.pi/4, np.pi/4)] + [(-0.1, 0.1)] * N
            de_res2 = differential_evolution(
                _neg_hierarchical_score,
                bounds=bounds_tight,
                seed=123,
                maxiter=80,
                popsize=12,
                tol=1e-7,
                mutation=(0.2, 0.8),
                recombination=0.9,
                polish=True,
                updating='deferred',
                init='latinhypercube'
            )
            if de_res2.success or de_res2.fun < 0:
                de_points2 = _eval_hierarchical(de_res2.x)
                de_score2 = _score(de_points2)
                de_results.append((de_score2, de_points2))
                if de_score2 > best_score:
                    best_score = de_score2
                    best_points = de_points2
        except Exception:
            pass
    
    # === PHASE 2: Traditional diverse initial configs ===
    initial_configs = []
    
    # Regular polygon
    angles = 2 * np.pi * np.arange(N) / N
    initial_configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # Slightly perturbed regular polygons
    for perturb in [0.02, 0.05, 0.1]:
        angles_p = 2 * np.pi * np.arange(N) / N + rng.uniform(-perturb, perturb, N)
        radii_p = 1.0 + rng.uniform(-perturb, perturb, N)
        initial_configs.append(np.column_stack([radii_p * np.cos(angles_p), radii_p * np.sin(angles_p)]))
    
    # Ring configurations
    for radius in [1.0, 1.2, 0.8]:
        for phase in [0.0, np.pi / N, np.pi / (2 * N)]:
            initial_configs.append(_make_ring_config(N, radius, phase_offset=phase))
    
    # Two-ring structures
    for r1, r2 in [(0.5, 1.0), (0.4, 1.1), (0.6, 0.9)]:
        for phase in [0.0, np.pi / N]:
            initial_configs.append(_make_ring_config(N, 1.0, phase_offset=phase, n_rings=2, ring_radii=[r1, r2]))
    
    # Radial perturbation
    for amp in [0.1, 0.2, 0.3]:
        for freq in [2, 3, 5]:
            angles = 2 * np.pi * np.arange(N) / N
            radii = 1.0 + amp * np.sin(freq * angles)
            initial_configs.append(np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]))
    
    # Random configurations
    for _ in range(5):
        initial_configs.append(rng.uniform(-1, 1, (N, 2)))
    
    # Add DE results to configs
    for s_de, pts_de in de_results:
        initial_configs.append(pts_de)
    
    # Sort by initial score
    scored_configs = []
    for config in initial_configs:
        try:
            s = _score(config)
            scored_configs.append((s, config))
        except Exception:
            pass
    scored_configs.sort(key=lambda x: -x[0])
    
    # Optimize top configs
    for s_init, config in scored_configs:
        if time.time() > deadline - 20:
            break
        
        try:
            cand_pts, score = _optimize_from_start(config, deadline, rng)
            if score > best_score:
                best_score = score
                best_points = cand_pts
        except Exception:
            continue
    
    # === PHASE 3: Polish best result with local search ===
    if best_points is not None and time.time() < deadline - 15:
        cand_pts, score = _polish_with_local_search(best_points, deadline, rng)
        if score > best_score:
            best_score = score
            best_points = cand_pts
    
    # === PHASE 4: Perturbation restarts from best ===
    if best_points is not None and time.time() < deadline - 12:
        n_restarts = 0
        while time.time() < deadline - 8:
            n_restarts += 1
            frac = min(n_restarts / 15.0, 1.0)
            noise_scale = 0.1 * (1.0 - frac) + 0.003
            perturbed = best_points + noise_scale * rng.normal(0, 1, (N, 2))
            
            try:
                cand_pts, score = _polish_with_local_search(perturbed, deadline, rng)
                if score > best_score:
                    best_score = score
                    best_points = cand_pts
            except Exception:
                continue
    
    # Fallback
    if best_points is None:
        angles = 2 * np.pi * np.arange(N) / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points

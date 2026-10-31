import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

N = 13

# Pre-compute triangle indices for N=13
_TRI_INDICES = np.array(list(combinations(range(N), 3)))
_I_IDX = _TRI_INDICES[:, 0]
_J_IDX = _TRI_INDICES[:, 1]
_K_IDX = _TRI_INDICES[:, 2]
_NUM_TRI = len(_TRI_INDICES)

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

def _neg_score(params):
    """Objective function for minimization (maximizing score)."""
    points = params.reshape(N, 2)
    s = _score(points)
    return -s

# --- Support Line Parameterization ---
def _support_lines_to_points(params):
    """
    Convert support line parameters to points.
    params: array of shape (2*N,) where params[2*i] = angle, params[2*i+1] = offset
    The polygon is defined by intersection of lines: x*cos(a_i) + y*sin(a_i) = d_i
    We compute vertices by intersecting adjacent lines.
    """
    angles = params[0::2]
    offsets = params[1::2]
    
    # Sort by angle to ensure consistent ordering
    sort_idx = np.argsort(angles)
    angles = angles[sort_idx]
    offsets = offsets[sort_idx]
    
    # Compute vertices by intersecting consecutive lines
    points = np.zeros((N, 2))
    for i in range(N):
        j = (i + 1) % N
        a1, d1 = angles[i], offsets[i]
        a2, d2 = angles[j], offsets[j]
        
        # Line 1: x*cos(a1) + y*sin(a1) = d1
        # Line 2: x*cos(a2) + y*sin(a2) = d2
        # Solve linear system
        A = np.array([[np.cos(a1), np.sin(a1)],
                      [np.cos(a2), np.sin(a2)]])
        b = np.array([d1, d2])
        
        try:
            det = np.linalg.det(A)
            if abs(det) < 1e-12:
                # Lines are parallel or nearly parallel, use midpoint as fallback
                points[i] = [d1 * np.cos(a1), d1 * np.sin(a1)]
            else:
                points[i] = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            points[i] = [d1 * np.cos(a1), d1 * np.sin(a1)]
    
    return points

def _neg_support_score(params, alpha):
    """Objective for support-line parameterized optimization with softmin."""
    points = _support_lines_to_points(params)
    s = _softmin_score(points, alpha)
    return -s

def _neg_support_score_hard(params):
    """Objective for support-line parameterized optimization (hard score)."""
    points = _support_lines_to_points(params)
    s = _score(points)
    return -s

def _optimize_support_lines(deadline, rng):
    """
    Optimize using support line parameterization.
    This is a structural change: instead of optimizing point positions directly,
    we optimize the defining lines of the convex polygon.
    """
    best_score = 0.0
    best_points = None
    
    # Initialize with regular polygon support lines
    # For a regular N-gon with circumradius R, the inradius is R*cos(pi/N)
    R = 1.0
    inradius = R * np.cos(np.pi / N)
    
    # Angles for support lines (perpendicular to vertices of regular polygon)
    base_angles = 2 * np.pi * np.arange(N) / N
    base_offsets = np.full(N, inradius)
    
    # Create initial parameter arrays
    initial_params_list = []
    
    # Regular polygon
    p_reg = np.zeros(2 * N)
    p_reg[0::2] = base_angles
    p_reg[1::2] = base_offsets
    initial_params_list.append(p_reg)
    
    # Perturbed regular polygons
    for perturb in [0.05, 0.1, 0.2]:
        p = p_reg.copy()
        p[0::2] += rng.uniform(-perturb, perturb, N)
        p[1::2] *= (1.0 + rng.uniform(-perturb, perturb, N))
        initial_params_list.append(p)
    
    # Random offsets
    for _ in range(5):
        p = p_reg.copy()
        p[1::2] *= (1.0 + rng.uniform(-0.3, 0.3, N))
        initial_params_list.append(p)
    
    # Optimize each initial parameter set
    for init_params in initial_params_list:
        if time.time() > deadline - 15:
            break
        
        current_params = init_params.copy()
        current_points = _support_lines_to_points(current_params)
        current_score = _score(current_points)
        
        # Stage 1: Softmin optimization with various alphas
        for alpha in [50, 20, 10, 5, 2]:
            if time.time() > deadline - 10:
                break
            try:
                res = minimize(
                    _neg_support_score,
                    current_params,
                    args=(alpha,),
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True}
                )
                cand_params = res.x
                cand_points = _support_lines_to_points(cand_params)
                cand_score = _score(cand_points)
                if cand_score > current_score:
                    current_score = cand_score
                    current_params = cand_params
                    current_points = cand_points
            except Exception:
                pass
        
        # Stage 2: L-BFGS-B polish
        if time.time() < deadline - 8:
            try:
                res = minimize(
                    _neg_support_score,
                    current_params,
                    args=(5.0,),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                cand_params = res.x
                cand_points = _support_lines_to_points(cand_params)
                cand_score = _score(cand_points)
                if cand_score > current_score:
                    current_score = cand_score
                    current_params = cand_params
                    current_points = cand_points
            except Exception:
                pass
        
        # Stage 3: Hard score Nelder-Mead
        if time.time() < deadline - 5:
            try:
                res = minimize(
                    _neg_support_score_hard,
                    current_params,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                )
                cand_params = res.x
                cand_points = _support_lines_to_points(cand_params)
                cand_score = _score(cand_points)
                if cand_score > current_score:
                    current_score = cand_score
                    current_params = cand_params
                    current_points = cand_points
            except Exception:
                pass
        
        # Update global best
        if current_score > best_score:
            best_score = current_score
            best_points = current_points.copy()
    
    return best_points, best_score

# --- Fourier Parameterization (from inspiration) ---
def _fourier_to_points(params):
    """
    Convert Fourier parameters to 13 points on a perturbed regular polygon.
    params: [r0, a1, b1, a2, b2, cx, cy] (7 params)
    """
    r0 = params[0]
    a1, b1 = params[1], params[2]
    a2, b2 = params[3], params[4]
    cx, cy = params[5], params[6]
    
    angles = 2 * np.pi * np.arange(N) / N
    radii = r0 + a1 * np.cos(angles) + b1 * np.sin(angles) + a2 * np.cos(2 * angles) + b2 * np.sin(2 * angles)
    radii = np.maximum(radii, 0.1)
    
    points = np.column_stack([radii * np.cos(angles) + cx, radii * np.sin(angles) + cy])
    return points

def _neg_fourier_score(params, alpha):
    """Objective for Fourier-parameterized optimization."""
    points = _fourier_to_points(params)
    s = _softmin_score(points, alpha)
    return -s

def _neg_fourier_score_hard(params):
    """Objective for Fourier-parameterized optimization (hard score)."""
    points = _fourier_to_points(params)
    s = _score(points)
    return -s

def _optimize_fourier(deadline, rng):
    """Optimize using Fourier parameterization."""
    best_score = 0.0
    best_params = None
    best_points = None
    
    initial_params_list = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0],
        [0.9, 0.03, 0.02, 0.02, 0.01, 0.0, 0.0],
        [1.1, -0.02, 0.03, 0.01, -0.02, 0.0, 0.0],
    ]
    
    for _ in range(5):
        p = [1.0 + rng.normal(0, 0.1)]
        for _ in range(4):
            p.append(rng.normal(0, 0.05))
        p.extend([rng.normal(0, 0.05), rng.normal(0, 0.05)])
        initial_params_list.append(p)
    
    for init_params in initial_params_list:
        if time.time() > deadline - 15:
            break
        
        init_params = np.array(init_params, dtype=float)
        current_params = init_params.copy()
        current_score = _score(_fourier_to_points(current_params))
        
        for alpha in [50, 20, 10, 5]:
            if time.time() > deadline - 10:
                break
            try:
                res = minimize(
                    _neg_fourier_score,
                    current_params,
                    args=(alpha,),
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True}
                )
                cand_params = res.x
                cand_score = _score(_fourier_to_points(cand_params))
                if cand_score > current_score:
                    current_score = cand_score
                    current_params = cand_params
            except Exception:
                pass
        
        if time.time() < deadline - 8:
            try:
                res = minimize(
                    _neg_fourier_score,
                    current_params,
                    args=(5.0,),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                cand_params = res.x
                cand_score = _score(_fourier_to_points(cand_params))
                if cand_score > current_score:
                    current_score = cand_score
                    current_params = cand_params
            except Exception:
                pass
        
        if time.time() < deadline - 5:
            try:
                res = minimize(
                    _neg_fourier_score_hard,
                    current_params,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                )
                cand_params = res.x
                cand_score = _score(_fourier_to_points(cand_params))
                if cand_score > current_score:
                    current_score = cand_score
                    current_params = cand_params
            except Exception:
                pass
        
        if current_score > best_score:
            best_score = current_score
            best_params = current_params.copy()
            best_points = _fourier_to_points(current_params)
    
    return best_points, best_score

# --- Refinement machinery from current program ---
def _optimize_from_start(start_pts, deadline, rng):
    """Multi-stage optimization from a starting point."""
    best_pts = start_pts.copy()
    best_score = _score(start_pts)
    
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

def _coordinate_local_search(points, deadline, rng, initial_step=0.05, min_step=1e-6, max_iter=200):
    """Coordinate-wise pattern search on the hard score."""
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
            step *= 1.2
        else:
            step *= 0.5
    
    return best_pts, best_score

def _polish_with_local_search(points, deadline, rng):
    """Multi-stage polishing."""
    best_pts = points.copy()
    best_score = _score(best_pts)
    
    if time.time() < deadline - 10:
        cand_pts, score = _coordinate_local_search(best_pts, deadline, rng)
        if score > best_score:
            best_score = score
            best_pts = cand_pts
    
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
    
    if time.time() < deadline - 5:
        cand_pts, score = _coordinate_local_search(best_pts, deadline, rng, initial_step=0.01)
        if score > best_score:
            best_score = score
            best_pts = cand_pts
    
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
    deadline = start_time + 110.0
    
    best_score = 0.0
    best_points = None
    rng = np.random.RandomState(42)
    
    # === PHASE 1: Support Line Parameterization (structural change) ===
    if time.time() < deadline - 40:
        try:
            sl_points, sl_score = _optimize_support_lines(deadline, rng)
            if sl_score > best_score:
                best_score = sl_score
                best_points = sl_points
        except Exception:
            pass
    
    # === PHASE 2: Fourier Parameterization ===
    if time.time() < deadline - 30:
        try:
            f_points, f_score = _optimize_fourier(deadline, rng)
            if f_score > best_score:
                best_score = f_score
                best_points = f_points
        except Exception:
            pass
    
    # === PHASE 3: Diverse initial configs + traditional optimization ===
    initial_configs = []
    
    angles = 2 * np.pi * np.arange(N) / N
    initial_configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    for perturb in [0.02, 0.05, 0.1]:
        angles_p = 2 * np.pi * np.arange(N) / N + rng.uniform(-perturb, perturb, N)
        radii_p = 1.0 + rng.uniform(-perturb, perturb, N)
        initial_configs.append(np.column_stack([radii_p * np.cos(angles_p), radii_p * np.sin(angles_p)]))
    
    for amp in [0.1, 0.2, 0.3]:
        for freq in [2, 3, 5]:
            angles = 2 * np.pi * np.arange(N) / N
            radii = 1.0 + amp * np.sin(freq * angles)
            initial_configs.append(np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]))
    
    for _ in range(5):
        initial_configs.append(rng.uniform(-1, 1, (N, 2)))
    
    # Add best from previous phases
    if best_points is not None:
        initial_configs.append(best_points)
    
    scored_configs = []
    for config in initial_configs:
        try:
            s = _score(config)
            scored_configs.append((s, config))
        except Exception:
            pass
    scored_configs.sort(key=lambda x: -x[0])
    
    for s_init, config in scored_configs[:5]:
        if time.time() > deadline - 20:
            break
        
        try:
            cand_pts, score = _optimize_from_start(config, deadline, rng)
            if score > best_score:
                best_score = score
                best_points = cand_pts
        except Exception:
            continue
    
    # === PHASE 4: Polish best result ===
    if best_points is not None and time.time() < deadline - 15:
        cand_pts, score = _polish_with_local_search(best_points, deadline, rng)
        if score > best_score:
            best_score = score
            best_points = cand_pts
    
    # === PHASE 5: Perturbation restarts ===
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

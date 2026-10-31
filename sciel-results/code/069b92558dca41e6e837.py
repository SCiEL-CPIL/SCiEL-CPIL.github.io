import time
import numpy as np
from scipy.optimize import minimize
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
    Smooth approximation of the score using adaptive softmin of triangle areas.
    Uses log-sum-exp for numerical stability.
    """
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    
    areas = _get_triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    
    # Adaptive alpha: scale based on the spread of areas
    a_min = np.min(areas)
    a_max = np.max(areas)
    spread = a_max - a_min
    if spread < 1e-10:
        return a_min / hull_area
    
    # Adaptive alpha: larger when spread is small (sharper), smaller when spread is large
    adaptive_alpha = alpha * (1.0 + spread)
    
    # Numerically stable softmin: a_min - log(sum(exp(-alpha*(areas - a_min)))) / alpha
    shifted = -adaptive_alpha * (areas - a_min)
    max_shifted = np.max(shifted)
    if max_shifted > 500:
        max_shifted = 500
    exp_terms = np.exp(shifted - max_shifted)
    log_sum = np.log(np.sum(exp_terms)) + max_shifted
    softmin_val = a_min - log_sum / adaptive_alpha
    
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

def _fourier_to_points(params):
    """
    Convert Fourier parameters to 13 points on a perturbed regular polygon.
    params: [r0, a1, b1, a2, b2, cx, cy] (7 params)
    - r0: base radius
    - a1, b1: first harmonic amplitudes
    - a2, b2: second harmonic amplitudes
    - cx, cy: center offset
    """
    r0 = params[0]
    a1, b1 = params[1], params[2]
    a2, b2 = params[3], params[4]
    cx, cy = params[5], params[6]
    
    angles = 2 * np.pi * np.arange(N) / N
    # Perturbed radius
    radii = r0 + a1 * np.cos(angles) + b1 * np.sin(angles) + a2 * np.cos(2 * angles) + b2 * np.sin(2 * angles)
    
    # Ensure positive radii
    radii = np.maximum(radii, 0.1)
    
    points = np.column_stack([radii * np.cos(angles) + cx, radii * np.sin(angles) + cy])
    return points

def _neg_fourier_score(params, alpha):
    """
    Objective for Fourier-parameterized optimization.
    Uses softmin for smoothness.
    """
    points = _fourier_to_points(params)
    s = _softmin_score(points, alpha)
    return -s

def _neg_fourier_score_hard(params):
    """
    Objective for Fourier-parameterized optimization (hard score).
    """
    points = _fourier_to_points(params)
    s = _score(points)
    return -s

def _optimize_fourier(deadline, rng):
    """
    Optimize using Fourier parameterization.
    This reduces dimensionality from 26 to 7 and enforces smooth convexity.
    """
    best_score = 0.0
    best_params = None
    best_points = None
    
    # Initial parameter guesses
    initial_params_list = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Regular polygon
        [1.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0],
        [0.9, 0.03, 0.02, 0.02, 0.01, 0.0, 0.0],
        [1.1, -0.02, 0.03, 0.01, -0.02, 0.0, 0.0],
    ]
    
    # Add random initializations
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
        
        # Multi-stage optimization in Fourier space
        current_params = init_params.copy()
        current_score = _score(_fourier_to_points(current_params))
        
        # Stage 1: Softmin with various alphas
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
        
        # Stage 2: L-BFGS-B polish
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
        
        # Stage 3: Hard score Nelder-Mead
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
        
        # Update global best
        if current_score > best_score:
            best_score = current_score
            best_params = current_params.copy()
            best_points = _fourier_to_points(current_params)
    
    return best_points, best_score

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
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
            )
            cand_pts = res.x.reshape(N, 2)
            score = _score(cand_pts)
            if score > best_score:
                best_score = score
                best_pts = cand_pts
        except Exception:
            pass
    
    # Stage 2: L-BFGS-B polish with softmin
    if time.time() < deadline - 15:
        try:
            res = minimize(
                _neg_softmin_score,
                best_pts.flatten(),
                args=(2.0,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
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
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            cand_pts = res.x.reshape(N, 2)
            score = _score(cand_pts)
            if score > best_score:
                best_score = score
                best_pts = cand_pts
        except Exception:
            pass
    
    return best_pts, best_score

def _polish_with_local_search(points, deadline):
    """
    Final high-precision polish: coordinate descent on the hard objective.
    Uses adaptive step sizes and multiple passes with decreasing step sizes.
    """
    best_pts = points.copy()
    best_score = _score(best_pts)
    
    # Adaptive step sizes: start large, decrease over iterations
    base_steps = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    
    for step_idx, base_step in enumerate(base_steps):
        if time.time() > deadline - 5:
            break
        
        # Number of iterations for this step size
        n_iters = max(5, 30 - step_idx * 3)
        
        for outer_iter in range(n_iters):
            if time.time() > deadline - 5:
                break
            
            improved = False
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                
                orig = best_pts[i].copy()
                best_local = orig
                best_local_score = best_score
                
                # Generate perturbation directions
                directions = [
                    [base_step, 0], [-base_step, 0],
                    [0, base_step], [0, -base_step],
                    [base_step * 0.707, base_step * 0.707],
                    [-base_step * 0.707, base_step * 0.707],
                    [base_step * 0.707, -base_step * 0.707],
                    [-base_step * 0.707, -base_step * 0.707],
                    [base_step * 0.5, 0], [-base_step * 0.5, 0],
                    [0, base_step * 0.5], [0, -base_step * 0.5],
                ]
                
                for dx, dy in directions:
                    cand = best_pts.copy()
                    cand[i] = orig + np.array([dx, dy])
                    s = _score(cand)
                    if s > best_local_score + 1e-12:
                        best_local_score = s
                        best_local = orig + np.array([dx, dy])
                
                if not np.allclose(best_local, orig, atol=1e-10):
                    best_pts[i] = best_local
                    improved = True
            
            if not improved:
                break
            
            current_score = _score(best_pts)
            if current_score > best_score:
                best_score = current_score
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s kill
    
    best_score = 0.0
    best_points = None
    rng = np.random.RandomState(42)
    
    # Phase 1: Fourier-parameterized optimization (reduced dimensionality)
    # Allocate ~35s for this phase
    fourier_deadline = start_time + 35.0
    if time.time() < fourier_deadline:
        try:
            fourier_pts, fourier_score = _optimize_fourier(fourier_deadline, rng)
            if fourier_pts is not None and fourier_score > best_score:
                best_score = fourier_score
                best_points = fourier_pts
        except Exception:
            pass
    
    # Phase 2: Generate diverse initial configurations and optimize
    # Allocate ~55s for this phase (from 35s to 90s)
    phase2_deadline = start_time + 90.0
    
    initial_configs = []
    
    # Regular polygon
    angles = 2 * np.pi * np.arange(N) / N
    initial_configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # If we have a good Fourier result, use it as an initial config
    if best_points is not None:
        initial_configs.append(best_points.copy())
    
    # Ring configurations with various parameters
    for radius in [1.0, 1.2, 0.8]:
        for phase in [0.0, np.pi / N, np.pi / (2 * N)]:
            initial_configs.append(_make_ring_config(N, radius, phase_offset=phase))
    
    # Two-ring structures
    for r1, r2 in [(0.5, 1.0), (0.4, 1.1), (0.6, 0.9)]:
        for phase in [0.0, np.pi / N]:
            initial_configs.append(_make_ring_config(N, 1.0, phase_offset=phase, n_rings=2, ring_radii=[r1, r2]))
    
    # Radial perturbation of regular polygon
    for amp in [0.1, 0.2, 0.3]:
        for freq in [2, 3, 5]:
            angles = 2 * np.pi * np.arange(N) / N
            radii = 1.0 + amp * np.sin(freq * angles)
            initial_configs.append(np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]))
    
    # Random configurations
    for _ in range(5):
        initial_configs.append(rng.uniform(-1, 1, (N, 2)))
    
    # Try each initial configuration with multi-stage optimization
    # Use a tighter deadline for each individual optimization to allow more configs
    for config in initial_configs:
        if time.time() > phase2_deadline - 15:
            break
        
        # Give each config up to 8 seconds
        config_deadline = min(time.time() + 8.0, phase2_deadline - 5)
        try:
            cand_pts, score = _optimize_from_start(config, config_deadline, rng)
            if score > best_score:
                best_score = score
                best_points = cand_pts
        except Exception:
            continue
    
    # Phase 3: Perturbation-based restarts around the best solution
    # Allocate ~15s for this phase (from 90s to 105s)
    phase3_deadline = start_time + 105.0
    if best_points is not None and time.time() < phase3_deadline:
        for i in range(8):
            if time.time() > phase3_deadline - 5:
                break
            
            # Perturb the best solution with decreasing noise
            noise_scale = 0.08 * (1 - i / 8.0) + 0.005
            perturbed = best_points + noise_scale * rng.normal(0, 1, (N, 2))
            
            # Give each restart up to 5 seconds
            restart_deadline = min(time.time() + 5.0, phase3_deadline - 3)
            try:
                cand_pts, score = _optimize_from_start(perturbed, restart_deadline, rng)
                if score > best_score:
                    best_score = score
                    best_points = cand_pts
            except Exception:
                continue
    
    # Phase 4: Final high-precision polish using coordinate descent
    # Allocate ~10s for this phase (from 105s to 115s)
    if best_points is not None and time.time() < deadline - 3:
        try:
            polished_pts, polished_score = _polish_with_local_search(best_points, deadline)
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished_pts
        except Exception:
            pass
    
    # Fallback to regular polygon if everything failed
    if best_points is None:
        angles = 2 * np.pi * np.arange(N) / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points

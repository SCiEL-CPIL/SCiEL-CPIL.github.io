import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

N = 13
_NUM_TRIPLES = len(list(combinations(range(N), 3)))

def _triangle_areas_vectorized(points):
    """
    Vectorized calculation of areas for all triples.
    points: (N, 2) array
    returns: array of shape (num_triangles,)
    """
    n = len(points)
    if n < 3:
        return np.array([])
    
    i_idx, j_idx, k_idx = np.indices((n, n, n)).reshape(3, -1)
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx, j_idx, k_idx = i_idx[mask], j_idx[mask], k_idx[mask]
    
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
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
    areas = _triangle_areas_vectorized(points)
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
    
    areas = _triangle_areas_vectorized(points)
    if len(areas) == 0:
        return 0.0
    
    # Softmin: -1/alpha * log(sum(exp(-alpha * areas)))
    # This is a smooth approximation of min(areas)
    # Subtract min for numerical stability
    a_min = np.min(areas)
    shifted = areas - a_min
    exp_terms = np.exp(-alpha * shifted)
    log_sum = np.log(np.sum(exp_terms))
    softmin_val = a_min - log_sum / alpha
    
    return softmin_val / hull_area

def _neg_softmin_score(params, alpha):
    """Objective for softmin-based optimization."""
    points = params.reshape(N, 2)
    s = _softmin_score(points, alpha)
    return -s

def _coordinate_descent_polish(points, deadline, rng):
    """
    Coordinate descent: optimize one point at a time while keeping others fixed.
    This can escape local minima that Nelder-Mead gets stuck in.
    """
    best_pts = points.copy()
    best_score = _score(best_pts)
    
    for iteration in range(50):
        if time.time() > deadline - 5:
            break
        
        improved = False
        for i in range(N):
            if time.time() > deadline - 5:
                break
            
            # Save current position
            saved_pt = best_pts[i].copy()
            
            # Try to optimize point i by moving it
            # Use small perturbations in random directions
            for attempt in range(20):
                if time.time() > deadline - 5:
                    break
                
                # Random perturbation
                direction = rng.normal(0, 1, 2)
                direction /= (np.linalg.norm(direction) + 1e-10)
                
                # Try different step sizes
                for step_size in [0.05, 0.02, 0.01, 0.005]:
                    candidate = best_pts.copy()
                    candidate[i] = saved_pt + step_size * direction
                    score = _score(candidate)
                    
                    if score > best_score + 1e-10:
                        best_pts = candidate
                        best_score = score
                        improved = True
                        break
                    elif score > best_score:
                        # Accept if slightly better (for noise tolerance)
                        best_pts = candidate
                        best_score = score
                        improved = True
                        break
                
                if improved:
                    break
            
            # If we moved point i, re-save its new position
            if improved:
                saved_pt = best_pts[i].copy()
        
        if not improved:
            break
    
    return best_pts, best_score

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
    2. Coordinate descent polish
    3. Nelder-Mead on hard score with tighter tolerances
    Returns best points and score found.
    """
    best_pts = start_pts.copy()
    best_score = _score(start_pts)
    
    # Stage 1: Softmin optimization with annealing alpha
    alphas = [200, 100, 50, 20, 10, 5, 2]
    for alpha in alphas:
        if time.time() > deadline - 15:
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
    
    # Stage 2: Coordinate descent polish
    if time.time() < deadline - 20:
        try:
            cand_pts, score = _coordinate_descent_polish(best_pts, deadline, rng)
            if score > best_score:
                best_score = score
                best_pts = cand_pts
        except Exception:
            pass
    
    # Stage 3: Nelder-Mead on hard score with high precision
    if time.time() < deadline - 5:
        try:
            res = minimize(
                _neg_score,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            cand_pts = res.x.reshape(N, 2)
            score = _score(cand_pts)
            if score > best_score:
                best_score = score
                best_pts = cand_pts
        except Exception:
            pass
    
    # Stage 4: Final coordinate descent for extra polish
    if time.time() < deadline - 3:
        try:
            cand_pts, score = _coordinate_descent_polish(best_pts, deadline, rng)
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

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s kill
    
    best_score = 0.0
    best_points = None
    rng = np.random.RandomState(42)
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # Regular polygon
    angles = 2 * np.pi * np.arange(N) / N
    initial_configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # Ring configurations with various parameters
    for radius in [1.0, 1.2, 0.8, 1.1, 0.9]:
        for phase in [0.0, np.pi / N, np.pi / (2 * N), np.pi / (3 * N)]:
            initial_configs.append(_make_ring_config(N, radius, phase_offset=phase))
    
    # Two-ring structures
    for r1, r2 in [(0.5, 1.0), (0.4, 1.1), (0.6, 0.9), (0.3, 1.2), (0.7, 0.8)]:
        for phase in [0.0, np.pi / N, np.pi / (2 * N)]:
            initial_configs.append(_make_ring_config(N, 1.0, phase_offset=phase, n_rings=2, ring_radii=[r1, r2]))
    
    # Three-ring structures
    for r1, r2, r3 in [(0.3, 0.6, 1.0), (0.4, 0.7, 0.9), (0.2, 0.5, 1.1)]:
        for phase in [0.0, np.pi / N]:
            initial_configs.append(_make_ring_config(N, 1.0, phase_offset=phase, n_rings=3, ring_radii=[r1, r2, r3]))
    
    # Radial perturbation of regular polygon
    for amp in [0.1, 0.2, 0.3, 0.4]:
        for freq in [2, 3, 4, 5, 6]:
            angles = 2 * np.pi * np.arange(N) / N
            radii = 1.0 + amp * np.sin(freq * angles)
            initial_configs.append(np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]))
    
    # Random configurations
    for _ in range(15):
        initial_configs.append(rng.uniform(-1, 1, (N, 2)))
    
    # Sort configs by initial score (try best ones first)
    config_scores = []
    for config in initial_configs:
        try:
            s = _score(config)
            config_scores.append((s, config))
        except Exception:
            config_scores.append((0.0, config))
    
    config_scores.sort(key=lambda x: -x[0])
    
    # Try each initial configuration with multi-stage optimization
    for score, config in config_scores:
        if time.time() > deadline - 25:
            break
        
        # Skip if this config is much worse than current best
        if best_score > 0 and score < best_score * 0.5:
            continue
        
        try:
            cand_pts, cand_score = _optimize_from_start(config, deadline, rng)
            if cand_score > best_score:
                best_score = cand_score
                best_points = cand_pts
        except Exception:
            continue
    
    # If we have a good candidate, do perturbation-based restarts
    if best_points is not None and time.time() < deadline - 40:
        for i in range(20):
            if time.time() > deadline - 10:
                break
            
            # Perturb the best solution with decreasing noise
            noise_scale = 0.15 * (1 - i / 20.0) + 0.005
            perturbed = best_points + noise_scale * rng.normal(0, 1, (N, 2))
            
            try:
                cand_pts, cand_score = _optimize_from_start(perturbed, deadline, rng)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = cand_pts
            except Exception:
                continue
    
    # Final polish on the best solution
    if best_points is not None and time.time() < deadline - 5:
        try:
            cand_pts, cand_score = _coordinate_descent_polish(best_points, deadline, rng)
            if cand_score > best_score:
                best_score = cand_score
                best_points = cand_pts
        except Exception:
            pass
    
    # Fallback to regular polygon if everything failed
    if best_points is None:
        angles = 2 * np.pi * np.arange(N) / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points

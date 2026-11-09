import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 14

# Pre-compute indices for all triples
_IDX_I = []
_IDX_J = []
_IDX_K = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _IDX_I.append(i)
            _IDX_J.append(j)
            _IDX_K.append(k)
_IDX_I = np.array(_IDX_I)
_IDX_J = np.array(_IDX_J)
_IDX_K = np.array(_IDX_K)

def _score(points):
    """Calculate the score: min triangle area / convex hull area."""
    try:
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            return 0.0
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
        except Exception:
            return 0.0
        if hull_area < 1e-10:
            return 0.0
        pi = points[_IDX_I]
        pj = points[_IDX_J]
        pk = points[_IDX_K]
        v1 = pj - pi
        v2 = pk - pi
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        min_area = np.min(areas)
        return min_area / hull_area
    except Exception:
        return 0.0

def _ring_config_to_points(params, n_inner, n_outer):
    """
    Convert ring parameters to points.
    params: [r_inner, r_outer, offset_inner, offset_outer]
    n_inner: number of points on inner ring (0 = center point)
    n_outer: number of points on outer ring
    """
    r_inner = max(0.01, params[0])
    r_outer = max(r_inner + 0.05, params[1])
    off_inner = params[2]
    off_outer = params[3]
    
    points = []
    
    if n_inner == 1:
        # Single center point
        points.append([0.0, 0.0])
    elif n_inner > 1:
        # Inner ring
        angles_in = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + off_inner
        x_in = r_inner * np.cos(angles_in)
        y_in = r_inner * np.sin(angles_in)
        points.append(np.column_stack([x_in, y_in]))
    
    # Outer ring
    angles_out = np.linspace(0, 2 * np.pi, n_outer, endpoint=False) + off_outer
    x_out = r_outer * np.cos(angles_out)
    y_out = r_outer * np.sin(angles_out)
    points.append(np.column_stack([x_out, y_out]))
    
    return np.vstack(points)

def _ring_objective(params, n_inner, n_outer):
    """Objective for ring parameterization."""
    try:
        if np.any(np.isnan(params)) or np.any(np.isinf(params)):
            return 1e6
        points = _ring_config_to_points(params, n_inner, n_outer)
        if len(points) != N:
            return 1e6
        s = _score(points)
        if s <= 1e-15:
            return 1e6
        return -s
    except Exception:
        return 1e6

def _direct_objective(params):
    """Objective for direct 2D coordinate optimization."""
    try:
        if np.any(np.isnan(params)) or np.any(np.isinf(params)):
            return 1e6
        points = params.reshape(N, 2)
        s = _score(points)
        if s <= 1e-15:
            return 1e6
        return -s
    except Exception:
        return 1e6

def _generate_ring_guesses():
    """Generate diverse ring configuration guesses."""
    rng = np.random.default_rng(42)
    guesses = []
    
    # Ring topologies: (n_inner, n_outer)
    topologies = [
        (1, 13),   # 1 center + 13 outer
        (2, 12),   # 2 inner + 12 outer
        (3, 11),   # 3 inner + 11 outer
        (4, 10),   # 4 inner + 10 outer
        (5, 9),    # 5 inner + 9 outer
        (6, 8),    # 6 inner + 8 outer
        (7, 7),    # 7 inner + 7 outer
    ]
    
    for n_inner, n_outer in topologies:
        # Regular ring configuration
        if n_inner == 1:
            r_in = 0.0
        else:
            r_in = 0.4 + rng.uniform(-0.1, 0.1)
        r_out = 1.0 + rng.uniform(-0.1, 0.1)
        off_in = rng.uniform(0, 2 * np.pi)
        off_out = rng.uniform(0, 2 * np.pi)
        guesses.append((np.array([r_in, r_out, off_in, off_out]), n_inner, n_outer))
        
        # Perturbed regular
        for _ in range(3):
            if n_inner == 1:
                r_in = 0.0
            else:
                r_in = 0.3 + rng.uniform(-0.15, 0.25)
            r_out = 0.8 + rng.uniform(-0.1, 0.3)
            off_in = rng.uniform(0, 2 * np.pi)
            off_out = rng.uniform(0, 2 * np.pi)
            guesses.append((np.array([r_in, r_out, off_in, off_out]), n_inner, n_outer))
        
        # Optimized-looking ratios
        for ratio in [0.3, 0.5, 0.6, 0.7]:
            r_in = ratio
            r_out = 1.0
            off_in = rng.uniform(0, 2 * np.pi)
            off_out = rng.uniform(0, 2 * np.pi)
            guesses.append((np.array([r_in, r_out, off_in, off_out]), n_inner, n_outer))
    
    # Also add some 3-ring configurations by treating them as 2-ring with different params
    # (handled by the topology list above)
    
    return guesses

def _coordinate_ascent_polish(points, time_limit):
    """Fine-tune with coordinate ascent."""
    pts = points.copy()
    current_score = _score(pts)
    step_sizes = [0.02, 0.005, 0.001, 0.0002]
    
    for ss in step_sizes:
        improved = True
        while improved and (time.time() < time_limit):
            improved = False
            for i in range(N):
                if time.time() >= time_limit:
                    break
                for dim in range(2):
                    for delta in [ss, -ss]:
                        trial = pts.copy()
                        trial[i, dim] += delta
                        s = _score(trial)
                        if s > current_score + 1e-12:
                            pts = trial
                            current_score = s
                            improved = True
    return pts, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    best_points = None
    best_score = -np.inf
    
    # Phase 1: Ring-based search
    ring_guesses = _generate_ring_guesses()
    scored_ring_guesses = []
    
    for params, n_inner, n_outer in ring_guesses:
        if time.time() > deadline - 30:
            break
        points = _ring_config_to_points(params, n_inner, n_outer)
        s = _score(points)
        scored_ring_guesses.append((s, params, n_inner, n_outer))
        if s > best_score:
            best_score = s
            best_points = points.copy()
    
    scored_ring_guesses.sort(key=lambda x: x[0], reverse=True)
    
    # Refine top ring candidates
    for idx in range(min(15, len(scored_ring_guesses))):
        if time.time() > deadline - 20:
            break
        
        s_init, params, n_inner, n_outer = scored_ring_guesses[idx]
        
        try:
            res = minimize(
                _ring_objective,
                params,
                args=(n_inner, n_outer),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            s = -res.fun
            if s > best_score:
                best_score = s
                best_points = _ring_config_to_points(res.x, n_inner, n_outer)
        except Exception:
            pass
    
    # Phase 2: Direct 2D refinement on best ring result
    if best_points is not None and time.time() < deadline - 15:
        try:
            res = minimize(
                _direct_objective,
                best_points.flatten(),
                method='Powell',
                options={'maxiter': 500, 'ftol': 1e-12, 'xtol': 1e-12}
            )
            s = -res.fun
            if s > best_score:
                best_score = s
                best_points = res.x.reshape(N, 2)
        except Exception:
            pass
    
    # Phase 3: Try direct optimization from regular polygon as well
    if time.time() < deadline - 10:
        angles_reg = np.linspace(0, 2 * np.pi, N, endpoint=False)
        poly_pts = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])
        s_poly = _score(poly_pts)
        
        if s_poly > best_score:
            best_score = s_poly
            best_points = poly_pts.copy()
        
        # Nelder-Mead on polygon
        try:
            res = minimize(
                _direct_objective,
                poly_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            s = -res.fun
            if s > best_score:
                best_score = s
                best_points = res.x.reshape(N, 2)
        except Exception:
            pass
    
    # Phase 4: Coordinate ascent polish
    if best_points is not None and time.time() < deadline - 3:
        best_points, best_score = _coordinate_ascent_polish(best_points, deadline - 2)
    
    # Fallback
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points

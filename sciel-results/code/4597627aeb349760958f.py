import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14

def _get_triangle_indices(n):
    """Pre-compute all (i,j,k) with i<j<k"""
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=np.int32)

_TRI_INDICES = _get_triangle_indices(N)
_I = _TRI_INDICES[:, 0]
_J = _TRI_INDICES[:, 1]
_K = _TRI_INDICES[:, 2]

def _min_triangle_area(points):
    """Compute minimum triangle area for given points."""
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross).min()

def _soft_min_triangle_area(points, beta=100.0):
    """
    Smooth surrogate for min triangle area using log-sum-exp.
    Returns a smooth approximation of the minimum area.
    """
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    # Soft min via log-sum-exp
    max_a = np.max(areas)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - max_a)))) / beta
    soft_min = max_a - log_sum
    return soft_min

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = scipy.spatial.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Score = min triangle area / convex hull area."""
    try:
        if len(points) < 3:
            return 0.0
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        mta = _min_triangle_area(points)
        return mta / ha
    except Exception:
        return 0.0

def _neg_score(points_flat):
    """Negative score for minimization."""
    points = points_flat.reshape(-1, 2)
    return -_score(points)

def _neg_soft_score(points_flat, beta=100.0):
    """Negative soft-score for gradient-based optimization."""
    points = points_flat.reshape(-1, 2)
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        smta = _soft_min_triangle_area(points, beta)
        return -(smta / ha)
    except Exception:
        return 0.0

def _normalize(points):
    """Scale so convex hull area = 1."""
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return points
        return points / np.sqrt(ha)
    except Exception:
        return points

def _regular_polygon(n, radius=1.0, rotation=0.0):
    """Regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _generate_parameterized_points(r, delta):
    """
    Generate 14 points: 7 on outer circle (radius 1) and 7 on inner circle (radius r).
    Outer points: angles 0, 2pi/7, ...
    Inner points: angles delta, delta + 2pi/7, ...
    """
    n_outer = 7
    n_inner = 7
    
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    points_outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    
    angles_inner = np.linspace(delta, delta + 2 * np.pi, n_inner, endpoint=False)
    points_inner = np.column_stack([r * np.cos(angles_inner), r * np.sin(angles_inner)])
    
    return np.vstack([points_outer, points_inner])

def _score_parameterized(r, delta):
    """Score the parameterized configuration."""
    try:
        if r <= 0:
            return 0.0
        pts = _generate_parameterized_points(r, delta)
        pts_norm = _normalize(pts)
        return _score(pts_norm)
    except Exception:
        return 0.0

def _neg_score_parameterized(params):
    """Negative score for parameterized optimization."""
    r, delta = params
    if r <= 0:
        r = 1e-6
    return -_score_parameterized(r, delta)

def _generate_flexible_ring_points(r_inner, delta_inner, angle_jitter_outer, angle_jitter_inner):
    """
    Generate 14 points with flexible angular positions on two rings.
    Outer ring: radius 1, base angles 0, 2pi/7, ... + jitter
    Inner ring: radius r_inner, base angles delta, delta+2pi/7, ... + jitter
    """
    n_ring = 7
    base_angles = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    
    outer_angles = base_angles + angle_jitter_outer
    inner_angles = base_angles + delta_inner + angle_jitter_inner
    
    points_outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    points_inner = np.column_stack([r_inner * np.cos(inner_angles), r_inner * np.sin(inner_angles)])
    
    return np.vstack([points_outer, points_inner])

def _neg_score_flexible(params):
    """
    Negative score for flexible ring parameterization.
    params: [r_inner, delta_inner, *outer_jitter(7), *inner_jitter(7)]
    Total: 16 parameters
    """
    try:
        r_inner = params[0]
        delta_inner = params[1]
        outer_jitter = params[2:9]
        inner_jitter = params[9:16]
        
        if r_inner <= 0:
            r_inner = 1e-6
        
        pts = _generate_flexible_ring_points(r_inner, delta_inner, outer_jitter, inner_jitter)
        pts_norm = _normalize(pts)
        return -_score(pts_norm)
    except Exception:
        return 0.0

def _neg_soft_score_flexible(params, beta=50.0):
    """
    Negative soft-score for flexible ring parameterization (for L-BFGS-B).
    """
    try:
        r_inner = params[0]
        delta_inner = params[1]
        outer_jitter = params[2:9]
        inner_jitter = params[9:16]
        
        if r_inner <= 0:
            r_inner = 1e-6
        
        pts = _generate_flexible_ring_points(r_inner, delta_inner, outer_jitter, inner_jitter)
        ha = _hull_area(pts)
        if ha < 1e-12:
            return 0.0
        smta = _soft_min_triangle_area(pts, beta)
        return -(smta / ha)
    except Exception:
        return 0.0

def _regular_14_gon():
    return _regular_polygon(14, 1.0, 0.0)

def _single_center_13_around():
    """1 point at center, 13 on a circle."""
    center = np.array([[0.0, 0.0]])
    outer = _regular_polygon(13, 1.0, 0.0)
    return np.vstack([center, outer])

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_score = 0.0
    best_points = None
    
    def _update_best(pts, sc):
        nonlocal best_score, best_points
        if sc > best_score:
            best_score = sc
            best_points = _normalize(pts).copy()

    def _time_left():
        return deadline - time.time()

    # 1. Quick initialization: Find a good starting point using the existing 2-param search
    # This provides a strong baseline (the "heptagon" structure)
    r_vals = np.linspace(0.4, 0.7, 10)
    delta_vals = np.linspace(0, 2 * np.pi / 7, 10, endpoint=False)
    
    best_init_score = 0.0
    best_init_pts = None
    
    for r in r_vals:
        for delta in delta_vals:
            pts = _generate_parameterized_points(r, delta)
            sc = _score(pts)
            if sc > best_init_score:
                best_init_score = sc
                best_init_pts = pts
    
    if best_init_pts is not None:
        _update_best(best_init_pts, best_init_score)
    else:
        _update_best(_regular_14_gon(), _score(_regular_14_gon()))

    # 2. Simulated Annealing on the full 28D coordinate space
    # This allows breaking the symmetry of the two-ring structure
    current_points = best_points.copy()
    current_score = best_score
    
    # SA Parameters
    T_start = 0.01
    T_end = 1e-6
    cooling_rate = 0.995
    T = T_start
    
    rng = np.random.RandomState(42)
    
    # Pre-calculate number of steps we can afford
    # Each step involves _score which is ~1-2ms. 150s / 2ms = 75000 steps.
    # We will run until time runs out, but keep a check.
    
    while _time_left() > 10:
        # Generate a neighbor
        # Step size is proportional to current temperature
        step_size = 0.05 * T / T_start + 0.001
        new_points = current_points + rng.normal(0, step_size, size=current_points.shape)
        
        new_score = _score(new_points)
        
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            _update_best(current_points, current_score)
        else:
            # Accept worse solution with probability exp(-(new_score - current_score) / T)
            # Note: we are maximizing, so delta = new - current < 0
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current_points = new_points
                    current_score = new_score
        
        T *= cooling_rate
        if T < T_end:
            # Reheat if we get stuck? Or just stop?
            # Reheating can help escape local minima
            if _time_left() > 30:
                T = T_start * 0.1
                # Random jump
                current_points = current_points + rng.normal(0, 0.1, size=current_points.shape)
                current_score = _score(current_points)
            else:
                break

    # 3. Final Local Optimization (Polish)
    # Use Nelder-Mead on the best point found by SA
    if best_points is not None and _time_left() > 5:
        try:
            # Reshape for Nelder-Mead
            x0 = best_points.flatten()
            res = scipy.optimize.minimize(
                _neg_score,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            pts_opt = _normalize(res.x.reshape(-1, 2))
            sc_opt = _score(pts_opt)
            _update_best(pts_opt, sc_opt)
        except Exception:
            pass

    # 4. If time permits, try a few random restarts of SA from different structures
    if _time_left() > 20:
        for _ in range(3):
            if _time_left() < 15:
                break
            # Random regular polygon with jitter
            pts_init = _regular_14_gon() + rng.normal(0, 0.1, size=(14, 2))
            pts_init = _normalize(pts_init)
            sc_init = _score(pts_init)
            
            if sc_init < best_score:
                # Only proceed if it's not terrible, or just run a short SA
                current_points = pts_init
                current_score = sc_init
                T = T_start * 0.5
                for _ in range(1000): # Short SA
                    if _time_left() < 5:
                        break
                    step_size = 0.05 * T / T_start + 0.001
                    new_points = current_points + rng.normal(0, step_size, size=current_points.shape)
                    new_score = _score(new_points)
                    if new_score > current_score:
                        current_points = new_points
                        current_score = new_score
                        _update_best(current_points, current_score)
                    else:
                        delta = new_score - current_score
                        if delta < 0 and rng.random() < np.exp(-delta / T):
                            current_points = new_points
                            current_score = new_score
                    T *= cooling_rate

    if best_points is None:
        best_points = _normalize(_regular_14_gon())
    else:
        best_points = _normalize(best_points)
        
    return best_points

import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools

N = 14
START_TIME = time.time()
DEADLINE = START_TIME + 115.0

def _get_triple_indices(n):
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=int)

TRIPLE_INDICES = _get_triple_indices(N)
I_IDX = TRIPLE_INDICES[:, 0]
J_IDX = TRIPLE_INDICES[:, 1]
K_IDX = TRIPLE_INDICES[:, 2]
NUM_TRIPLES = len(TRIPLE_INDICES)

def _calculate_score(points):
    """Compute min triangle area / convex hull area."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    if hull_area < 1e-12:
        return 0.0
    p_i = points[I_IDX]
    p_j = points[J_IDX]
    p_k = points[K_IDX]
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    min_area = np.min(areas)
    return min_area / hull_area

def _soft_min_score(points, beta=50.0):
    """Soft minimum of triangle areas / convex hull area."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    if hull_area < 1e-12:
        return 0.0
    p_i = points[I_IDX]
    p_j = points[J_IDX]
    p_k = points[K_IDX]
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    # Soft min: -log(sum(exp(-beta * areas))) / beta
    log_sum = np.log(np.sum(np.exp(-beta * areas)))
    soft_min = -log_sum / beta
    return soft_min / hull_area

def _superellipse_points(n, exponent=4.0, rotation=0.0, scale=1.0):
    """Generate n points on a superellipse x^(2/p) + y^(2/p) = 1."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False) + rotation
    # For superellipse, parametrize with angle t:
    # x = sign(cos(t)) * |cos(t)|^(2/p)
    # y = sign(sin(t)) * |sin(t)|^(2/p)
    cos_t = np.cos(angles)
    sin_t = np.sin(angles)
    x = np.sign(cos_t) * np.abs(cos_t) ** (2.0 / exponent)
    y = np.sign(sin_t) * np.abs(sin_t) ** (2.0 / exponent)
    return scale * np.column_stack([x, y])

def _score_ring_params(params):
    """Score for ring parameterization: [exponent, rotation, scale]."""
    exponent = np.clip(params[0], 2.0, 8.0)
    rotation = params[1]
    scale = np.clip(params[2], 0.5, 2.0)
    pts = _superellipse_points(N, exponent, rotation, scale)
    return -_calculate_score(pts)

def _soft_min_ring_score(params, beta=50.0):
    """Soft-min score for ring parameterization."""
    exponent = np.clip(params[0], 2.0, 8.0)
    rotation = params[1]
    scale = np.clip(params[2], 0.5, 2.0)
    pts = _superellipse_points(N, exponent, rotation, scale)
    return -_soft_min_score(pts, beta=beta)

def _optimize_ring(start_params, deadline, beta=50.0):
    """Optimize ring parameters with Nelder-Mead using soft-min."""
    try:
        res = minimize(
            _soft_min_ring_score,
            start_params,
            args=(beta,),
            method='Nelder-Mead',
            options={'maxiter': 300, 'maxfev': 600, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        if time.time() > deadline:
            return None
        # Polish with hard min
        exponent = np.clip(res.x[0], 2.0, 8.0)
        rotation = res.x[1]
        scale = np.clip(res.x[2], 0.5, 2.0)
        pts = _superellipse_points(N, exponent, rotation, scale)
        return pts
    except Exception:
        return None

def _get_points_2p(params):
    """Two-parameter ring: [exponent, rotation]."""
    exponent = np.clip(params[0], 2.0, 8.0)
    rotation = params[1]
    return _superellipse_points(N, exponent, rotation, 1.0)

def _score_2p_params(params):
    """Score for 2-param ring."""
    pts = _get_points_2p(params)
    return -_calculate_score(pts)

def _soft_min_2p_score(params, beta=50.0):
    """Soft-min score for 2-param ring."""
    pts = _get_points_2p(params)
    return -_soft_min_score(pts, beta=beta)

def _optimize_2p(start_params, deadline, beta=50.0):
    """Optimize 2-param ring."""
    try:
        res = minimize(
            _soft_min_2p_score,
            start_params,
            args=(beta,),
            method='Nelder-Mead',
            options={'maxiter': 200, 'maxfev': 400, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        if time.time() > deadline:
            return None
        return _get_points_2p(res.x)
    except Exception:
        return None

def _coordinate_polish(points, deadline, beta_start=50.0, beta_end=250.0, n_stages=3):
    """Polish points using L-BFGS-B with soft-min, then hard-min."""
    current = points.copy()
    
    for stage in range(n_stages):
        if time.time() > deadline:
            break
        # Interpolate beta
        t = stage / max(n_stages - 1, 1)
        beta = beta_start + (beta_end - beta_start) * t
        
        def _neg_soft(flat):
            pts = flat.reshape((N, 2))
            return -_soft_min_score(pts, beta=beta)
        
        try:
            res = minimize(
                _neg_soft,
                current.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 50, 'maxfun': 100}
            )
            if res.x is not None:
                current = res.x.reshape((N, 2))
        except Exception:
            pass
    
    # Final hard-min polish with Nelder-Mead
    if time.time() > deadline:
        return current
    
    def _neg_hard(flat):
        pts = flat.reshape((N, 2))
        return -_calculate_score(pts)
    
    try:
        res = minimize(
            _neg_hard,
            current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 150, 'maxfev': 300, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        if res.x is not None:
            current = res.x.reshape((N, 2))
    except Exception:
        pass
    
    return current

def construct():
    global START_TIME, DEADLINE
    START_TIME = time.time()
    DEADLINE = START_TIME + 115.0
    
    np.random.seed(42)
    
    best_score = -1
    best_points = None
    
    def _update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        s = _calculate_score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
    
    # === Phase 1: Superellipse Ring Optimization ===
    # Try multiple starting points with different exponents
    ring_starts = []
    # Regular polygon (circle)
    ring_starts.append(np.array([2.0, 0.0, 1.0]))
    # Square-like
    ring_starts.append(np.array([4.0, 0.0, 1.0]))
    # Very square-like
    ring_starts.append(np.array([6.0, 0.0, 1.0]))
    # Very square-like
    ring_starts.append(np.array([8.0, 0.0, 1.0]))
    # With rotation
    ring_starts.append(np.array([4.0, np.pi/14, 1.0]))
    ring_starts.append(np.array([6.0, np.pi/14, 1.0]))
    # Random exponents
    for _ in range(6):
        exp = np.random.uniform(2.5, 7.0)
        rot = np.random.uniform(0, 2*np.pi)
        scale = np.random.uniform(0.8, 1.2)
        ring_starts.append(np.array([exp, rot, scale]))
    
    for sp in ring_starts:
        if time.time() > DEADLINE - 30:
            break
        # Stage 1: soft-min with beta=50
        pts = _optimize_ring(sp, DEADLINE - 25, beta=50.0)
        _update_best(pts)
        
        # Stage 2: polish with higher beta
        if pts is not None and time.time() < DEADLINE - 20:
            pts2 = _optimize_ring(sp, DEADLINE - 15, beta=100.0)
            _update_best(pts2)
    
    # === Phase 2: 2-Param Ring (faster, more starts) ===
    for _ in range(10):
        if time.time() > DEADLINE - 25:
            break
        exp = np.random.uniform(2.0, 8.0)
        rot = np.random.uniform(0, 2*np.pi)
        pts = _optimize_2p(np.array([exp, rot]), DEADLINE - 20, beta=50.0)
        _update_best(pts)
    
    # === Phase 3: Coordinate-Level Polish ===
    # Polish the best point configuration
    if best_points is not None and time.time() < DEADLINE - 15:
        polished = _coordinate_polish(best_points, DEADLINE - 5, beta_start=50.0, beta_end=250.0, n_stages=3)
        _update_best(polished)
    
    # === Phase 4: Additional polish from scratch ===
    # Try polishing from a regular polygon
    if time.time() < DEADLINE - 10:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        reg_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        polished_reg = _coordinate_polish(reg_pts, DEADLINE - 5, beta_start=50.0, beta_end=250.0, n_stages=2)
        _update_best(polished_reg)
    
    # === Phase 5: Final hard-min polish ===
    if best_points is not None and time.time() < DEADLINE - 3:
        def _neg_hard(flat):
            pts = flat.reshape((N, 2))
            return -_calculate_score(pts)
        try:
            res = minimize(
                _neg_hard,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 100, 'maxfev': 200, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res.x is not None:
                final_pts = res.x.reshape((N, 2))
                _update_best(final_pts)
        except Exception:
            pass
    
    if best_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

_TRIPLETS = np.array(list(combinations(range(N), 3)))
_N_TRIPLES = _TRIPLETS.shape[0]

def _triangle_areas(pts):
    """Compute all C(n,3) triangle areas vectorized."""
    i, j, k = _TRIPLETS.T
    a = pts[i]
    b = pts[j]
    c = pts[k]
    areas = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    return areas

def _convex_hull_area(pts):
    """Compute area of convex hull."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 0.0
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    return np.min(areas) / hull_area

def _soft_min_objective(x, beta=50.0):
    """Negative soft-min of (min triangle area / hull area) for minimization.
    x: (2N,) flattened point coordinates."""
    pts = x.reshape(-1, 2)
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 1e6
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 1e6
    scaled = -areas * beta
    max_scaled = np.max(scaled)
    soft_min = -max_scaled / beta - np.log(np.sum(np.exp(scaled - max_scaled))) / beta
    return -soft_min / hull_area

def _soft_min_objective_grad(x, beta=50.0):
    """Gradient of negative soft-min of (min triangle area / hull area).
    x: (2N,) flattened point coordinates."""
    pts = x.reshape(-1, 2)
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return np.zeros_like(x)
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return np.zeros_like(x)
    
    # Soft-min weights
    scaled = -areas * beta
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    weights = exp_scaled / np.sum(exp_scaled)  # d(soft_min)/d(area_k) = -weights_k
    
    # Gradient of areas w.r.t. points
    i, j, k = _TRIPLETS.T
    a = pts[i]
    b = pts[j]
    c = pts[k]
    
    # area_k = 0.5 * |cross(b-a, c-a)|
    # For each point, accumulate gradient contribution
    grad = np.zeros_like(pts)
    
    # Cross product: cross = (b_x - a_x)(c_y - a_y) - (c_x - a_x)(b_y - a_y)
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    signs = np.sign(cross)
    
    # d(area_k)/d(a) = -0.5 * sign * (c-b) rotated 90 deg
    # d(area_k)/d(b) = 0.5 * sign * (a-c) rotated 90 deg  
    # d(area_k)/d(c) = 0.5 * sign * (b-a) rotated 90 deg
    
    # For point a (index i):
    # d(area)/d(a_x) = -0.5 * sign * (c_y - b_y)
    # d(area)/d(a_y) =  0.5 * sign * (c_x - b_x)
    da_x = -0.5 * signs * (c[:, 1] - b[:, 1])
    da_y =  0.5 * signs * (c[:, 0] - b[:, 0])
    
    # For point b (index j):
    # d(area)/d(b_x) =  0.5 * sign * (a_y - c_y)
    # d(area)/d(b_y) = -0.5 * sign * (a_x - c_x)
    db_x =  0.5 * signs * (a[:, 1] - c[:, 1])
    db_y = -0.5 * signs * (a[:, 0] - c[:, 0])
    
    # For point c (index k):
    # d(area)/d(c_x) = -0.5 * sign * (b_y - a_y)
    # d(area)/d(c_y) =  0.5 * sign * (b_x - a_x)
    dc_x = -0.5 * signs * (b[:, 1] - a[:, 1])
    dc_y =  0.5 * signs * (b[:, 0] - a[:, 0])
    
    # Weighted sum: grad = -sum_k weights_k * d(area_k)/d(pts)
    # soft_min = sum_k -area_k * weights_k (approximately)
    # d(soft_min)/d(pts) = -sum_k weights_k * d(area_k)/d(pts)
    # d(-soft_min/hull)/d(pts) = sum_k weights_k * d(area_k)/d(pts) / hull
    
    w = weights / hull_area
    
    np.add.at(grad, i, np.column_stack([da_x * w, da_y * w]) * -1)  # negative because we minimize -soft_min/hull
    np.add.at(grad, j, np.column_stack([db_x * w, db_y * w]) * -1)
    np.add.at(grad, k, np.column_stack([dc_x * w, dc_y * w]) * -1)
    
    return grad

def _exact_objective(x):
    """Negative exact score for minimization (non-smooth)."""
    pts = x.reshape(-1, 2)
    return -_score(pts)

def _halton_sequence(n, base):
    """Generate first n Halton sequence values with given base."""
    result = []
    for i in range(1, n + 1):
        f = 1.0
        r = 0.0
        idx = i
        while idx > 0:
            f /= base
            r += f * (idx % base)
            idx //= base
        result.append(r)
    return np.array(result)

def _halton_points_2d(n, seed=0):
    """Generate n 2D points using Halton sequences (bases 2 and 3)."""
    rng = np.random.RandomState(seed)
    offset = rng.randint(1, 1000)
    h2 = _halton_sequence(n, 2)
    h3 = _halton_sequence(n, 3)
    pts = np.column_stack([h2[offset - 1:offset - 1 + n] if offset - 1 + n <= len(h2) else h2[:n],
                           h3[offset - 1:offset - 1 + n] if offset - 1 + n <= len(h3) else h3[:n]])
    return pts

def _concentric_ring_init(seed=0, n_inner=5, n_outer=8):
    """Generate points on two concentric rings for diverse initialization."""
    rng = np.random.RandomState(seed)
    pts = np.zeros((N, 2))
    # Inner ring
    angles_in = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + rng.uniform(0, 2*np.pi)
    r_in = 0.4 + rng.uniform(-0.1, 0.1)
    pts[:n_inner, 0] = r_in * np.cos(angles_in)
    pts[:n_inner, 1] = r_in * np.sin(angles_in)
    # Outer ring
    angles_out = np.linspace(0, 2 * np.pi, n_outer, endpoint=False) + rng.uniform(0, 2*np.pi)
    r_out = 1.0 + rng.uniform(-0.1, 0.1)
    pts[n_inner:, 0] = r_out * np.cos(angles_out)
    pts[n_inner:, 1] = r_out * np.sin(angles_out)
    return pts

def _regular_polygon_init(seed=0):
    """Regular polygon with small perturbation."""
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.uniform(0, 2*np.pi)
    r = 1.0 + rng.normal(0, 0.05, N)
    pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    return pts

def _nelder_mead_polish(pts, max_iter=500, tol=1e-10):
    """Polish using Nelder-Mead on the exact (non-smooth) objective."""
    x0 = pts.flatten()
    try:
        result = minimize(
            _exact_objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol, 'adaptive': True}
        )
        return result.x.reshape(-1, 2)
    except Exception:
        return pts

def _adaptive_beta_optimize(pts0, deadline, start_beta=10.0, end_beta=200.0, n_stages=4):
    """Multi-stage optimization with increasing beta for coarse-to-fine search."""
    best_pts = pts0.copy()
    best_score = _score(best_pts)
    betas = np.geomspace(start_beta, end_beta, n_stages)
    
    for beta in betas:
        if time.time() > deadline - 5:
            break
        
        try:
            x0 = best_pts.flatten()
            result = minimize(
                _soft_min_objective,
                x0,
                args=(beta,),
                jac=lambda x, b=beta: _soft_min_objective_grad(x, b),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            pts = result.x.reshape(-1, 2)
            score = _score(pts)
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            continue
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_score = 0.0
    best_pts = _regular_polygon_init(0)
    
    # Phase 1: Multi-start with adaptive beta optimization
    init_strategies = [
        _regular_polygon_init,
        _concentric_ring_init,
        lambda s: _halton_points_2d(N, seed=s) * 0.8,
    ]
    
    n_starts = 20
    for i in range(n_starts):
        if time.time() > deadline - 40:
            break
        
        strategy_idx = i % len(init_strategies)
        seed = i
        try:
            pts0 = init_strategies[strategy_idx](seed)
            pts, score = _adaptive_beta_optimize(pts0, deadline - 35)
            
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            continue
    
    # Phase 2: Nelder-Mead polish on the best solution
    if time.time() < deadline - 15:
        polished = _nelder_mead_polish(best_pts, max_iter=800, tol=1e-12)
        score = _score(polished)
        if score > best_score:
            best_score = score
            best_pts = polished.copy()
    
    # Phase 3: Additional diverse starts with full polish if time permits
    for seed in range(8):
        if time.time() > deadline - 8:
            break
        
        try:
            pts0 = _concentric_ring_init(seed + 100)
            pts, score = _adaptive_beta_optimize(pts0, deadline - 5)
            
            # Quick NM polish
            if time.time() < deadline - 3:
                pts = _nelder_mead_polish(pts, max_iter=300, tol=1e-11)
                score = _score(pts)
            
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            continue
    
    return best_pts
# EVOLVE-BLOCK-END

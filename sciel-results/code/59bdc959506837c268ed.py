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
    # Use log-sum-exp for numerical stability
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
    
    # Soft-min gradient: d(soft_min)/d(area_k) = -exp(-beta * area_k) / sum(exp(-beta * area_j))
    scaled = -areas * beta
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    weights = exp_scaled / np.sum(exp_scaled)  # d(soft_min)/d(area_k) = -weights[k]
    
    # Gradient of areas w.r.t. points
    # area_k = 0.5 * |cross(b-a, c-a)| for triplet k
    # We need d(area_k)/d(pts) for each point
    grad_areas = np.zeros((N, 2))
    i, j, k = _TRIPLETS.T
    a = pts[i]
    b = pts[j]
    c = pts[k]
    
    # Cross product: (b_x - a_x)*(c_y - a_y) - (c_x - a_x)*(b_y - a_y)
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    sign = np.sign(cross)
    
    # d(area)/d(a) = -0.5 * sign * [(c_y - a_y), -(c_x - a_x)]
    da_dx = -0.5 * sign * (c[:, 1] - a[:, 1])
    da_dy = 0.5 * sign * (c[:, 0] - a[:, 0])
    
    # d(area)/d(b) = 0.5 * sign * [(c_y - a_y), -(c_x - a_x)]
    db_dx = 0.5 * sign * (c[:, 1] - a[:, 1])
    db_dy = -0.5 * sign * (c[:, 0] - a[:, 0])
    
    # d(area)/d(c) = -0.5 * sign * [(b_y - a_y), -(b_x - a_x)]
    dc_dx = 0.5 * sign * (b[:, 1] - a[:, 1])
    dc_dy = -0.5 * sign * (b[:, 0] - a[:, 0])
    
    # Accumulate gradients weighted by soft-min weights
    for idx in range(_N_TRIPLES):
        w = weights[idx]
        grad_areas[i[idx], 0] += w * da_dx[idx]
        grad_areas[i[idx], 1] += w * da_dy[idx]
        grad_areas[j[idx], 0] += w * db_dx[idx]
        grad_areas[j[idx], 1] += w * db_dy[idx]
        grad_areas[k[idx], 0] += w * dc_dx[idx]
        grad_areas[k[idx], 1] += w * dc_dy[idx]
    
    # Chain rule: d(score)/d(pts) = d(soft_min)/d(areas) * d(areas)/d(pts) / hull_area
    # But we're minimizing -score, so negate
    grad_score = -grad_areas / hull_area
    return grad_score.flatten()

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

def _adaptive_soft_min_optimize(pts0, max_time=10.0):
    """Adaptive soft-min optimization with increasing beta."""
    start = time.time()
    x0 = pts0.flatten()
    best_x = x0.copy()
    best_score = _score(pts0)
    
    # Start with moderate beta, increase for precision
    betas = [30.0, 80.0, 150.0, 300.0]
    for beta in betas:
        if time.time() - start > max_time - 2:
            break
        try:
            result = minimize(
                lambda x: _soft_min_objective(x, beta=beta),
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            pts = result.x.reshape(-1, 2)
            score = _score(pts)
            if score > best_score:
                best_score = score
                best_x = result.x.copy()
        except Exception:
            continue
    
    return best_x.reshape(-1, 2), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_score = 0.0
    best_pts = _regular_polygon_init(0)
    
    # Phase 1: Multi-start with adaptive soft-min L-BFGS-B
    init_strategies = [
        _regular_polygon_init,
        _concentric_ring_init,
        lambda s: _halton_points_2d(N, seed=s) * 0.8,
    ]
    
    n_starts = 20
    for i in range(n_starts):
        if time.time() > deadline - 25:
            break
        
        strategy_idx = i % len(init_strategies)
        seed = i
        try:
            pts0 = init_strategies[strategy_idx](seed)
            pts, score = _adaptive_soft_min_optimize(pts0, max_time=6.0)
            
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            continue
    
    # Phase 2: Final high-precision polish with very high beta
    if time.time() < deadline - 8:
        try:
            x0 = best_pts.flatten()
            result = minimize(
                lambda x: _soft_min_objective(x, beta=500.0),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            pts = result.x.reshape(-1, 2)
            score = _score(pts)
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            pass
    
    # Phase 3: A few more diverse starts if time permits
    for seed in range(3):
        if time.time() > deadline - 5:
            break
        
        try:
            pts0 = _concentric_ring_init(seed + 100)
            pts, score = _adaptive_soft_min_optimize(pts0, max_time=4.0)
            
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            continue
    
    return best_pts
# EVOLVE-BLOCK-END

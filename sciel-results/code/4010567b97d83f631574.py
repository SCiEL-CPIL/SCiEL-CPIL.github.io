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

def _soft_min_objective(x, beta=100.0):
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

def _adaptive_soft_min_objective(x, beta_start=50.0, beta_end=200.0, max_iter=300):
    """Soft-min objective with adaptive beta that increases over iterations."""
    # Use a closure to track iteration count
    if not hasattr(_adaptive_soft_min_objective, '_iter_count'):
        _adaptive_soft_min_objective._iter_count = 0
    _adaptive_soft_min_objective._iter_count += 1
    
    # Linearly interpolate beta from start to end
    t = min(_adaptive_soft_min_objective._iter_count / max_iter, 1.0)
    beta = beta_start + (beta_end - beta_start) * t
    
    return _soft_min_objective(x, beta=beta)

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

def _three_ring_init(seed=0):
    """Generate points on three rings for more diverse initialization."""
    rng = np.random.RandomState(seed)
    pts = np.zeros((N, 2))
    n1, n2, n3 = 4, 5, 4
    
    # Inner ring
    angles1 = np.linspace(0, 2 * np.pi, n1, endpoint=False) + rng.uniform(0, 2*np.pi)
    r1 = 0.3 + rng.uniform(-0.05, 0.05)
    pts[:n1, 0] = r1 * np.cos(angles1)
    pts[:n1, 1] = r1 * np.sin(angles1)
    
    # Middle ring
    angles2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + rng.uniform(0, 2*np.pi)
    r2 = 0.65 + rng.uniform(-0.05, 0.05)
    pts[n1:n1+n2, 0] = r2 * np.cos(angles2)
    pts[n1:n1+n2, 1] = r2 * np.sin(angles2)
    
    # Outer ring
    angles3 = np.linspace(0, 2 * np.pi, n3, endpoint=False) + rng.uniform(0, 2*np.pi)
    r3 = 1.0 + rng.uniform(-0.05, 0.05)
    pts[n1+n2:, 0] = r3 * np.cos(angles3)
    pts[n1+n2:, 1] = r3 * np.sin(angles3)
    
    return pts

def _random_convex_init(seed=0):
    """Generate random points in a convex region with some structure."""
    rng = np.random.RandomState(seed)
    pts = np.zeros((N, 2))
    
    # Mix of ring-based and random points
    n_ring1 = 5
    n_ring2 = 4
    n_random = N - n_ring1 - n_ring2
    
    # Ring 1 (inner)
    angles1 = np.linspace(0, 2 * np.pi, n_ring1, endpoint=False) + rng.uniform(0, 2*np.pi)
    r1 = 0.35 + rng.uniform(-0.05, 0.05)
    pts[:n_ring1, 0] = r1 * np.cos(angles1)
    pts[:n_ring1, 1] = r1 * np.sin(angles1)
    
    # Ring 2 (outer)
    angles2 = np.linspace(0, 2 * np.pi, n_ring2, endpoint=False) + rng.uniform(0, 2*np.pi)
    r2 = 0.9 + rng.uniform(-0.05, 0.05)
    pts[n_ring1:n_ring1+n_ring2, 0] = r2 * np.cos(angles2)
    pts[n_ring1:n_ring1+n_ring2, 1] = r2 * np.sin(angles2)
    
    # Random points in the region
    for i in range(n_random):
        angle = rng.uniform(0, 2*np.pi)
        r = rng.uniform(0.5, 0.8)
        pts[n_ring1+n_ring2+i, 0] = r * np.cos(angle)
        pts[n_ring1+n_ring2+i, 1] = r * np.sin(angle)
    
    return pts

def _regular_polygon_init(seed=0):
    """Regular polygon with small perturbation."""
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.uniform(0, 2*np.pi)
    r = 1.0 + rng.normal(0, 0.05, N)
    pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    return pts

def _nelder_mead_polish(pts, max_iter=1000, tol=1e-12):
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

def _high_precision_polish(pts, max_iter=2000, tol=1e-14):
    """High-precision polish with very tight tolerances."""
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

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_score = 0.0
    best_pts = _regular_polygon_init(0)
    
    # Phase 1: Multi-start with adaptive soft-min L-BFGS-B on full 2D coordinates
    init_strategies = [
        _regular_polygon_init,
        _concentric_ring_init,
        _three_ring_init,
        _random_convex_init,
        lambda s: _halton_points_2d(N, seed=s) * 0.8,
    ]
    
    n_starts = 30
    for i in range(n_starts):
        if time.time() > deadline - 40:
            break
        
        strategy_idx = i % len(init_strategies)
        seed = i
        try:
            pts0 = init_strategies[strategy_idx](seed)
            x0 = pts0.flatten()
            
            # Reset iteration counter for adaptive objective
            _adaptive_soft_min_objective._iter_count = 0
            
            result = minimize(
                _adaptive_soft_min_objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            pts = result.x.reshape(-1, 2)
            score = _score(pts)
            
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            continue
    
    # Phase 2: Nelder-Mead polish on the best solution
    if time.time() < deadline - 20:
        polished = _nelder_mead_polish(best_pts, max_iter=1000, tol=1e-12)
        score = _score(polished)
        if score > best_score:
            best_score = score
            best_pts = polished.copy()
    
    # Phase 3: More diverse starts with full polish if time permits
    for seed in range(10):
        if time.time() > deadline - 10:
            break
        
        try:
            # Use different initialization strategies
            if seed % 2 == 0:
                pts0 = _three_ring_init(seed + 100)
            else:
                pts0 = _random_convex_init(seed + 200)
            x0 = pts0.flatten()
            
            # Reset iteration counter
            _adaptive_soft_min_objective._iter_count = 0
            
            result = minimize(
                _adaptive_soft_min_objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            pts = result.x.reshape(-1, 2)
            # Quick NM polish
            pts = _nelder_mead_polish(pts, max_iter=500, tol=1e-12)
            score = _score(pts)
            
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            continue
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 5:
        final_polished = _high_precision_polish(best_pts, max_iter=2000, tol=1e-14)
        score = _score(final_polished)
        if score > best_score:
            best_score = score
            best_pts = final_polished.copy()
    
    return best_pts
# EVOLVE-BLOCK-END

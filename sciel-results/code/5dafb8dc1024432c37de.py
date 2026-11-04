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

def _adaptive_soft_min_objective_factory(betas):
    """Create a factory that returns objectives with increasing beta."""
    def make_obj(beta):
        def obj(x):
            return _soft_min_objective(x, beta=beta)
        return obj
    return [make_obj(b) for b in betas]

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
    """Three rings: inner (3 pts), middle (5 pts), outer (5 pts)."""
    rng = np.random.RandomState(seed)
    pts = np.zeros((N, 2))
    # Inner ring: 3 points
    angles1 = np.linspace(0, 2*np.pi, 3, endpoint=False) + rng.uniform(0, 2*np.pi)
    r1 = 0.25 + rng.uniform(-0.05, 0.05)
    pts[:3, 0] = r1 * np.cos(angles1)
    pts[:3, 1] = r1 * np.sin(angles1)
    # Middle ring: 5 points
    angles2 = np.linspace(0, 2*np.pi, 5, endpoint=False) + rng.uniform(0, 2*np.pi)
    r2 = 0.6 + rng.uniform(-0.05, 0.05)
    pts[3:8, 0] = r2 * np.cos(angles2)
    pts[3:8, 1] = r2 * np.sin(angles2)
    # Outer ring: 5 points
    angles3 = np.linspace(0, 2*np.pi, 5, endpoint=False) + rng.uniform(0, 2*np.pi)
    r3 = 1.0 + rng.uniform(-0.05, 0.05)
    pts[8:, 0] = r3 * np.cos(angles3)
    pts[8:, 1] = r3 * np.sin(angles3)
    return pts

def _random_convex_init(seed=0):
    """Random convex polygon vertices with jitter."""
    rng = np.random.RandomState(seed)
    angles = np.sort(rng.uniform(0, 2*np.pi, N))
    # Ensure minimum angular separation
    min_sep = 2*np.pi / N * 0.6
    for i in range(1, N):
        if angles[i] - angles[i-1] < min_sep:
            angles[i] = angles[i-1] + min_sep
    if angles[-1] - angles[0] + 2*np.pi < min_sep:
        angles[0] = angles[-1] - 2*np.pi + min_sep
    radii = 0.8 + 0.2 * rng.uniform(0, 1, N)
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
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

def _multi_nm_polish(pts, max_iter_per_pass=300, n_passes=3, tol=1e-12):
    """Multiple Nelder-Mead passes with decreasing initial simplex size."""
    current = pts.copy()
    for i in range(n_passes):
        if time.time() > _DEADLINE - 5:
            break
        polished = _nelder_mead_polish(current, max_iter=max_iter_per_pass, tol=tol)
        score_new = _score(polished)
        score_old = _score(current)
        if score_new > score_old:
            current = polished
    return current

def construct():
    global _DEADLINE
    start_time = time.time()
    _DEADLINE = start_time + 145
    
    best_score = 0.0
    best_pts = _regular_polygon_init(0)
    
    # Phase 1: Multi-start with adaptive beta L-BFGS-B
    init_strategies = [
        _regular_polygon_init,
        _concentric_ring_init,
        _three_ring_init,
        _random_convex_init,
        lambda s: _halton_points_2d(N, seed=s) * 0.8,
    ]
    
    # Use adaptive beta: start low, increase for refinement
    betas = [30.0, 80.0, 150.0]
    
    n_starts = 20
    for i in range(n_starts):
        if time.time() > _DEADLINE - 35:
            break
        
        strategy_idx = i % len(init_strategies)
        seed = i
        try:
            pts0 = init_strategies[strategy_idx](seed)
            x0 = pts0.flatten()
            
            # First pass with lower beta for exploration
            result1 = minimize(
                lambda x: _soft_min_objective(x, beta=betas[0]),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            # Second pass with higher beta for refinement
            if time.time() < _DEADLINE - 30:
                result2 = minimize(
                    lambda x: _soft_min_objective(x, beta=betas[1]),
                    result1.x,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-10}
                )
                pts = result2.x.reshape(-1, 2)
            else:
                pts = result1.x.reshape(-1, 2)
            
            score = _score(pts)
            
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            continue
    
    # Phase 2: Multi-pass Nelder-Mead polish on the best solution
    if time.time() < _DEADLINE - 15:
        polished = _multi_nm_polish(best_pts, max_iter_per_pass=400, n_passes=3, tol=1e-12)
        score = _score(polished)
        if score > best_score:
            best_score = score
            best_pts = polished.copy()
    
    # Phase 3: More diverse starts with full polish if time permits
    for seed in range(8):
        if time.time() > _DEADLINE - 8:
            break
        
        try:
            # Alternate between different init strategies
            if seed % 2 == 0:
                pts0 = _three_ring_init(seed + 200)
            else:
                pts0 = _random_convex_init(seed + 300)
            x0 = pts0.flatten()
            
            # Adaptive beta optimization
            result1 = minimize(
                lambda x: _soft_min_objective(x, beta=betas[0]),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if time.time() < _DEADLINE - 10:
                result2 = minimize(
                    lambda x: _soft_min_objective(x, beta=betas[2]),
                    result1.x,
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-13, 'gtol': 1e-10}
                )
                pts = result2.x.reshape(-1, 2)
            else:
                pts = result1.x.reshape(-1, 2)
            
            # Multi-pass NM polish
            pts = _multi_nm_polish(pts, max_iter_per_pass=200, n_passes=2, tol=1e-11)
            score = _score(pts)
            
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            continue
    
    # Final high-precision polish
    if time.time() < _DEADLINE - 3:
        final_polish = _nelder_mead_polish(best_pts, max_iter=1000, tol=1e-14)
        score = _score(final_polish)
        if score > best_score:
            best_score = score
            best_pts = final_polish.copy()
    
    return best_pts
# EVOLVE-BLOCK-END

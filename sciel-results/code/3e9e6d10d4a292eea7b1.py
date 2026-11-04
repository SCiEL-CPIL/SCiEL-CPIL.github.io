import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools
from itertools import combinations

N = 13

# EVOLVE-BLOCK-START
# Precompute all triangle index triplets for vectorized area computation
_TRIPLETS = np.array(list(combinations(range(N), 3)))
_N_TRIPLES = _TRIPLETS.shape[0]
_I, _J, _K = _TRIPLETS.T

def _triangle_areas(pts):
    """Compute all C(n,3) triangle areas vectorized."""
    a = pts[_I]
    b = pts[_J]
    c = pts[_K]
    areas = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
                          (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    return areas

def _convex_hull_area(pts):
    """Compute area of convex hull."""
    try:
        if len(pts) < 3:
            return 0.0
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
    return float(np.min(areas)) / hull_area

def _soft_min_objective(x, beta=50.0):
    """
    Negative soft-min of (min triangle area / hull area) for minimization.
    Uses log-sum-exp for numerical stability.
    """
    pts = x.reshape(-1, 2)
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 1e6
    
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 1e6
    
    # Soft minimum: min(x) ≈ -1/beta * log(sum(exp(-beta*x)))
    # We want to maximize min(areas)/hull_area, so minimize -min(areas)/hull_area
    # -min(areas) ≈ 1/beta * log(sum(exp(-beta*areas)))
    scaled = -areas * beta
    max_scaled = np.max(scaled)
    soft_min_val = -max_scaled / beta - np.log(np.sum(np.exp(scaled - max_scaled))) / beta
    
    # This soft_min_val is an approximation of -min(areas)
    # So we want to minimize soft_min_val / hull_area
    return soft_min_val / hull_area

def _exact_objective(x):
    """Negative exact score for minimization (non-smooth)."""
    pts = x.reshape(-1, 2)
    s = _score(pts)
    if s <= 0:
        return 1e6
    return -s

def _regular_polygon(n, radius=1.0, angle_offset=0.0):
    """Generate regular n-gon centered at origin."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False) + angle_offset
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

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
    h2 = _halton_sequence(n + offset, 2)
    h3 = _halton_sequence(n + offset, 3)
    pts = np.column_stack([h2[offset - 1:offset - 1 + n], h3[offset - 1:offset - 1 + n]])
    return pts

def _grid_init(seed=0):
    """Grid-based initialization, structurally good for small N."""
    rng = np.random.RandomState(seed)
    # Create a jittered grid
    side = int(np.ceil(np.sqrt(N)))
    # We have 13 points, so a 4x4 grid with one removed or similar
    # Let's use a 4x4 grid and pick 13 points
    x = np.linspace(-1, 1, 4) + rng.uniform(-0.1, 0.1, 4)
    y = np.linspace(-1, 1, 4) + rng.uniform(-0.1, 0.1, 4)
    pts = []
    for i in range(4):
        for j in range(4):
            pts.append([x[i], y[j]])
            if len(pts) == N:
                break
        if len(pts) == N:
            break
    return np.array(pts[:N])

def _elliptical_distortion_init(seed=0):
    """
    Structural idea: Apply an elliptical distortion (positive definite quadratic form)
    to a regular polygon. This enforces convexity and smoothness.
    """
    rng = np.random.RandomState(seed)
    # Start with regular polygon
    pts = _regular_polygon(N, radius=1.0, angle_offset=rng.uniform(0, 2*np.pi))
    
    # Apply a random symmetric positive definite matrix
    # A = [[a, b], [b, c]] with a>0, c>0, ac-b^2>0
    a = 1.0 + rng.uniform(-0.3, 0.3)
    c = 1.0 + rng.uniform(-0.3, 0.3)
    b = rng.uniform(-0.2, 0.2)
    
    # Ensure positive definiteness
    if a * c - b * b <= 0:
        b = 0.0
        a = 1.0
        c = 1.0
    
    M = np.array([[a, b], [b, c]])
    pts = pts @ M.T
    
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

def _lbfgs_polish(pts, beta=50.0, max_iter=500):
    """Polish using L-BFGS-B on the soft-min objective."""
    x0 = pts.flatten()
    try:
        result = minimize(
            lambda x: _soft_min_objective(x, beta=beta),
            x0,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        return result.x.reshape(-1, 2)
    except Exception:
        return pts

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    best_score = 0.0
    best_pts = None
    
    # Phase 1: Multi-start with diverse initializations
    # Use the structural elliptical distortion idea combined with other initializations
    init_strategies = [
        lambda s: _regular_polygon(N, angle_offset=s * 0.1),
        lambda s: _concentric_ring_init(seed=s),
        lambda s: _halton_points_2d(N, seed=s) * 0.8,
        lambda s: _grid_init(seed=s),
        lambda s: _elliptical_distortion_init(seed=s),
    ]
    
    n_starts = 30
    for i in range(n_starts):
        if time.time() > deadline - 30:
            break
        
        strategy_idx = i % len(init_strategies)
        seed = i
        try:
            pts0 = init_strategies[strategy_idx](seed)
            pts0 = np.asarray(pts0, dtype=float)
            
            # First pass: L-BFGS-B with soft-min
            result = minimize(
                lambda x: _soft_min_objective(x, beta=50.0),
                pts0.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            pts = result.x.reshape(-1, 2)
            score = _score(pts)
            
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            continue
    
    # Phase 2: Polish the best solution with Nelder-Mead on exact objective
    if best_pts is not None and time.time() < deadline - 15:
        polished = _nelder_mead_polish(best_pts, max_iter=500)
        score = _score(polished)
        if score > best_score:
            best_score = score
            best_pts = polished.copy()
    
    # Phase 3: Try a few more diverse starts with full polish if time permits
    for seed in range(10):
        if time.time() > deadline - 5:
            break
        
        try:
            # Mix of initializations
            if seed % 2 == 0:
                pts0 = _elliptical_distortion_init(seed + 100)
            else:
                pts0 = _concentric_ring_init(seed + 200)
            
            pts0 = np.asarray(pts0, dtype=float)
            
            # L-BFGS-B polish
            result = minimize(
                lambda x: _soft_min_objective(x, beta=100.0),
                pts0.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-8}
            )
            
            pts = result.x.reshape(-1, 2)
            
            # Quick NM polish
            pts = _nelder_mead_polish(pts, max_iter=200)
            score = _score(pts)
            
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            continue
    
    # Final fallback
    if best_pts is None:
        best_pts = _regular_polygon(N)
    
    return np.asarray(best_pts, dtype=float)
# EVOLVE-BLOCK-END

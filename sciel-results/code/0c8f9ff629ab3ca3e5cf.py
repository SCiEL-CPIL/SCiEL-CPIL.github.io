"""Heilbronn convex regions n=14: maximize min triangle area / hull area."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 14
HALF_N = N // 2  # 7 pairs for central symmetry


def _hull_area(points):
    """Compute convex hull area of points (n,2)."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _min_triangle_area(points):
    """Compute minimum triangle area over all triples of n points."""
    n = len(points)
    min_area = float('inf')
    # Use vectorized approach for speed
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            # Area = 0.5 * |cross(p_j - p_i, p_k - p_i)| for all k > j
            v1 = points[j] - points[i]
            v2 = points[j + 1:] - points[i]
            # Cross product in 2D: v1[0]*v2[1] - v1[1]*v2[0]
            areas = 0.5 * np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area


def _score_from_half(half_points):
    """
    Given 7 half-points (nearly convex), construct the full 14-point set
    via central symmetry: full = half + (-half), then normalize by hull area.
    Returns min_triangle_area / hull_area.
    """
    half_points = np.asarray(half_points, dtype=np.float64)
    full_points = np.vstack([half_points, -half_points])
    hull_area = _hull_area(full_points)
    if hull_area <= 1e-15:
        return 0.0
    min_area = _min_triangle_area(full_points)
    return min_area / hull_area


def _neg_objective(half_points):
    """Negative score for minimization."""
    s = _score_from_half(half_points)
    return -s


def _initial_half_points(rng, mode='regular'):
    """Generate initial 7 half-points in a roughly convex configuration."""
    if mode == 'regular':
        # Regular heptagon vertices (slightly perturbed)
        angles = np.linspace(0, 2 * np.pi, HALF_N, endpoint=False)
        r = 1.0 + 0.01 * rng.gauss(0, 1, HALF_N)
        pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    elif mode == 'square_plus':
        # 4 corners of a square + 3 on edges
        pts = np.array([
            [1, 1], [-1, 1], [-1, -1], [1, -1],
            [0, 1], [1, 0], [0, 0]
        ], dtype=np.float64)
        pts += 0.05 * rng.gauss(0, 1, pts.shape)
    elif mode == 'random_convex':
        # Random points on a circle with random radii
        angles = np.sort(rng.uniform(0, 2 * np.pi, HALF_N))
        r = 0.5 + 0.5 * rng.random(HALF_N)
        pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    else:
        angles = np.linspace(0, 2 * np.pi, HALF_N, endpoint=False)
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
    return pts


def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    rng = np.random.RandomState(42)
    
    best_score = 0.0
    best_half = None
    
    # Multiple starting configurations
    initial_modes = ['regular', 'square_plus', 'random_convex']
    best_half = _initial_half_points(rng, initial_modes[0])
    best_score = _score_from_half(best_half)
    
    # Differential Evolution for global search in 14D
    from scipy.optimize import differential_evolution
    
    bounds = [(-2, 2)] * (2 * HALF_N)
    
    de_result = differential_evolution(
        _neg_objective,
        bounds=bounds,
        seed=42,
        maxiter=80,
        tol=1e-8,
        popsize=15,
        polish=True,
        init='sohs',
        constraints=()
    )
    
    if de_result.fun < 0:
        de_score = -de_result.fun
        if de_score > best_score:
            best_score = de_score
            best_half = de_result.x.reshape(HALF_N, 2)
    
    if time.time() > deadline - 30:
        # Not enough time for polish; return what we have
        half_pts = best_half
        full_pts = np.vstack([half_pts, -half_pts])
        return full_pts
    
    # Local polish with L-BFGS-B
    # Try multiple local optimizations from the best point
    for attempt in range(3):
        if time.time() > deadline - 10:
            break
        
        start = best_half.copy()
        if attempt > 0:
            start = start + 0.01 * rng.gauss(0, 1, start.shape)
        
        result = minimize(
            _neg_objective,
            start,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        
        if result.fun < 0:
            local_score = -result.fun
            if local_score > best_score:
                best_score = local_score
                best_half = result.x.reshape(HALF_N, 2)
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        result = minimize(
            _neg_objective,
            best_half,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-12}
        )
        if result.fun < 0:
            local_score = -result.fun
            if local_score > best_score:
                best_score = local_score
                best_half = result.x.reshape(HALF_N, 2)
    
    # Construct final 14-point set
    half_pts = best_half
    full_pts = np.vstack([half_pts, -half_pts])
    
    return full_pts

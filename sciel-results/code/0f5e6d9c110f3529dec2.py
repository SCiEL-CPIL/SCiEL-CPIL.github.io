"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
import itertools


def _pairwise_distances(points):
    """Compute all pairwise distances between points."""
    n = points.shape[0]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    # Zero out diagonal
    np.fill_diagonal(dists, 0)
    return dists


def _compute_score(points):
    """Compute the squared ratio of max to min pairwise distance."""
    dists = _pairwise_distances(points)
    # Get unique pairwise distances (upper triangle, excluding zero)
    iu = np.triu_indices_from(dists, k=1)
    pd = dists[iu]
    pd = pd[pd > 1e-12]
    if len(pd) == 0:
        return 1e10
    dmin = np.min(pd)
    dmax = np.max(pd)
    if dmin < 1e-12:
        return 1e10
    return (dmax / dmin) ** 2


def _objective_flat(x):
    """Objective function for flat parameter vector."""
    points = x.reshape(-1, D)
    return _compute_score(points)


def _tetrahedron_with_midpoints():
    """Create 14 points: 4 tetrahedron vertices + 6 edge midpoints + 4 face centers."""
    # Regular tetrahedron vertices
    a = 1.0
    v1 = np.array([1, 1, 1]) * a
    v2 = np.array([1, -1, -1]) * a
    v3 = np.array([-1, 1, -1]) * a
    v4 = np.array([-1, -1, 1]) * a
    
    # 6 edge midpoints
    m12 = (v1 + v2) / 2
    m13 = (v1 + v3) / 2
    m14 = (v1 + v4) / 2
    m23 = (v2 + v3) / 2
    m24 = (v2 + v4) / 2
    m34 = (v3 + v4) / 2
    
    # 4 face centers (centroid of each face)
    f123 = (v1 + v2 + v3) / 3
    f124 = (v1 + v2 + v4) / 3
    f134 = (v1 + v3 + v4) / 3
    f234 = (v2 + v3 + v4) / 3
    
    points = np.array([v1, v2, v3, v4, m12, m13, m14, m23, m24, m34, f123, f124, f134, f234])
    return points


def _random_start():
    """Generate a random starting configuration."""
    return np.random.randn(N, D) * 1.0


def _optimize_from_start(start_points, deadline):
    """Optimize from a given starting configuration."""
    x0 = start_points.flatten()
    
    # Use Nelder-Mead for robustness
    try:
        result = minimize(_objective_flat, x0, method='Nelder-Mead',
                         options={'maxiter': 500, 'fatol': 1e-10, 'xatol': 1e-10,
                                 'adaptive': True})
        if result.success or _compute_score(result.x.reshape(-1, D)) < _compute_score(x0.reshape(-1, D)):
            return result.x.reshape(-1, D), _compute_score(result.x.reshape(-1, D))
    except Exception:
        pass
    
    return start_points, _compute_score(start_points)


def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    best_points = None
    best_score = 1e10
    
    # Try multiple starting configurations
    starts = []
    
    # Tetrahedron with midpoints and face centers
    starts.append(_tetrahedron_with_midpoints())
    
    # Random starts
    rng = np.random.RandomState(42)
    for _ in range(5):
        starts.append(rng.randn(N, D) * 1.0)
    
    # Try each start
    for i, start in enumerate(starts):
        if time.time() > deadline - 5:
            break
        
        points, score = _optimize_from_start(start, deadline)
        if score < best_score:
            best_score = score
            best_points = points.copy()
    
    # If we still have time, do more random restarts
    while time.time() < deadline - 10:
        start = rng.randn(N, D) * 1.0
        points, score = _optimize_from_start(start, deadline)
        if score < best_score:
            best_score = score
            best_points = points.copy()
    
    if best_points is None:
        # Fallback to grid
        side = int(np.ceil(N ** (1 / D)))
        grid = list(itertools.product(range(side), repeat=D))[:N]
        best_points = np.array(grid, dtype=float)
    
    return best_points
# EVOLVE-BLOCK-END

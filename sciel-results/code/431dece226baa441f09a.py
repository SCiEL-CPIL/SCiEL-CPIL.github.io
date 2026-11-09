# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time


def _all_triangles(n):
    """Precompute all C(n,3) triangle index triples."""
    idx = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx.append((i, j, k))
    return np.array(idx, dtype=int)


def _triangle_areas(pts, tri_idx):
    """Compute areas of all triangles given points and index triples."""
    a = pts[tri_idx[:, 0]]
    b = pts[tri_idx[:, 1]]
    c = pts[tri_idx[:, 2]]
    # area = 0.5*|x1(y2-y3)+x2(y3-y1)+x3(y1-y2)|
    area = 0.5 * np.abs(a[:, 0] * (b[:, 1] - c[:, 1]) +
                        b[:, 0] * (c[:, 1] - a[:, 1]) +
                        c[:, 0] * (a[:, 1] - b[:, 1]))
    return area


def _hull_area(pts):
    """Convex hull area; 0 on failure."""
    try:
        hull = ConvexHull(pts)
        return float(hull.volume)
    except Exception:
        return 0.0


def _score(pts):
    """Compute A_min / A_hull."""
    n = len(pts)
    tri_idx = _all_triangles(n)
    areas = _triangle_areas(pts, tri_idx)
    a_min = float(areas.min())
    a_hull = _hull_area(pts)
    if a_hull <= 0:
        return 0.0
    return a_min / a_hull


def _halton_sequence(n, base=2):
    """Generate a Halton sequence in 2D."""
    def halton_1d(n, base):
        result = np.zeros(n)
        f = 1.0
        for i in range(1, n + 1):
            f /= base
            result[i - 1] = f * (i % base)
        return result

    # Use bases 2 and 3 for 2D
    h1 = halton_1d(n, 2)
    h2 = halton_1d(n, 3)
    return np.column_stack([h1, h2])


def _regular_polygon_init(n, n_sides=12, center_weight=1.0):
    """Initialize with regular polygon vertices + center."""
    angles = 2 * np.pi * np.arange(n_sides) / n_sides
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    # Add center
    pts = np.vstack([pts, np.array([[0.0, 0.0]])])
    # Shuffle to mix
    rng = np.random.default_rng(seed=42)
    perm = rng.permutation(n)
    return pts[perm]


def _grid_init(n):
    """Initialize with a grid-like pattern."""
    rng = np.random.default_rng(seed=42)
    # 4x4 grid minus one point, then jitter
    grid = []
    for i in range(4):
        for j in range(4):
            grid.append([i - 1.5, j - 1.5])
    grid = np.array(grid[:n])
    # Add small random jitter
    grid += rng.random((n, 2)) * 0.1
    return grid


def _random_init(n, seed=42):
    """Random initialization in unit square."""
    rng = np.random.default_rng(seed)
    return rng.random((n, 2))


def _neg_score(pts_flat):
    """Objective function for minimizer: negative of the score."""
    n = 13
    pts = pts_flat.reshape((n, 2))
    if not np.all(np.isfinite(pts)):
        return 1e6
    return -_score(pts)


def _nelder_mead_optimize(pts, deadline, max_iter=500, maxfev=1000):
    """Optimize using Nelder-Mead."""
    n = len(pts)
    x0 = pts.flatten()
    
    # Define options
    options = {
        'maxiter': max_iter,
        'maxfev': maxfev,
        'xatol': 1e-8,
        'fatol': 1e-8
    }
    
    try:
        res = minimize(_neg_score, x0, method='Nelder-Mead', options=options)
        if res.success:
            best_pts = res.x.reshape((n, 2))
            best_score = _score(best_pts)
            return best_pts, best_score
    except Exception:
        pass
    
    return pts, _score(pts)


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize A_min / A_hull.
    
    Returns:
        points: np.ndarray of shape (13,2)
    """
    n = 13
    
    deadline = time.time() + 600
    
    # Multiple initializations
    inits = []
    inits.append(('polygon', _regular_polygon_init(n)))
    inits.append(('grid', _grid_init(n)))
    inits.append(('halton', _halton_sequence(n)))
    
    # Several random inits with different seeds
    for s in [0, 1, 2, 3, 4]:
        inits.append((f'rand{s}', _random_init(n, seed=s)))
    
    best_pts = None
    best_score = 0.0
    
    for name, pts in inits:
        if time.time() > deadline - 10:
            break
        
        # Run Nelder-Mead optimization
        result_pts, result_score = _nelder_mead_optimize(
            pts, 
            deadline,
            max_iter=200, 
            maxfev=500
        )
        
        if result_score > best_score:
            best_score = result_score
            best_pts = result_pts.copy()
            
        # Optional: Run a second, finer pass on the best result found so far
        # if time permits, we could do this, but let's stick to one pass per init for now
        # to ensure we explore all inits.
        
    if best_pts is None:
        # Fallback: return random points
        best_pts = _random_init(n, seed=42)
    
    # Final validation
    if not np.all(np.isfinite(best_pts)):
        best_pts = _random_init(n, seed=42)
    
    return best_pts


# EVOLVE-BLOCK-END

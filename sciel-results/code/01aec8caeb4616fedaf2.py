"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

DEADLINE = 150.0
START_TIME = time.time()
SAFETY_MARGIN = 5.0


def _min_triangle_area(pts):
    """Compute minimum triangle area among all triples."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-9:
        return 0.0
    min_ta = _min_triangle_area(pts)
    return min_ta / hull_area


def _neg_score(pts):
    """Negative score for minimization."""
    return -_score(pts)


def _grid_initialization():
    """Generate a 4x4 grid and select best 13 points."""
    coords = np.array([[i/3.0, j/3.0] for i in range(4) for j in range(4)])
    
    best_subset = None
    best_score = -1
    
    indices = range(16)
    for subset in combinations(indices, 13):
        pts = coords[list(subset)]
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_subset = subset
            
    return coords[list(best_subset)]


def construct():
    """Construct 13 points maximizing min triangle area / convex hull area."""
    
    # 1. Get a good initial guess from grid search
    pts0 = _grid_initialization()
    
    # 2. Local refinement
    best_pts = pts0.copy()
    best_score = _score(pts0)
    
    elapsed = time.time() - START_TIME
    remaining = DEADLINE - SAFETY_MARGIN - elapsed
    
    # Try a few restarts with small perturbations if time allows
    # But first, refine the best one
    
    if remaining > 10:
        def _flatten(pts):
            return pts.flatten()
            
        def _unflatten(x):
            return x.reshape(N, 2)
            
        # Refine the best grid solution
        try:
            result = minimize(
                _neg_score,
                _flatten(best_pts),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if result.fun < 0:
                candidate = _unflatten(result.x)
                cand_score = -result.fun
                if cand_score > best_score:
                    best_pts = candidate
                    best_score = cand_score
        except Exception:
            pass
        
        # Try 3 more random perturbations of the initial grid points
        elapsed = time.time() - START_TIME
        remaining = DEADLINE - SAFETY_MARGIN - elapsed
        for _ in range(3):
            if remaining < 10:
                break
            pts_perturbed = pts0 + np.random.normal(0, 0.05, size=(N, 2))
            try:
                result = minimize(
                    _neg_score,
                    _flatten(pts_perturbed),
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                if result.fun < 0:
                    candidate = _unflatten(result.x)
                    cand_score = -result.fun
                    if cand_score > best_score:
                        best_pts = candidate
                        best_score = cand_score
            except Exception:
                pass
            elapsed = time.time() - START_TIME
            remaining = DEADLINE - SAFETY_MARGIN - elapsed

    return best_pts
# EVOLVE-BLOCK-END

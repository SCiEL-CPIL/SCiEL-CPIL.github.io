"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

DEADLINE = 150.0
START_TIME = time.time()
SAFETY_MARGIN = 5.0


def _min_triangle_area(pts):
    """Compute minimum triangle area among all triples using vectorization."""
    n = len(pts)
    min_area = float('inf')
    # Vectorized calculation for speed
    # Create indices for all triples
    i, j, k = np.indices((n, n, n))
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p0 = pts[i]
    p1 = pts[j]
    p2 = pts[k]
    
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


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


def construct():
    """Construct 13 points maximizing min triangle area / convex hull area."""
    
    # 1. Initial guess: Regular 13-gon on unit circle
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts0 = np.column_stack([np.cos(angles), np.sin(angles)])
    
    best_pts = pts0.copy()
    best_score = _score(pts0)
    
    # 2. Local refinement using Nelder-Mead
    # We limit the time and iterations to avoid timeout
    elapsed = time.time() - START_TIME
    remaining = DEADLINE - SAFETY_MARGIN - elapsed
    
    if remaining > 5:
        def _flatten(pts):
            return pts.flatten()
            
        def _unflatten(x):
            return x.reshape(N, 2)
            
        # Try optimizing from the regular polygon
        try:
            # Use a tighter tolerance and maxiter to keep it fast
            result = minimize(
                _neg_score,
                _flatten(best_pts),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            if result.fun < 0:
                candidate = _unflatten(result.x)
                cand_score = -result.fun
                if cand_score > best_score:
                    best_pts = candidate
                    best_score = cand_score
        except Exception:
            pass
        
        # Try a few random restarts if time remains
        elapsed = time.time() - START_TIME
        remaining = DEADLINE - SAFETY_MARGIN - elapsed
        for _ in range(5):
            if remaining < 10:
                break
            
            # Random perturbation of the current best
            pts_perturbed = best_pts + np.random.normal(0, 0.1, size=(N, 2))
            try:
                result = minimize(
                    _neg_score,
                    _flatten(pts_perturbed),
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
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

    # Final check to ensure valid output
    if _score(best_pts) <= 0:
        # Fallback to regular polygon if something went wrong
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_pts
# EVOLVE-BLOCK-END

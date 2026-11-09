# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull


def _min_tri(pts):
    """Return (min_area, (i,j,k)) for the smallest triangle among 13 points."""
    n = len(pts)
    best = float('inf')
    best_idx = (0, 1, 2)
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            for k in range(j + 1, n):
                xk, yk = pts[k]
                area = 0.5 * abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
                if area < best:
                    best = area
                    best_idx = (i, j, k)
    return best, best_idx


def _hull_area(pts):
    """Return area of the convex hull, or a large penalty if hull fails."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # in 2D, volume is area
    except Exception:
        return float('inf')


def _score(pts):
    """A_min / A_hull; return 0.0 if invalid."""
    if not np.all(np.isfinite(pts)):
        return 0.0
    A_min, _ = _min_tri(pts)
    A_hull = _hull_area(pts)
    if not np.isfinite(A_hull) or A_hull <= 0:
        return 0.0
    return A_min / A_hull


def _objective(pts):
    """Maximize _score; return negative for Nelder-Mead minimization."""
    return -_score(pts)


def _initial_points(seed):
    """Structured starting point: regular 12-gon on unit circle + center."""
    rng = np.random.default_rng(seed)
    n_out = 12
    angles = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    # Slight random radius jitter to break perfect symmetry
    radii = 1.0 + rng.uniform(-0.05, 0.05, n_out)
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    # Center point
    pts = np.vstack([pts, [[0.0, 0.0]]])
    return pts


def _nelder_mead(pts, max_iter=300, tol=1e-8, deadline=None):
    """Run Nelder-Mead on the 26-dimensional point coordinates."""
    from scipy.optimize import minimize
    x0 = pts.flatten()
    # Initial simplex: perturb each coordinate slightly
    x0 = np.array(x0, dtype=float)
    try:
        res = minimize(
            _objective,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': max_iter,
                'xatol': tol,
                'fatol': tol,
                'adaptive': True,
                'disp': False,
            },
        )
        return res.x.reshape(-1, 2)
    except Exception:
        return pts


def _local_refine(pts, deadline=None):
    """Nelder-Mead from multiple perturbations of the given point set."""
    best_pts = pts.copy()
    best_score = _score(best_pts)
    rng = np.random.default_rng(12345)

    for attempt in range(6):
        if deadline is not None and time.time() > deadline - 30:
            break
        # Create a perturbed copy
        perturbed = best_pts + rng.normal(0, 0.05, size=best_pts.shape)
        # Optimize
        result = _nelder_mead(perturbed, max_iter=200, deadline=deadline)
        s = _score(result)
        if s > best_score:
            best_score = s
            best_pts = result
    return best_pts


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points in a convex region to maximize
    A_min / A_hull, where A_min is the smallest triangle area and A_hull
    is the convex hull area.

    Returns:
        points: np.ndarray of shape (13,2) with finite coordinates.
    """
    start_time = time.time()
    deadline = start_time + 600  # hard limit; use 570s as safety margin
    safety_deadline = start_time + 570

    best_pts = None
    best_score = 0.0

    # Phase 1: Try several structured initializations with Nelder-Mead
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for seed in seeds:
        if time.time() > safety_deadline:
            break
        pts = _initial_points(seed)
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
        # Optimize from this starting point
        if time.time() > safety_deadline - 60:
            break
        optimized = _nelder_mead(pts, max_iter=400, deadline=safety_deadline)
        s = _score(optimized)
        if s > best_score:
            best_score = s
            best_pts = optimized.copy()

    # Phase 2: Local refinement from the best point found
    if best_pts is not None and time.time() < safety_deadline - 30:
        refined = _local_refine(best_pts, deadline=safety_deadline)
        s = _score(refined)
        if s > best_score:
            best_score = s
            best_pts = refined.copy()

    # Final validation
    if best_pts is None:
        # Fallback: regular 12-gon + center
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        best_pts = np.vstack([best_pts, [[0.0, 0.0]]])

    # Ensure finite values
    best_pts = np.where(np.isfinite(best_pts), best_pts, 0.0)

    return best_pts


# EVOLVE-BLOCK-END

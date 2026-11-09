# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize

def _triangle_areas(pts, tri_idx):
    """Compute all triangle areas vectorized."""
    a = pts[tri_idx[:, 0]]
    b = pts[tri_idx[:, 1]]
    c = pts[tri_idx[:, 2]]
    area = 0.5 * np.abs(
        a[:, 0] * (b[:, 1] - c[:, 1]) +
        b[:, 0] * (c[:, 1] - a[:, 1]) +
        c[:, 0] * (a[:, 1] - b[:, 1])
    )
    return area

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _get_tri_indices(n):
    idx = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx.append((i, j, k))
    return np.array(idx, dtype=int)

def _compute_score(pts, tri_idx):
    """Compute score given points and precomputed triangle indices."""
    areas = _triangle_areas(pts, tri_idx)
    min_area = areas.min()
    hull_area = _hull_area(pts)
    if hull_area <= 1e-9:
        return 0.0
    return min_area / hull_area

def _objective(x, n, tri_idx):
    """Objective function for minimization (negative score)."""
    pts = x.reshape(n, 2)
    if not np.all(np.isfinite(pts)):
        return 1e6
    score = _compute_score(pts, tri_idx)
    return -score

def _regular_polygon(n, radius=1.0, center=(0, 0), angle_offset=0.0):
    angles = angle_offset + np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs = center[0] + radius * np.cos(angles)
    ys = center[1] + radius * np.sin(angles)
    return np.column_stack([xs, ys])

def heilbronn_convex13() -> np.ndarray:
    n = 13
    tri_idx = _get_tri_indices(n)
    
    deadline = time.time() + 590  # Safety margin before 600s limit
    
    best_pts = None
    best_score = -1.0
    
    # 1. Initial configuration: Regular 13-gon
    # This is a convex configuration, so ConvexHull will always work.
    pts_reg = _regular_polygon(13, radius=1.0)
    score_reg = _compute_score(pts_reg, tri_idx)
    if score_reg > best_score:
        best_score = score_reg
        best_pts = pts_reg.copy()
        
    # 2. Differential Evolution for global search
    # Bounds: Keep points within a reasonable box, e.g., [-2, 2]
    # This allows the hull to change shape and size.
    # We scale the problem to unit radius initially.
    bounds = [(-2.0, 2.0)] * (n * 2)
    
    try:
        # Use DE to find a better global configuration
        # maxiter and popsize are tuned to fit within time budget
        # DE is stochastic, so set seed for reproducibility
        np.random.seed(42)
        
        # We will run DE for a limited number of iterations
        # The callback will check time
        def de_callback(xk, convergence=True):
            nonlocal best_pts, best_score
            pts = xk.reshape(n, 2)
            if np.all(np.isfinite(pts)):
                score = _compute_score(pts, tri_idx)
                if score > best_score:
                    best_score = score
                    best_pts = pts.copy()
            return time.time() > deadline - 30  # Stop if close to deadline
        
        result_de = differential_evolution(
            _objective,
            bounds,
            args=(n, tri_idx),
            seed=42,
            maxiter=200,
            popsize=15,
            tol=1e-4,
            callback=de_callback,
            polish=False  # We will do our own polishing
        )
        
        if result_de.x is not None:
            pts_de = result_de.x.reshape(n, 2)
            score_de = _compute_score(pts_de, tri_idx)
            if score_de > best_score:
                best_score = score_de
                best_pts = pts_de.copy()
                
    except Exception as e:
        pass # If DE fails, proceed with best found so far

    # 3. Local Refinement using Nelder-Mead
    # Nelder-Mead is good for polishing non-smooth functions
    if best_pts is not None and time.time() < deadline - 20:
        try:
            result_nm = minimize(
                _objective,
                best_pts.flatten(),
                args=(n, tri_idx),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            pts_nm = result_nm.x.reshape(n, 2)
            score_nm = _compute_score(pts_nm, tri_idx)
            if score_nm > best_score:
                best_score = score_nm
                best_pts = pts_nm.copy()
        except Exception:
            pass

    # 4. Final check and return
    if best_pts is None or not np.all(np.isfinite(best_pts)):
        return _regular_polygon(13)
    
    return best_pts

# EVOLVE-BLOCK-END

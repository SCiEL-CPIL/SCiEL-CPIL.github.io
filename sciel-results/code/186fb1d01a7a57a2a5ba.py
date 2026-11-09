# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

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
    if not np.all(np.isfinite(pts)):
        return 0.0
    areas = _triangle_areas(pts, tri_idx)
    min_area = areas.min()
    hull_area = _hull_area(pts)
    if hull_area <= 1e-9:
        return 0.0
    return min_area / hull_area

def _smooth_min_objective(x, tri_idx, n, beta=80.0):
    """
    Smooth approximation of -min(triangle_areas) / hull_area.
    Uses log-sum-exp for smooth min of areas.
    """
    pts = x.reshape(n, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10
    
    areas = _triangle_areas(pts, tri_idx)
    if np.any(areas < 1e-12):
        return 1e10
    
    hull_area = _hull_area(pts)
    if hull_area < 1e-9:
        return 1e10
    
    # Smooth min: -log(sum(exp(-beta*areas)))/beta approximates min(areas)
    shifted = -beta * areas
    max_shifted = shifted.max()
    if max_shifted < -1000:
        soft_min_val = 0.0
    else:
        soft_min_val = -(max_shifted + np.log(np.exp(shifted - max_shifted).sum())) / beta
    
    score_approx = soft_min_val / hull_area
    return -score_approx

def _nelder_mead_objective(x, tri_idx, n):
    """Exact objective for Nelder-Mead: -min_area / hull_area."""
    pts = x.reshape(n, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10
    areas = _triangle_areas(pts, tri_idx)
    a_min = np.min(areas)
    if a_min < 1e-15:
        return 1e10
    hull_area = _hull_area(pts)
    if hull_area < 1e-15:
        return 1e10
    return -(a_min / hull_area)

def _regular_polygon(n):
    """Generate regular n-gon centered at origin."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _triangular_lattice_subset(n):
    """Generate points from triangular lattice, take n closest to centroid."""
    # Generate triangular lattice points in a grid
    points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i + 0.5 * j
            y = j * np.sqrt(3) / 2
            points.append([x, y])
    points = np.array(points)
    # Center the lattice
    centroid = points.mean(axis=0)
    points -= centroid
    # Sort by distance to centroid and take n closest
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    return points[idx[:n]]

def _random_points(n, seed):
    """Generate random points in unit square."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, (n, 2))

def _optimize_from_start(start_pts, tri_idx, n, deadline, beta=80.0):
    """
    Optimize from a starting point using L-BFGS-B with smooth objective,
    then polish with Nelder-Mead.
    Returns (best_pts, best_score).
    """
    best_pts = start_pts.copy()
    best_score = _compute_score(best_pts, tri_idx)
    
    # Phase 1: L-BFGS-B with smooth objective
    if time.time() < deadline - 30:
        try:
            res = minimize(
                _smooth_min_objective,
                start_pts.flatten(),
                args=(tri_idx, n, beta),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            res_pts = res.x.reshape(n, 2)
            res_score = _compute_score(res_pts, tri_idx)
            if res_score > best_score:
                best_score = res_score
                best_pts = res_pts
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead polish
    if time.time() < deadline - 10:
        try:
            res = minimize(
                _nelder_mead_objective,
                best_pts.flatten(),
                args=(tri_idx, n),
                method='nelder-mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            res_pts = res.x.reshape(n, 2)
            res_score = _compute_score(res_pts, tri_idx)
            if res_score > best_score:
                best_score = res_score
                best_pts = res_pts
        except Exception:
            pass
    
    return best_pts, best_score

def heilbronn_convex13() -> np.ndarray:
    n = 13
    tri_idx = _get_tri_indices(n)
    
    start_time = time.time()
    deadline = start_time + 580  # Safety margin for 600s limit
    
    # Generate diverse starting points
    starts = []
    
    # 1. Regular polygon
    starts.append(_regular_polygon(n))
    
    # 2. Triangular lattice subset
    starts.append(_triangular_lattice_subset(n))
    
    # 3. Random points with different seeds
    for seed in [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]:
        starts.append(_random_points(n, seed))
    
    # 4. Slightly perturbed regular polygon
    rng = np.random.default_rng(999)
    reg = _regular_polygon(n)
    perturbed = reg + rng.normal(0, 0.1, (n, 2))
    starts.append(perturbed)
    
    # 5. Elliptical distortion of regular polygon
    for sx, sy in [(1.3, 0.7), (0.7, 1.3), (1.2, 0.8), (0.8, 1.2)]:
        ell = _regular_polygon(n)
        ell[:, 0] *= sx
        ell[:, 1] *= sy
        starts.append(ell)
    
    # Optimize from each starting point
    best_pts = None
    best_score = -1.0
    
    for i, start_pts in enumerate(starts):
        if time.time() > deadline - 60:
            break
        
        # Skip if this start is already worse than best by a lot
        start_score = _compute_score(start_pts, tri_idx)
        if best_pts is not None and start_score < best_score * 0.5:
            continue
        
        try:
            opt_pts, opt_score = _optimize_from_start(start_pts, tri_idx, n, deadline)
            if opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts
        except Exception:
            pass
        
        # Check time
        if time.time() > deadline - 30:
            break
    
    # Final polish with Nelder-Mead if we have time
    if best_pts is not None and time.time() < deadline - 5:
        try:
            res = minimize(
                _nelder_mead_objective,
                best_pts.flatten(),
                args=(tri_idx, n),
                method='nelder-mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            res_pts = res.x.reshape(n, 2)
            res_score = _compute_score(res_pts, tri_idx)
            if res_score > best_score:
                best_score = res_score
                best_pts = res_pts
        except Exception:
            pass
    
    # Fallback
    if best_pts is None or not np.all(np.isfinite(best_pts)):
        best_pts = _regular_polygon(n)
    
    return best_pts

# EVOLVE-BLOCK-END

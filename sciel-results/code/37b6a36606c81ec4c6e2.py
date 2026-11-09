# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools


def _compute_score(points):
    """Compute A_min / A_hull. Returns 0 if invalid."""
    n = len(points)
    if n != 13:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    
    if hull_area <= 0:
        return 0.0
    
    # Compute min triangle area (vectorized)
    min_area = float('inf')
    for i in range(13):
        for j in range(i + 1, 13):
            for k in range(j + 1, 13):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + 
                                  p2[0] * (p3[1] - p1[1]) + 
                                  p3[0] * (p1[1] - p2[1]))
                if area < min_area:
                    min_area = area
    
    if min_area <= 0:
        return 0.0
    
    return min_area / hull_area


def _all_triangle_indices(n=13):
    """Precompute all triangle index triples."""
    return list(itertools.combinations(range(n), 3))


def _signed_triangle_areas(points, tri_indices):
    """Compute signed areas for all triangles. Returns array of shape (286,)."""
    n = len(points)
    areas = np.zeros(len(tri_indices))
    for idx, (i, j, k) in enumerate(tri_indices):
        p1, p2, p3 = points[i], points[j], points[k]
        areas[idx] = 0.5 * (p1[0] * (p2[1] - p3[1]) + 
                            p2[0] * (p3[1] - p1[1]) + 
                            p3[0] * (p1[1] - p2[1]))
    return areas


def _soft_min(values, beta):
    """Compute soft minimum: -log(sum(exp(-beta*v)))/beta"""
    # Use log-sum-exp trick for numerical stability
    min_val = np.min(values)
    shifted = values - min_val
    log_sum_exp = np.log(np.sum(np.exp(-beta * shifted))) + beta * min_val
    return -log_sum_exp / beta


def _soft_min_grad(values, beta):
    """Gradient of soft min with respect to values."""
    min_val = np.min(values)
    shifted = values - min_val
    weights = np.exp(-beta * shifted) / np.sum(np.exp(-beta * shifted))
    return -weights


def _soft_min_objective(points, tri_indices, beta):
    """Objective: negative soft-min of absolute triangle areas, normalized by hull area."""
    n = len(points)
    # Compute absolute triangle areas
    signed = _signed_triangle_areas(points, tri_indices)
    abs_areas = np.abs(signed)
    
    # Soft min of absolute areas
    sm = _soft_min(abs_areas, beta)
    
    # Convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 1e10  # Large penalty for invalid points
    
    if hull_area <= 0:
        return 1e10
    
    # We want to maximize sm/hull_area, so minimize negative
    return -sm / hull_area


def _soft_min_objective_grad(points, tri_indices, beta):
    """Gradient of the soft-min objective."""
    n = len(points)
    signed = _signed_triangle_areas(points, tri_indices)
    abs_areas = np.abs(signed)
    
    # Gradient of soft min w.r.t. abs_areas
    sm_grad = _soft_min_grad(abs_areas, beta)
    
    # Gradient of abs(signed) w.r.t. signed
    # d|s|/ds = sign(s), but at 0 it's undefined; use subgradient 0
    sign_signed = np.sign(signed)
    sign_signed[np.abs(signed) < 1e-12] = 0
    
    # Gradient of signed areas w.r.t. points
    # signed_area = 0.5*(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
    # d/dp1[0] = 0.5*(p2[1]-p3[1])
    # d/dp1[1] = 0.5*(p3[0]-p2[0])
    # d/dp2[0] = 0.5*(p3[1]-p1[1])
    # d/dp2[1] = 0.5*(p1[0]-p3[0])
    # d/dp3[0] = 0.5*(p1[1]-p2[1])
    # d/dp3[1] = 0.5*(p2[0]-p1[0])
    
    grad_points = np.zeros_like(points)
    for idx, (i, j, k) in enumerate(tri_indices):
        w = sm_grad[idx] * sign_signed[idx]
        p1, p2, p3 = points[i], points[j], points[k]
        # d signed / d p1
        grad_points[i, 0] += 0.5 * w * (p2[1] - p3[1])
        grad_points[i, 1] += 0.5 * w * (p3[0] - p2[0])
        # d signed / d p2
        grad_points[j, 0] += 0.5 * w * (p3[1] - p1[1])
        grad_points[j, 1] += 0.5 * w * (p1[0] - p3[0])
        # d signed / d p3
        grad_points[k, 0] += 0.5 * w * (p1[1] - p2[1])
        grad_points[k, 1] += 0.5 * w * (p2[0] - p1[0])
    
    # Now we need to account for the division by hull_area
    # f = -sm / hull_area
    # df/dpoints = -dsm/dpoints / hull_area + sm * dhull_area/dpoints / hull_area^2
    # For simplicity, we'll ignore the hull_area gradient term initially
    # and just use -sm_grad / hull_area approximation
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return grad_points  # Return raw gradient if hull fails
    
    if hull_area <= 0:
        return grad_points
    
    # The objective is -sm/hull_area, so gradient is -grad_sm/hull_area
    # But we computed grad_points as d(sm)/dpoints, so we need negative
    return -grad_points / hull_area


def _regular_polygon(n=13):
    """Generate points on a regular polygon."""
    angles = 2 * np.pi * np.arange(n) / n
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _concentric_rings(n=13):
    """Generate points in concentric rings."""
    points = []
    # Center point
    points.append([0.0, 0.0])
    # First ring: 6 points
    for i in range(6):
        angle = 2 * np.pi * i / 6
        points.append([0.5 * np.cos(angle), 0.5 * np.sin(angle)])
    # Second ring: 6 points
    for i in range(6):
        angle = 2 * np.pi * (i + 0.5) / 6
        points.append([1.0 * np.cos(angle), 1.0 * np.sin(angle)])
    
    # We have 13 points now
    return np.array(points[:13])


def _halton_sequence(n, base):
    """Generate Halton sequence."""
    result = np.zeros(n)
    for i in range(n):
        f = 1.0
        r = 0.0
        idx = i + 1
        while idx > 0:
            f /= base
            r += f * (idx % base)
            idx //= base
        result[i] = r
    return result


def _halton_points(n=13):
    """Generate 2D Halton sequence points."""
    x = _halton_sequence(n, 2)
    y = _halton_sequence(n, 3)
    points = np.column_stack([x, y])
    # Center and scale to unit square centered at origin
    points = (points - 0.5) * 2.0
    return points


def _random_points(n=13, seed=42):
    """Generate random points in unit square."""
    rng = np.random.default_rng(seed)
    return rng.random((n, 2)) - 0.5


def _optimize_from_start(points, tri_indices, time_limit, deadline, rng):
    """Run multi-phase optimization from a starting point."""
    best_points = points.copy()
    best_score = _compute_score(points)
    
    # Phase 1: L-BFGS-B with increasing beta
    betas = [5, 10, 20, 50, 100, 200, 500, 1000]
    
    current_points = points.copy()
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
        
        def objective(x):
            pts = x.reshape(13, 2)
            return _soft_min_objective(pts, tri_indices, beta)
        
        def gradient(x):
            pts = x.reshape(13, 2)
            return _soft_min_objective_grad(pts, tri_indices, beta).flatten()
        
        x0 = current_points.flatten()
        
        try:
            result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                             options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-12})
            new_points = result.x.reshape(13, 2)
            
            if np.all(np.isfinite(new_points)):
                new_score = _compute_score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
                current_points = new_points
        except Exception:
            pass
        
        if time.time() > deadline - 30:
            break
    
    # Phase 2: Final polish with very high beta
    if time.time() < deadline - 15:
        for beta in [2000, 5000]:
            if time.time() > deadline - 10:
                break
            
            def objective(x):
                pts = x.reshape(13, 2)
                return _soft_min_objective(pts, tri_indices, beta)
            
            def gradient(x):
                pts = x.reshape(13, 2)
                return _soft_min_objective_grad(pts, tri_indices, beta).flatten()
            
            x0 = best_points.flatten()
            
            try:
                result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                                 options={'maxiter': 100, 'gtol': 1e-10, 'ftol': 1e-14})
                new_points = result.x.reshape(13, 2)
                
                if np.all(np.isfinite(new_points)):
                    new_score = _compute_score(new_points)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_points.copy()
            except Exception:
                pass
    
    return best_points, best_score


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    deadline = time.time() + 620  # Leave 10s safety margin
    
    tri_indices = _all_triangle_indices(13)
    
    # Diverse initializations
    initializations = [
        _regular_polygon(13),
        _concentric_rings(13),
        _halton_points(13),
        _random_points(13, seed=1),
        _random_points(13, seed=2),
        _random_points(13, seed=3),
        _random_points(13, seed=4),
        _random_points(13, seed=5),
    ]
    
    best_points = None
    best_score = 0.0
    
    # Try each initialization with time budget
    n_starts = len(initializations)
    time_per_start = (deadline - time.time()) / n_starts
    
    for idx, init_points in enumerate(initializations):
        if time.time() > deadline - 30:
            break
        
        # Adjust time per start based on remaining
        remaining_starts = n_starts - idx
        time_left = deadline - time.time()
        time_for_this = min(time_per_start, time_left / max(1, remaining_starts))
        
        if time_for_this < 5:
            break
        
        rng = np.random.default_rng(seed=1000 + idx)
        points, score = _optimize_from_start(init_points, tri_indices, time_for_this, deadline, rng)
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # If we still have time, do additional random restarts from best
    if best_points is not None and time.time() < deadline - 60:
        remaining_time = deadline - time.time()
        n_extra = max(1, int(remaining_time / 60))
        
        for extra_idx in range(n_extra):
            if time.time() > deadline - 30:
                break
            
            rng = np.random.default_rng(seed=5000 + extra_idx)
            perturbed = best_points + rng.normal(0, 0.3, size=(13, 2))
            
            time_left = deadline - time.time()
            if time_left < 10:
                break
            
            points, score = _optimize_from_start(perturbed, tri_indices, time_left, deadline, rng)
            
            if score > best_score:
                best_score = score
                best_points = points.copy()
    
    if best_points is None:
        # Fallback
        best_points = _regular_polygon(13)
    
    return best_points


# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time


def _triangle_areas(pts):
    """Compute all triangle areas for 13 points (286 triangles)."""
    n = len(pts)
    areas = np.empty((n * (n - 1) * (n - 2)) // 6)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                areas[idx] = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
                idx += 1
    return areas


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return np.inf


def _score(pts):
    """Compute A_min / A_hull."""
    areas = _triangle_areas(pts)
    a_min = np.min(areas)
    if a_min <= 0:
        return 0.0
    a_hull = _hull_area(pts)
    if not np.isfinite(a_hull) or a_hull <= 0:
        return 0.0
    return a_min / a_hull


def _soft_min_penalty(pts, temperature=0.05):
    """
    Smooth approximation of the minimum triangle area using soft-min.
    Returns -soft_min(areas) so we can minimize it.
    """
    areas = _triangle_areas(pts)
    min_val = np.min(areas)
    if min_val < 1e-10:
        return min_val
    shifted = areas - min_val
    log_sum = np.log(np.sum(np.exp(-shifted / temperature)))
    soft_min = min_val - temperature * log_sum
    return -soft_min


def _objective(pts_flat, temperature=0.05):
    """Objective for L-BFGS-B: minimize -soft_min + penalty for hull area."""
    n = 13
    pts = pts_flat.reshape(n, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10
    
    penalty = _soft_min_penalty(pts, temperature)
    a_hull = _hull_area(pts)
    if not np.isfinite(a_hull) or a_hull <= 0:
        return 1e10
    
    alpha = 1.0
    return penalty + alpha * np.log(a_hull)


def _coordinate_descent(pts, max_iters=500, step=0.01, deadline=None):
    """
    Coordinate descent to refine points.
    """
    n = 13
    pts = pts.copy()
    best_score = _score(pts)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            for dx, dy in [(step, 0), (-step, 0), (0, step), (0, -step),
                           (step/2, step/2), (-step/2, step/2),
                           (step/2, -step/2), (-step/2, -step/2)]:
                new_pts = pts.copy()
                new_pts[i, 0] += dx
                new_pts[i, 1] += dy
                s = _score(new_pts)
                if s > best_score:
                    pts = new_pts
                    best_score = s
                    improved = True
        if not improved:
            step *= 0.5
            if step < 1e-6:
                break
    
    return pts, best_score


def _triangular_lattice_seed_13():
    """
    Generate 13 points from the triangular lattice.
    These are the 13 closest points to the origin in the triangular lattice.
    Basis: u=(1,0), v=(0.5, sqrt(3)/2)
    Points (i,j) with i^2 + ij + j^2 <= 3
    """
    pts = []
    # (0,0)
    pts.append([0.0, 0.0])
    
    # Norm squared 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    coords_1 = [
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)
    ]
    for i, j in coords_1:
        x = i + 0.5 * j
        y = (np.sqrt(3) / 2.0) * j
        pts.append([x, y])
        
    # Norm squared 3: (1,1), (-1,2), (2,-1), (-2,1), (1,-2), (-1,-2)
    coords_3 = [
        (1, 1), (-1, 2), (2, -1), (-2, 1), (1, -2), (-1, -2)
    ]
    for i, j in coords_3:
        x = i + 0.5 * j
        y = (np.sqrt(3) / 2.0) * j
        pts.append([x, y])
        
    return np.array(pts)


def _hexagonal_seed():
    """Create a hexagonal lattice seed for 13 points."""
    pts = []
    pts.append([0.0, 0.0])
    r1 = 0.5
    for i in range(6):
        angle = i * np.pi / 3
        pts.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    r2 = 1.0
    for i in range(6):
        angle = i * np.pi / 3 + np.pi / 6
        pts.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    return np.array(pts)


def heilbronn_convex13() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580
    
    # Primary seed: Triangular Lattice Subset
    seed1 = _triangular_lattice_seed_13()
    # Normalize
    seed1 = seed1 - np.mean(seed1, axis=0)
    scale = np.max(np.abs(seed1))
    if scale > 0:
        seed1 = seed1 / scale * 0.5
        
    seeds = [seed1]
    
    # Secondary seeds
    seeds.append(_hexagonal_seed())
    
    rng = np.random.default_rng(seed=42)
    for _ in range(3):
        seeds.append(rng.random((13, 2)))

    best_pts = None
    best_score = 0.0
    
    for seed_idx, seed in enumerate(seeds):
        if time.time() > deadline:
            break
        
        # Normalize seed
        seed = seed - np.mean(seed, axis=0)
        scale = np.max(np.abs(seed))
        if scale > 0:
            seed = seed / scale * 0.5
        
        # 1. L-BFGS-B on soft objective
        try:
            res = minimize(
                _objective,
                seed.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.success or res.fun < 1e8:
                pts = res.x.reshape(13, 2)
                if np.all(np.isfinite(pts)):
                    s = _score(pts)
                    if s > best_score:
                        best_score = s
                        best_pts = pts.copy()
        except Exception:
            pass
        
        # 2. Nelder-Mead on raw score (negated)
        # NM is derivative-free, good for non-smooth
        def nm_objective(x_flat):
            pts = x_flat.reshape(13, 2)
            if not np.all(np.isfinite(pts)):
                return 1e10
            s = _score(pts)
            if s == 0:
                return 1e10
            return -np.log(s) # Maximize log score
        
        try:
            res_nm = minimize(
                nm_objective,
                seed.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res_nm.success or res_nm.fun < 1e8:
                pts = res_nm.x.reshape(13, 2)
                if np.all(np.isfinite(pts)):
                    s = _score(pts)
                    if s > best_score:
                        best_score = s
                        best_pts = pts.copy()
        except Exception:
            pass
        
        # Refine best found so far
        if best_pts is not None:
            refined_pts, refined_score = _coordinate_descent(
                best_pts, max_iters=100, step=0.02, deadline=deadline
            )
            if refined_score > best_score:
                best_score = refined_score
                best_pts = refined_pts.copy()

    if best_pts is None:
        best_pts = _triangular_lattice_seed_13()
        best_pts = best_pts - np.mean(best_pts, axis=0)
        scale = np.max(np.abs(best_pts))
        if scale > 0:
            best_pts = best_pts / scale * 0.5

    # Final polish
    if time.time() < deadline:
        best_pts, best_score = _coordinate_descent(
            best_pts, max_iters=200, step=0.01, deadline=deadline
        )
    
    return best_pts


# EVOLVE-BLOCK-END

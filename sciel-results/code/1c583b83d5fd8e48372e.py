# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull, Delaunay, Voronoi
import itertools
from scipy.optimize import minimize

_DEADLINE = time.time() + 590.0

def _min_triangle_area(points):
    n = len(points)
    if n < 3:
        return 0.0
    idx = np.array(list(itertools.combinations(range(n), 3)))
    if len(idx) == 0:
        return 0.0
    a = points[idx[:, 0]]
    b = points[idx[:, 1]]
    c = points[idx[:, 2]]
    areas = 0.5 * np.abs(a[:, 0] * (b[:, 1] - c[:, 1])
                         + b[:, 0] * (c[:, 1] - a[:, 1])
                         + c[:, 0] * (a[:, 1] - b[:, 1]))
    return float(np.min(areas))

def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0

def _score(points):
    if not np.all(np.isfinite(points)):
        return 0.0
    ha = _hull_area(points)
    if ha <= 0:
        return 0.0
    ma = _min_triangle_area(points)
    if ma <= 0:
        return 0.0
    return ma / ha

def _negative_ratio(points_flat):
    """Objective to minimize: -min_area/hull_area"""
    if time.time() > _DEADLINE:
        return 1e6
    pts = points_flat.reshape(-1, 2)
    try:
        ha = _hull_area(pts)
        if ha <= 0:
            return 1e6
        ma = _min_triangle_area(pts)
        if ma <= 0:
            return 1e6
        return -ma / ha
    except Exception:
        return 1e6

def _soft_min_objective(points_flat, beta=200.0):
    """Smooth surrogate: -soft_min(triangle_areas) / hull_area."""
    pts = points_flat.reshape(-1, 2)
    try:
        ha = _hull_area(pts)
        if ha <= 0:
            return 1e6
        n = len(pts)
        idx = np.array(list(itertools.combinations(range(n), 3)))
        a = pts[idx[:, 0]]
        b = pts[idx[:, 1]]
        c = pts[idx[:, 2]]
        areas = 0.5 * np.abs(a[:, 0] * (b[:, 1] - c[:, 1])
                             + b[:, 0] * (c[:, 1] - a[:, 1])
                             + c[:, 0] * (a[:, 1] - b[:, 1]))
        if np.any(areas <= 0):
            return 1e6
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min = -log_sum_exp / beta
        return -soft_min / ha
    except Exception:
        return 1e6

def _thomson_energy(points_flat, alpha=20.0, boundary_penalty=100.0):
    """Thomson-like repulsive energy with boundary penalty.
    Minimize this to spread points out (maximize min distance)."""
    pts = points_flat.reshape(-1, 2)
    n = len(pts)
    # Pairwise distances
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, 1.0)  # avoid division by zero
    # Repulsive energy: sum of 1/d^alpha
    energy = np.sum(1.0 / (dists ** alpha))
    # Boundary penalty: push points away from edges [0.05, 0.95]
    x_pen = np.sum(np.minimum(0, pts[:, 0] - 0.05)**2 + np.maximum(0, 0.95 - pts[:, 0])**2)
    y_pen = np.sum(np.minimum(0, pts[:, 1] - 0.05)**2 + np.maximum(0, 0.95 - pts[:, 1])**2)
    energy += boundary_penalty * (x_pen + y_pen)
    return energy

def _thomson_grad(points_flat, alpha=20.0, boundary_penalty=100.0):
    """Gradient of Thomson energy."""
    pts = points_flat.reshape(-1, 2)
    n = len(pts)
    grad = np.zeros_like(pts)
    # Repulsive force: for each pair (i,j), force on i is alpha/d^(alpha+1) * (p_i - p_j)
    for i in range(n):
        for j in range(i + 1, n):
            diff = pts[i] - pts[j]
            d = np.linalg.norm(diff)
            if d < 1e-10:
                continue
            force = alpha / (d ** (alpha + 1)) * diff
            grad[i] += force
            grad[j] -= force
    # Boundary gradient
    for i in range(n):
        for dim in range(2):
            val = pts[i, dim]
            if val < 0.05:
                grad[i, dim] += 2 * boundary_penalty * (val - 0.05)
            elif val > 0.95:
                grad[i, dim] += 2 * boundary_penalty * (val - 0.95)
    return grad.flatten()

def _optimize_thomson(points0, alpha=20.0, max_iter=500):
    """Optimize using Thomson energy with L-BFGS-B."""
    pts = points0.astype(float).flatten()
    n = len(points0)
    bounds = [(0.05, 0.95)] * (2 * n)
    try:
        result = minimize(
            _thomson_energy,
            pts,
            jac=_thomson_grad,
            args=(alpha,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'maxfun': max_iter * 5, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if result.fun < _thomson_energy(pts, alpha):
            return result.x.reshape(-1, 2)
    except Exception:
        pass
    return points0

def _optimize_soft_min(points0, betas=(50, 100, 200, 400, 600), max_iter_per_beta=200):
    """Optimize using progressive beta annealing with L-BFGS-B."""
    pts = points0.astype(float).flatten()
    n = len(points0)
    bounds = [(0.05, 0.95)] * (2 * n)
    
    for beta in betas:
        if time.time() > _DEADLINE - 30:
            break
        try:
            result = minimize(
                _soft_min_objective,
                pts,
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter_per_beta, 'maxfun': max_iter_per_beta * 3}
            )
            if result.fun < _soft_min_objective(pts, beta):
                pts = result.x
        except Exception:
            pass
    
    return pts.reshape(-1, 2)

def _polish(points0, max_iter=500):
    """Final Powell polish with tight tolerances."""
    pts = points0.astype(float).flatten()
    n = len(points0)
    bounds = [(0.05, 0.95)] * (2 * n)
    try:
        result = minimize(
            _negative_ratio,
            pts,
            method='Powell',
            bounds=bounds,
            options={'maxiter': max_iter, 'xtol': 1e-14, 'ftol': 1e-16, 'maxfev': max_iter * 10}
        )
        if result.fun < _negative_ratio(pts):
            return result.x.reshape(-1, 2)
    except Exception:
        pass
    return points0

def _nelder_mead_polish(points0, max_iter=2000):
    """High-precision Nelder-Mead polish on the exact objective."""
    pts = points0.astype(float).flatten()
    n = len(points0)
    try:
        result = minimize(
            _negative_ratio,
            pts,
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        if result.fun < _negative_ratio(pts):
            clipped = np.clip(result.x, 0.05, 0.95)
            return clipped.reshape(-1, 2)
    except Exception:
        pass
    return points0

def _coordinate_descent(points0, max_sweeps=200, step_sizes=None):
    """Cyclic coordinate descent on each point's x and y."""
    if step_sizes is None:
        step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    pts = points0.copy()
    best_pts = pts.copy()
    best_score = _score(pts)
    n = len(pts)
    
    for sweep in range(max_sweeps):
        if time.time() > _DEADLINE - 20:
            break
        improved = False
        for i in range(n):
            x0, y0 = pts[i]
            best_x, best_y = x0, y0
            best_local_score = best_score
            
            for step in step_sizes:
                for dx, dy in [(step,0),(-step,0),(0,step),(0,-step),
                               (step*0.707,step*0.707),(-step*0.707,step*0.707),
                               (step*0.707,-step*0.707),(-step*0.707,-step*0.707)]:
                    nx, ny = x0 + dx, y0 + dy
                    if nx < 0.05 or nx > 0.95 or ny < 0.05 or ny > 0.95:
                        continue
                    new_pts = pts.copy()
                    new_pts[i] = [nx, ny]
                    s = _score(new_pts)
                    if s > best_local_score + 1e-10:
                        best_local_score = s
                        best_x, best_y = nx, ny
                        improved = True
            pts[i] = [best_x, best_y]
        
        cur_score = _score(pts)
        if cur_score > best_score + 1e-10:
            best_score = cur_score
            best_pts = pts.copy()
        if not improved:
            break
    
    return best_pts, best_score

def _regular_polygon(n, radius=0.5, center=(0.5, 0.5)):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([center[0] + radius * np.cos(angles),
                            center[1] + radius * np.sin(angles)])

def _concentric_circles(n_inner, n_outer, r_inner=0.3, r_outer=1.0, center=(0.5, 0.5)):
    """Points on two concentric circles."""
    inner = _regular_polygon(n_inner, radius=r_inner, center=center)
    outer = _regular_polygon(n_outer, radius=r_outer, center=center)
    return np.vstack([inner, outer])

def _lloyd_iteration(points, n_iter=100):
    """
    Perform Lloyd's algorithm iterations for interior points.
    Hull points are fixed. Interior points move to Voronoi cell centroids.
    """
    points = points.copy()
    n = len(points)
    
    # Determine which points are on the convex hull
    try:
        hull = ConvexHull(points)
        hull_indices = set(hull.vertices)
    except Exception:
        return points
    
    interior_indices = [i for i in range(n) if i not in hull_indices]
    
    if not interior_indices:
        return points
    
    for _ in range(n_iter):
        if time.time() > _DEADLINE - 30:
            break
        
        # Compute Voronoi diagram
        try:
            vor = Voronoi(points)
        except Exception:
            break
        
        # Move interior points to their Voronoi cell centroids
        for i in interior_indices:
            # Find the Voronoi region corresponding to point i
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            
            if -1 in region:
                # Region is unbounded (shouldn't happen for interior points if hull is fixed)
                continue
                
            vertices = vor.vertices[region]
            if len(vertices) == 0:
                continue
                
            centroid = np.mean(vertices, axis=0)
            
            # Check if centroid is inside the convex hull of the current points
            # A simple check: is the centroid inside the triangle formed by the 3 nearest hull points?
            # Or better, just ensure it doesn't leave the bounding box of the hull
            hull_min = points[hull_indices].min(axis=0)
            hull_max = points[hull_indices].max(axis=0)
            
            if (centroid >= hull_min - 1e-9).all() and (centroid <= hull_max + 1e-9).all():
                points[i] = centroid
            else:
                # If centroid is outside, project it back to the nearest point on the hull boundary
                # For simplicity, we just keep the old position or move slightly towards center
                pass

    return points

def _lloyd_optimize(points0, lloyd_iters=50, polish_iters=1000):
    """
    Optimize using Lloyd's algorithm for interior points, then polish all points.
    """
    points = points0.copy()
    
    # Run Lloyd's iterations
    points = _lloyd_iteration(points, n_iter=lloyd_iters)
    
    # Polish with Nelder-Mead
    points = _nelder_mead_polish(points, max_iter=polish_iters)
    
    return points

def heilbronn_convex13() -> np.ndarray:
    n = 13
    rng = np.random.default_rng(seed=42)
    
    best_points = None
    best_score = -np.inf
    
    def _update(pts):
        nonlocal best_points, best_score
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
    
    # Multiple starting configurations - focus on structured topologies
    starts = []
    
    # 1. Regular 13-gon (primary)
    starts.append(_regular_polygon(13, radius=0.48, center=(0.5, 0.5)))
    
    # 2. Concentric circles: 6 inner + 7 outer (known-good topology from inspiration)
    for r_inner in [0.15, 0.2, 0.25, 0.3, 0.35]:
        starts.append(_concentric_circles(6, 7, r_inner=r_inner, r_outer=0.48, center=(0.5, 0.5)))
    
    # 3. Concentric circles: 5 inner + 8 outer
    starts.append(_concentric_circles(5, 8, r_inner=0.25, r_outer=0.48, center=(0.5, 0.5)))
    
    # 4. Concentric circles: 4 inner + 9 outer
    starts.append(_concentric_circles(4, 9, r_inner=0.2, r_outer=0.48, center=(0.5, 0.5)))
    
    # 5. Ring + interior: 9 outer, 4 inner
    outer9 = _regular_polygon(9, radius=0.48, center=(0.5, 0.5))
    angles4 = np.linspace(0, 2*np.pi, 4, endpoint=False)
    inner4 = np.column_stack([0.5 + 0.2*np.cos(angles4), 0.5 + 0.2*np.sin(angles4)])
    starts.append(np.vstack([outer9, inner4]))
    
    # 6. Ring + interior: 10 outer, 3 inner
    outer10 = _regular_polygon(10, radius=0.48, center=(0.5, 0.5))
    angles3 = np.linspace(0, 2*np.pi, 3, endpoint=False)
    inner3 = np.column_stack([0.5 + 0.18*np.cos(angles3), 0.5 + 0.18*np.sin(angles3)])
    starts.append(np.vstack([outer10, inner3]))
    
    # 7. All 13 on hull (regular)
    starts.append(_regular_polygon(13, radius=0.5, center=(0.5, 0.5)))
    
    # 8. Two random starts
    for _ in range(2):
        starts.append(rng.uniform(0.1, 0.9, (13, 2)))
    
    # For each start, try multiple optimization strategies
    for start_idx, start in enumerate(starts):
        if time.time() > _DEADLINE - 30:
            break
        
        # Strategy A: Thomson energy optimization (NEW - smooth, differentiable)
        if time.time() < _DEADLINE - 60:
            thomson_pts = _optimize_thomson(start, alpha=20, max_iter=300)
            _update(thomson_pts)
        
        # Strategy B: Soft-min with progressive beta annealing
        if time.time() < _DEADLINE - 60:
            sm_pts = _optimize_soft_min(start, betas=(50, 100, 200, 400, 600), max_iter_per_beta=150)
            _update(sm_pts)
        
        # Strategy C: Powell polish on best of Thomson/soft-min
        if time.time() < _DEADLINE - 40:
            # Use the better of thomson_pts and sm_pts
            candidate = thomson_pts if _score(thomson_pts) > _score(sm_pts) else sm_pts
            polish_pts = _polish(candidate, max_iter=300)
            _update(polish_pts)
            
        # Strategy D: Lloyd's Algorithm (New Structural Change)
        if time.time() < _DEADLINE - 40:
            lloyd_pts = _lloyd_optimize(start, lloyd_iters=50, polish_iters=1000)
            _update(lloyd_pts)
            
        # Strategy E: Nelder-Mead high-precision polish
        if time.time() < _DEADLINE - 30:
            # Pick the best so far from this start's strategies
            candidates = [thomson_pts, sm_pts, polish_pts, lloyd_pts]
            best_candidate = max(candidates, key=lambda p: _score(p))
            nm_pts = _nelder_mead_polish(best_candidate, max_iter=1500)
            _update(nm_pts)
        
        # Strategy F: Coordinate descent for fine-tuning
        if time.time() < _DEADLINE - 25:
            cd_pts, _ = _coordinate_descent(nm_pts, max_sweeps=100)
            _update(cd_pts)
    
    # Final refinement pass on the best solution
    if best_points is not None and best_score > 0.01 and time.time() < _DEADLINE - 20:
        final_pts = _nelder_mead_polish(best_points, max_iter=2000)
        _update(final_pts)
    
    if best_points is None:
        best_points = _regular_polygon(13, radius=0.48, center=(0.5, 0.5))
    
    if not np.all(np.isfinite(best_points)):
        best_points = _regular_polygon(13, radius=0.48, center=(0.5, 0.5))
    
    return best_points

# EVOLVE-BLOCK-END

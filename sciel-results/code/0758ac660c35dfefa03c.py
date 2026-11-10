# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull, Delaunay, Voronoi
from scipy.optimize import minimize
import itertools

_DEADLINE = time.time() + 625.0

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
    ha = _hull_area(points)
    if ha <= 0:
        return 0.0
    ma = _min_triangle_area(points)
    if ma <= 0:
        return 0.0
    return ma / ha

def _soft_min_objective(pts_flat, fixed_pts, alpha=100):
    """
    Objective for optimizing interior points while keeping hull points fixed.
    pts_flat: flattened coordinates of ALL points, but only interior ones are free?
    No, simpler: pass interior points only, and fixed hull points separately.
    """
    n_interior = len(pts_flat) // 2
    interior_pts = pts_flat.reshape(-1, 2)
    all_pts = np.vstack([fixed_pts, interior_pts])
    
    try:
        ha = _hull_area(all_pts)
        if ha <= 0:
            return 1e6
        
        n = len(all_pts)
        idx = np.array(list(itertools.combinations(range(n), 3)))
        a = all_pts[idx[:, 0]]
        b = all_pts[idx[:, 1]]
        c = all_pts[idx[:, 2]]
        areas = 0.5 * np.abs(a[:, 0] * (b[:, 1] - c[:, 1])
                             + b[:, 0] * (c[:, 1] - a[:, 1])
                             + c[:, 0] * (a[:, 1] - b[:, 1]))
        
        # Soft minimum
        max_neg = np.max(-alpha * areas)
        log_sum_exp = np.log(np.sum(np.exp(-alpha * areas - max_neg))) + max_neg
        soft_min_area = -log_sum_exp / alpha
        
        return -(soft_min_area / ha)
    except Exception:
        return 1e6

def _lloyd_interior_only(points, n_iter=50, step_size=0.5):
    """Lloyd's algorithm: only move interior points toward Voronoi centroids."""
    pts = points.copy()
    n = len(pts)
    
    try:
        hull = ConvexHull(pts)
        hull_idx = set(hull.vertices)
    except Exception:
        return pts
    
    for it in range(n_iter):
        if time.time() > _DEADLINE - 30:
            break
        
        # Recompute Voronoi once per iteration
        try:
            vor = Voronoi(pts)
        except Exception:
            break
        
        for i in range(n):
            if i in hull_idx:
                continue
            
            region = vor.regions[i]
            if -1 in region or len(region) == 0:
                # Fallback to Delaunay neighbors
                try:
                    tri = Delaunay(pts)
                    mask = np.any(tri.simplices == i, axis=1)
                    if np.any(mask):
                        neighbors = np.unique(tri.simplices[mask].reshape(-1))
                        neighbors = neighbors[neighbors != i]
                        if len(neighbors) > 0:
                            centroid = np.mean(pts[neighbors], axis=0)
                            pts[i] = pts[i] + step_size * (centroid - pts[i])
                            pts[i] = np.clip(pts[i], 0.01, 0.99)
                except Exception:
                    pass
                continue
            
            vertices = vor.vertices[region]
            if len(vertices) < 3:
                continue
            
            x = vertices[:, 0]
            y = vertices[:, 1]
            x2 = np.roll(x, -1)
            y2 = np.roll(y, -1)
            cross = x * y2 - x2 * y
            area = 0.5 * np.sum(cross)
            
            if abs(area) < 1e-12:
                continue
            
            cx = np.sum((x + x2) * cross) / (6.0 * area)
            cy = np.sum((y + y2) * cross) / (6.0 * area)
            centroid = np.array([cx, cy])
            
            pts[i] = pts[i] + step_size * (centroid - pts[i])
            pts[i] = np.clip(pts[i], 0.01, 0.99)
        
        # Recompute hull
        try:
            hull = ConvexHull(pts)
            hull_idx = set(hull.vertices)
        except Exception:
            break
    
    return pts

def _optimize_interior(fixed_hull, interior_init, max_iter=200):
    """Optimize interior points using L-BFGS with soft-min."""
    if len(interior_init) == 0:
        return interior_init
    
    def objective(pts_flat):
        return _soft_min_objective(pts_flat, fixed_hull, alpha=100)
    
    try:
        result = minimize(
            objective,
            interior_init.flatten(),
            method='L-BFGS-B',
            bounds=[(0.01, 0.99)] * (2 * len(interior_init)),
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        return result.x.reshape(-1, 2)
    except Exception:
        return interior_init

def _coordinate_descent_polish(points0, max_sweeps=20):
    """Fine-grained coordinate descent polish."""
    pts = points0.copy()
    best_pts = pts.copy()
    best_score = _score(pts)
    n = len(pts)
    
    step_sizes = [0.002, 0.001, 0.0005, 0.0002]
    
    for sweep in range(max_sweeps):
        if time.time() > _DEADLINE - 15:
            break
        
        improved = False
        for i in range(n):
            x0, y0 = pts[i]
            best_x, best_y = x0, y0
            best_local = best_score
            
            for step in step_sizes:
                for dx, dy in [(step, 0), (-step, 0), (0, step), (0, -step),
                               (step*0.707, step*0.707), (-step*0.707, step*0.707),
                               (step*0.707, -step*0.707), (-step*0.707, -step*0.707)]:
                    nx, ny = x0 + dx, y0 + dy
                    if nx < 0.01 or nx > 0.99 or ny < 0.01 or ny > 0.99:
                        continue
                    new_pts = pts.copy()
                    new_pts[i] = [nx, ny]
                    s = _score(new_pts)
                    if s > best_local + 1e-12:
                        best_local = s
                        best_x, best_y = nx, ny
                        improved = True
            
            pts[i] = [best_x, best_y]
        
        cur = _score(pts)
        if cur > best_score + 1e-12:
            best_score = cur
            best_pts = pts.copy()
        
        if not improved:
            break
    
    return best_pts, best_score

def heilbronn_convex13() -> np.ndarray:
    n = 13
    best_points = None
    best_score = -np.inf
    
    # Strategy: Fix hull to regular k-gon, optimize interior 13-k points.
    # Try k from 12 down to 6.
    for k in range(12, 5, -1):
        if time.time() > _DEADLINE - 60:
            break
            
        # Create regular k-gon hull
        angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
        radius = 0.45
        center = (0.5, 0.5)
        hull_pts = np.column_stack([center[0] + radius * np.cos(angles),
                                    center[1] + radius * np.sin(angles)])
        
        n_interior = n - k
        if n_interior == 0:
            continue # At least 3 points needed, but we have 13 total.
            
        # Initialize interior points: place them in a smaller regular grid or random inside
        # For simplicity, start with a small regular polygon of size n_interior at the center
        if n_interior == 1:
            interior_init = np.array([[0.5, 0.5]])
        elif n_interior == 2:
            interior_init = np.array([[0.45, 0.5], [0.55, 0.5]])
        else:
            angles_int = np.linspace(0, 2 * np.pi, n_interior, endpoint=False)
            radius_int = 0.15
            interior_init = np.column_stack([center[0] + radius_int * np.cos(angles_int),
                                             center[1] + radius_int * np.sin(angles_int)])
        
        # Combine
        init_pts = np.vstack([hull_pts, interior_init])
        
        # 1. Lloyd relaxation on interior points (keeping hull fixed? 
        # The standard Lloyd moves all points. We need a version that keeps hull fixed.
        # Let's modify _lloyd_interior_only to accept a mask of fixed points.
        # For now, we will just run Lloyd on the whole set but then project hull back?
        # No, that's messy. Let's just use the L-BFGS optimizer directly on the interior points.
        # But L-BFGS might get stuck. Let's do a simple manual Lloyd for interior points only.
        
        # Custom Lloyd for interior points with fixed hull
        pts = init_pts.copy()
        fixed_indices = set(range(k))
        
        for it in range(30):
            if time.time() > _DEADLINE - 30:
                break
            try:
                vor = Voronoi(pts)
            except Exception:
                break
                
            for i in range(k, n):
                region = vor.regions[i]
                if -1 in region or len(region) == 0:
                    # Fallback: average of Delaunay neighbors (excluding fixed?)
                    try:
                        tri = Delaunay(pts)
                        mask = np.any(tri.simplices == i, axis=1)
                        if np.any(mask):
                            neighbors = np.unique(tri.simplices[mask].reshape(-1))
                            neighbors = neighbors[neighbors != i]
                            if len(neighbors) > 0:
                                centroid = np.mean(pts[neighbors], axis=0)
                                pts[i] = pts[i] + 0.5 * (centroid - pts[i])
                                pts[i] = np.clip(pts[i], 0.01, 0.99)
                    except Exception:
                        pass
                    continue
                
                vertices = vor.vertices[region]
                if len(vertices) < 3:
                    continue
                
                x = vertices[:, 0]
                y = vertices[:, 1]
                x2 = np.roll(x, -1)
                y2 = np.roll(y, -1)
                cross = x * y2 - x2 * y
                area = 0.5 * np.sum(cross)
                
                if abs(area) < 1e-12:
                    continue
                
                cx = np.sum((x + x2) * cross) / (6.0 * area)
                cy = np.sum((y + y2) * cross) / (6.0 * area)
                centroid = np.array([cx, cy])
                
                pts[i] = pts[i] + 0.5 * (centroid - pts[i])
                pts[i] = np.clip(pts[i], 0.01, 0.99)
                
            # Ensure hull points stay fixed (they shouldn't move in this loop, but just in case)
            pts[:k] = hull_pts
            
        # 2. L-BFGS optimization on interior points
        if time.time() < _DEADLINE - 20:
            interior_optimized = _optimize_interior(hull_pts, pts[k:], max_iter=100)
            candidate_pts = np.vstack([hull_pts, interior_optimized])
            s = _score(candidate_pts)
            if s > best_score:
                best_score = s
                best_points = candidate_pts.copy()
                
        # 3. Coordinate Descent Polish
        if best_points is not None and time.time() < _DEADLINE - 10:
            polished_pts, polished_s = _coordinate_descent_polish(best_points, max_sweeps=10)
            if polished_s > best_score:
                best_score = polished_s
                best_points = polished_pts

    # Fallback to original best if nothing found
    if best_points is None or not np.all(np.isfinite(best_points)):
        # Regular 12-gon + center
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        radius = 0.45
        center = (0.5, 0.5)
        hull_pts = np.column_stack([center[0] + radius * np.cos(angles),
                                    center[1] + radius * np.sin(angles)])
        best_points = np.vstack([hull_pts, np.array([center])])
    
    return best_points

# EVOLVE-BLOCK-END

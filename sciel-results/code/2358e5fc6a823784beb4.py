import numpy as np
from scipy.spatial import ConvexHull, Voronoi
from scipy.optimize import minimize
import time

N = 14

def _compute_hull_area(pts):
    """Compute area of convex hull of points."""
    if len(pts) < 3:
        return 0.0
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1e-10

def _min_triangle_area_raw(pts):
    """Compute the minimum area of all triangles formed by pts."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        p1 = pts[i]
        for j in range(i + 1, n):
            p2 = pts[j]
            v1 = p2 - p1
            for k in range(j + 1, n):
                p3 = pts[k]
                v2 = p3 - p1
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def _calculate_score(pts):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _compute_hull_area(pts)
    if hull_area < 1e-9:
        return 0.0
    min_area = _min_triangle_area_raw(pts)
    return min_area / hull_area

def _voronoi_relax(pts, n_iters=50, disk_radius=1.0):
    """
    Voronoi relaxation: iteratively move each point to the centroid of its
    Voronoi cell clipped to the unit disk.
    """
    pts = pts.copy()
    for _ in range(n_iters):
        # Compute Voronoi diagram
        try:
            vor = Voronoi(pts)
        except Exception:
            break
        
        # For each point, compute centroid of its Voronoi cell
        # clipped to the unit disk
        new_pts = np.zeros_like(pts)
        for i in range(N):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            
            if len(region) == 0:
                # Point is on the boundary of the Voronoi diagram
                # Keep it where it is
                new_pts[i] = pts[i]
                continue
            
            # Get the vertices of this Voronoi cell
            vertices = vor.vertices[region]
            
            if len(vertices) == 0:
                new_pts[i] = pts[i]
                continue
            
            # Compute centroid of the polygon formed by these vertices
            # and clip to the unit disk
            # Simple approach: average the vertices (approximate centroid)
            centroid = vertices.mean(axis=0)
            
            # Clip to unit disk
            dist = np.linalg.norm(centroid)
            if dist > disk_radius:
                centroid = centroid / dist * disk_radius
            
            new_pts[i] = centroid
        
        # Blend to avoid instability
        pts = 0.5 * pts + 0.5 * new_pts
    
    return pts

def _soft_min_triangle_area(pts, beta=50.0):
    """Compute a soft minimum of triangle areas for smooth optimization."""
    n = len(pts)
    areas = []
    for i in range(n):
        p1 = pts[i]
        for j in range(i + 1, n):
            p2 = pts[j]
            v1 = p2 - p1
            for k in range(j + 1, n):
                p3 = pts[k]
                v2 = p3 - p1
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                areas.append(0.5 * abs(cross))
    
    areas = np.array(areas)
    a_min = np.min(areas)
    soft_min_val = -a_min - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (areas - a_min))))
    return soft_min_val

def _objective_voronoi(params):
    """
    Objective function for optimization.
    params: flattened array of 28 coordinates (14 points in 2D)
    """
    pts = params.reshape(N, 2)
    
    # Center the points
    pts = pts - pts.mean(axis=0)
    
    # Check for degeneracy
    if np.allclose(pts, 0):
        return 1e10
    
    # Scale to unit hull area
    hull_area = _compute_hull_area(pts)
    if hull_area < 1e-9:
        return 1e10
    
    pts = pts / np.sqrt(hull_area)
    
    # Apply Voronoi relaxation to improve the configuration
    relaxed_pts = _voronoi_relax(pts, n_iters=5)
    
    # Compute soft minimum triangle area
    soft_min = _soft_min_triangle_area(relaxed_pts, beta=50.0)
    
    return -soft_min

def _nelder_mead_polish(pts, max_iter=200, deadline=None):
    """
    Polish points using Nelder-Mead on the raw score.
    """
    def objective(params):
        p = params.reshape(N, 2)
        p = p - p.mean(axis=0)
        hull_area = _compute_hull_area(p)
        if hull_area < 1e-9:
            return 1e10
        p = p / np.sqrt(hull_area)
        min_area = _min_triangle_area_raw(p)
        return -min_area
    
    result = minimize(
        objective,
        pts.flatten(),
        method='Nelder-Mead',
        options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-10}
    )
    
    if result.success:
        return result.x.reshape(N, 2)
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s limit
    
    # Initial guess: Regular 14-gon
    angles = 2 * np.pi * np.arange(N) / N
    init_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    init_pts -= init_pts.mean(axis=0)
    
    best_score = -1.0
    best_pts = init_pts.copy()
    
    # Phase 1: Voronoi relaxation from multiple initial conditions
    initial_configs = []
    
    # Regular 14-gon
    initial_configs.append(init_pts.copy())
    
    # Two-ring configuration (7 inner, 7 outer)
    inner_angles = 2 * np.pi * np.arange(7) / 7
    outer_angles = inner_angles + np.pi / 7  # Offset by half step
    inner_pts = np.column_stack([0.5 * np.cos(inner_angles), 0.5 * np.sin(inner_angles)])
    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    two_ring = np.vstack([inner_pts, outer_pts])
    two_ring -= two_ring.mean(axis=0)
    initial_configs.append(two_ring)
    
    # Perturbed regular 14-gon
    np.random.seed(42)
    perturbed = init_pts.copy()
    perturbed += 0.1 * np.random.randn(N, 2)
    perturbed -= perturbed.mean(axis=0)
    initial_configs.append(perturbed)
    
    for init in initial_configs:
        if time.time() > deadline:
            break
        
        # Apply Voronoi relaxation
        relaxed = _voronoi_relax(init, n_iters=100, disk_radius=1.0)
        
        # Normalize to unit hull area
        relaxed = relaxed - relaxed.mean(axis=0)
        hull_area = _compute_hull_area(relaxed)
        if hull_area > 1e-9:
            relaxed = relaxed / np.sqrt(hull_area)
        
        score = _calculate_score(relaxed)
        if score > best_score:
            best_score = score
            best_pts = relaxed.copy()
    
    # Phase 2: L-BFGS-B optimization with soft-min objective
    if time.time() < deadline - 30:
        try:
            res = minimize(
                _objective_voronoi,
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success:
                candidate = res.x.reshape(N, 2)
                candidate = candidate - candidate.mean(axis=0)
                hull_area = _compute_hull_area(candidate)
                if hull_area > 1e-9:
                    candidate = candidate / np.sqrt(hull_area)
                    score = _calculate_score(candidate)
                    if score > best_score:
                        best_score = score
                        best_pts = candidate.copy()
        except Exception:
            pass
    
    # Phase 3: Nelder-Mead polish on raw score
    if time.time() < deadline - 15:
        try:
            polished = _nelder_mead_polish(best_pts, max_iter=150, deadline=deadline)
            polished = polished - polished.mean(axis=0)
            hull_area = _compute_hull_area(polished)
            if hull_area > 1e-9:
                polished = polished / np.sqrt(hull_area)
                score = _calculate_score(polished)
                if score > best_score:
                    best_score = score
                    best_pts = polished.copy()
        except Exception:
            pass
    
    # Final normalization
    final_pts = best_pts - best_pts.mean(axis=0)
    final_area = _compute_hull_area(final_pts)
    if final_area > 1e-9:
        final_pts = final_pts / np.sqrt(final_area)
    
    return final_pts

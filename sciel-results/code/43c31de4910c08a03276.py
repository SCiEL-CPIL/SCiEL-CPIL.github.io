"""SDP relaxation approach for Heilbronn triangles."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools
import time

def _triangle_area(p1, p2, p3):
    """Compute area of triangle given three points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by three points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area

def _is_inside_triangle(p, tri):
    """Check if point p is inside or on the boundary of triangle tri (3 points)."""
    p = np.array(p)
    a, b, c = [np.array(v) for v in tri]
    
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    
    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)
    
    has_neg = (d1 < -1e-9) or (d2 < -1e-9) or (d3 < -1e-9)
    has_pos = (d1 > 1e-9) or (d2 > 1e-9) or (d3 > 1e-9)
    
    return not (has_neg and has_pos)

def _project_to_triangle(p, tri):
    """Project point p onto the triangle tri."""
    h = np.sqrt(3) / 2
    tri_arr = np.array(tri)
    a, b, c = tri_arr
    
    v0 = b - a
    v1 = c - a
    v2 = p - a
    
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    
    denom = d00 * d11 - d01 * d01
    if denom < 1e-12:
        return a.copy()
    
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    
    if v >= -1e-9 and w >= -1e-9 and v + w <= 1 + 1e-9:
        return a + v * v0 + w * v1
    
    va = d20
    vb = d20 - d00
    if va <= 0:
        if vb >= 0:
            return a
        else:
            return b
    va = d21
    vb = d21 - d11
    if va <= 0:
        if vb >= 0:
            return a
        else:
            return c
    vc = d00 * d21 - d01 * d20
    vb = d11 * d20 - d01 * d21
    if vb <= 0:
        if vc >= 0:
            return b
        else:
            return c
    
    edge = vc - vb
    if edge <= 0:
        alpha = vb / edge
        return b + alpha * (c - b)
    else:
        alpha = vc / edge
        return c + alpha * (b - c)

def _sdp_distance_max(points_init, max_time=55):
    """
    Maximize minimum pairwise distance using coordinate descent with SDP-inspired
    initialization. The key insight: min triangle area is bounded by min pairwise distance,
    so maximizing min distance gives a good lower bound and initialization.
    """
    start_time = time.time()
    n = N
    h = np.sqrt(3) / 2
    tri = [(0, 0), (1, 0), (0.5, h)]
    
    def _min_pairwise_dist_sq(pts):
        min_d2 = float('inf')
        for i in range(n):
            for j in range(i + 1, n):
                d2 = np.sum((pts[i] - pts[j]) ** 2)
                if d2 < min_d2:
                    min_d2 = d2
        return min_d2
    
    best_points = None
    best_dist_sq = -float('inf')
    
    # Multiple initialization strategies
    np.random.seed(2024)
    
    for trial in range(25):
        # Strategy: triangular lattice with random perturbations
        points = np.zeros((n, 2))
        
        # Place in a hexagonal/triangular pattern
        positions = [
            (0.1, 0.05), (0.3, 0.05), (0.5, 0.05), (0.7, 0.05), (0.9, 0.05),
            (0.2, 0.2), (0.4, 0.2), (0.6, 0.2), (0.8, 0.2),
            (0.35, 0.38), (0.65, 0.38)
        ]
        
        for i, (x, y) in enumerate(positions):
            points[i] = [x + np.random.normal(0, 0.02), y + np.random.normal(0, 0.02)]
        
        # Project to triangle
        for i in range(n):
            if not _is_inside_triangle(points[i], tri):
                points[i] = _project_to_triangle(points[i], tri)
                centroid = np.array([0.5, h/3])
                points[i] = 0.95 * points[i] + 0.05 * centroid
        
        # Optimize: maximize min pairwise distance
        def _obj(params):
            pts = params.reshape((n, 2))
            # Penalty for out-of-triangle
            pen = 0.0
            for p in pts:
                if not _is_inside_triangle(p, tri):
                    proj = _project_to_triangle(p, tri)
                    pen += np.linalg.norm(p - proj) ** 2 * 100
            if pen > 1e-6:
                return pen
            return -_min_pairwise_dist_sq(pts)
        
        bounds = [(0, 1), (0, h)] * n
        
        try:
            result = minimize(
                _obj, points.flatten(),
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 800, 'ftol': 1e-12}
            )
            opt_pts = result.x.reshape((n, 2))
            
            for i in range(n):
                if not _is_inside_triangle(opt_pts[i], tri):
                    opt_pts[i] = _project_to_triangle(opt_pts[i], tri)
            
            valid = all(_is_inside_triangle(p, tri) for p in opt_pts)
            if valid:
                d2 = _min_pairwise_dist_sq(opt_pts)
                if d2 > best_dist_sq:
                    best_dist_sq = d2
                    best_points = opt_pts.copy()
        except Exception:
            continue
        
        if time.time() - start_time > max_time:
            break
    
    return best_points, best_dist_sq

def _refine_nm(points, max_iter=2000):
    """Refine points using Nelder-Mead on the true min triangle area objective."""
    n = N
    h = np.sqrt(3) / 2
    tri = [(0, 0), (1, 0), (0.5, h)]
    
    def _objective_nm(params):
        pts = params.reshape((n, 2))
        penalty = 0.0
        for p in pts:
            if not _is_inside_triangle(p, tri):
                proj = _project_to_triangle(p, tri)
                penalty += np.linalg.norm(p - proj) * 10.0
        if penalty > 1e-8:
            return penalty
        return -_min_triangle_area(pts)
    
    try:
        result = minimize(
            _objective_nm, points.flatten(),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-9, 'fatol': 1e-11}
        )
        opt_points = result.x.reshape((n, 2))
        
        for i in range(n):
            if not _is_inside_triangle(opt_points[i], tri):
                opt_points[i] = _project_to_triangle(opt_points[i], tri)
        
        valid = all(_is_inside_triangle(p, tri) for p in opt_points)
        if valid:
            return opt_points
    except Exception:
        pass
    
    return points

def construct():
    start_time = time.time()
    
    h = np.sqrt(3) / 2
    tri = [(0, 0), (1, 0), (0.5, h)]
    
    # Phase 1: Maximize min pairwise distance (SDP-inspired surrogate)
    sdp_points, _ = _sdp_distance_max(None, max_time=55)
    
    if sdp_points is not None:
        best_points = sdp_points
        best_score = _min_triangle_area(sdp_points)
    else:
        best_points = np.array([
            [0.5, 0.05], [0.25, 0.15], [0.75, 0.15], [0.5, 0.25],
            [0.15, 0.35], [0.85, 0.35], [0.35, 0.45], [0.65, 0.45],
            [0.5, 0.55], [0.25, 0.5], [0.75, 0.5]
        ])
        best_score = _min_triangle_area(best_points)
    
    # Phase 2: Refine with Nelder-Mead on true objective
    if time.time() - start_time < 100:
        refined = _refine_nm(best_points, max_iter=3000)
        refined_score = _min_triangle_area(refined)
        if refined_score > best_score:
            best_points = refined
            best_score = refined_score
    
    # Phase 3: Multi-start local refinement
    if time.time() - start_time < 112:
        np.random.seed(42)
        for _ in range(10):
            perturbed = best_points + np.random.normal(0, 0.004, (N, 2))
            refined2 = _refine_nm(perturbed, max_iter=800)
            score2 = _min_triangle_area(refined2)
            if score2 > best_score:
                best_points = refined2
                best_score = score2
    
    return best_points
# EVOLVE-BLOCK-END

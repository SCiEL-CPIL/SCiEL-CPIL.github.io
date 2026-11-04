"""Heilbronn convex regions n=14 — differential evolution + soft-min + bottleneck polish."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize

N = 14

def _min_triangle_area(points):
    """Compute minimum triangle area over all triples."""
    n = len(points)
    if n < 3:
        return 0.0
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        n = len(points)
        if n < 3:
            return 0.0
        cx, cy = points[:, 0].mean(), points[:, 1].mean()
        angles = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
        order = np.argsort(angles)
        hpts = points[order]
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += hpts[i][0] * hpts[j][1] - hpts[j][0] * hpts[i][1]
        return abs(area) * 0.5

def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _hull_area(points)
    if hull_area < 1e-9:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _soft_min_score(points, temperature=0.05):
    """Differentiable soft-min approximation of score for DE optimization."""
    n = len(points)
    if n < 3:
        return 0.0
    
    # Compute all triangle areas
    areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
                areas.append(area)
    
    areas = np.array(areas)
    if len(areas) == 0:
        return 0.0
    
    # Soft min
    min_area_soft = temperature * np.log(np.sum(np.exp(-areas / temperature)))
    
    hull_area = _hull_area(points)
    if hull_area < 1e-9:
        return 0.0
    
    return min_area_soft / hull_area

def _two_temp_soft_min_score(points, temp1=0.02, temp2=0.1):
    """Two-temperature soft min for better gradient information."""
    n = len(points)
    if n < 3:
        return 0.0
    
    areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
                areas.append(area)
    
    areas = np.array(areas)
    if len(areas) == 0:
        return 0.0
    
    # Two-temperature soft min
    soft_min_1 = temp1 * np.log(np.sum(np.exp(-areas / temp1)))
    soft_min_2 = temp2 * np.log(np.sum(np.exp(-areas / temp2)))
    
    # Blend
    min_area_soft = 0.5 * soft_min_1 + 0.5 * soft_min_2
    
    hull_area = _hull_area(points)
    if hull_area < 1e-9:
        return 0.0
    
    return min_area_soft / hull_area

def _generate_two_ring(r_inner, theta_offset, n_outer=7, n_inner=7):
    """Generate points on two concentric rings."""
    angles_out = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    angles_in = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + theta_offset
    
    x_out = np.cos(angles_out)
    y_out = np.sin(angles_out)
    x_in = r_inner * np.cos(angles_in)
    y_in = r_inner * np.sin(angles_in)
    
    return np.vstack([
        np.column_stack((x_out, y_out)),
        np.column_stack((x_in, y_in))
    ])

def _generate_three_ring(r1, r2, theta1, theta2, n1=5, n2=4, n3=5):
    """Generate points on three concentric rings."""
    angles1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    angles2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + theta1
    angles3 = np.linspace(0, 2 * np.pi, n3, endpoint=False) + theta2
    
    x1 = np.cos(angles1)
    y1 = np.sin(angles1)
    x2 = r1 * np.cos(angles2)
    y2 = r1 * np.sin(angles2)
    x3 = r2 * np.cos(angles3)
    y3 = r2 * np.sin(angles3)
    
    return np.vstack([
        np.column_stack((x1, y1)),
        np.column_stack((x2, y2)),
        np.column_stack((x3, y3))
    ])

def _bottleneck_polish(points, time_limit=15.0):
    """Polish by focusing on the worst (smallest area) triangles."""
    start_time = time.time()
    
    # Find the worst triangles
    n = len(points)
    worst_triangles = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
                worst_triangles.append((area, i, j, k))
    
    worst_triangles.sort()
    worst_10 = worst_triangles[:10]
    
    best_score = _score(points)
    best_pts = points.copy()
    
    # For each worst triangle, try to increase its area by moving points
    for area, i, j, k in worst_10:
        if time.time() > start_time + time_limit:
            break
        
        # Try small perturbations of points i, j, k
        for di in [-0.02, 0.02]:
            for dj in [-0.02, 0.02]:
                for dk in [-0.02, 0.02]:
                    if time.time() > start_time + time_limit:
                        break
                    new_pts = best_pts.copy()
                    new_pts[i, 0] += di
                    new_pts[j, 1] += dj
                    new_pts[k, 0] += dk
                    s = _score(new_pts)
                    if s > best_score:
                        best_score = s
                        best_pts = new_pts.copy()
    
    # Final Nelder-Mead polish
    if time.time() < start_time + time_limit:
        def obj(x):
            pts = x.reshape(-1, 2)
            return -_score(pts)
        
        try:
            res = minimize(
                obj, best_pts.flatten(), method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
            )
            if time.time() < start_time + time_limit:
                if res.fun < -best_score:
                    best_score = -res.fun
                    best_pts = res.x.reshape(-1, 2).copy()
        except Exception:
            pass
    
    return best_score, best_pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_score = 0.0
    best_points = None
    
    # Strategy 1: Differential evolution over full 2D coordinates
    # Use soft-min score for differentiability
    def de_obj(x):
        pts = x.reshape(-1, 2)
        # Add small penalty if points are too close or outside unit circle
        norms = np.linalg.norm(pts, axis=1)
        if np.any(norms > 1.05):
            return 10.0
        return -_two_temp_soft_min_score(pts, temp1=0.03, temp2=0.08)
    
    # Bounds: points in [-1, 1] x [-1, 1]
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    try:
        if time.time() < deadline - 60:
            de_result = differential_evolution(
                de_obj,
                bounds=bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-4,
                polish=False,
                init='latinhypercube'
            )
            if time.time() < deadline - 50:
                de_pts = de_result.x.reshape(-1, 2)
                s = _score(de_pts)
                if s > best_score:
                    best_score = s
                    best_points = de_pts.copy()
    except Exception:
        pass
    
    # Strategy 2: Two-ring configurations with DE on parameters
    if time.time() < deadline - 40:
        def two_ring_obj(params):
            r_inner, theta = params
            if r_inner <= 0.05 or r_inner > 0.98:
                return 10.0
            pts = _generate_two_ring(r_inner, theta)
            return -_two_temp_soft_min_score(pts, temp1=0.03, temp2=0.08)
        
        bounds_2ring = [(0.1, 0.95), (0, 2 * np.pi / 7)]
        
        try:
            de_result = differential_evolution(
                two_ring_obj,
                bounds=bounds_2ring,
                seed=43,
                maxiter=30,
                popsize=10,
                tol=1e-4,
                polish=False
            )
            if time.time() < deadline - 35:
                r_opt, theta_opt = de_result.x
                pts = _generate_two_ring(r_opt, theta_opt)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
        except Exception:
            pass
    
    # Strategy 3: Three-ring configurations
    if time.time() < deadline - 30:
        def three_ring_obj(params):
            r1, r2, theta1, theta2 = params
            if r1 <= 0.05 or r1 > 0.7 or r2 <= 0.05 or r2 > 0.95 or r1 >= r2:
                return 10.0
            pts = _generate_three_ring(r1, r2, theta1, theta2)
            return -_two_temp_soft_min_score(pts, temp1=0.03, temp2=0.08)
        
        bounds_3ring = [(0.1, 0.65), (0.4, 0.95), (0, 2 * np.pi / 4), (0, 2 * np.pi / 5)]
        
        try:
            de_result = differential_evolution(
                three_ring_obj,
                bounds=bounds_3ring,
                seed=44,
                maxiter=25,
                popsize=8,
                tol=1e-4,
                polish=False
            )
            if time.time() < deadline - 25:
                r1_opt, r2_opt, t1_opt, t2_opt = de_result.x
                pts = _generate_three_ring(r1_opt, r2_opt, t1_opt, t2_opt)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
        except Exception:
            pass
    
    # Strategy 4: Regular polygon baseline
    if time.time() < deadline - 20:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
    
    # Strategy 5: Bottleneck polish on best found
    if best_points is not None and time.time() < deadline - 10:
        s, pts_polished = _bottleneck_polish(best_points, time_limit=15.0)
        if s > best_score:
            best_score = s
            best_points = pts_polished.copy()
    
    # Fallback
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    final_score = _score(best_points)
    if final_score <= 0:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points

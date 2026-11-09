import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

def get_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except:
        return 0.0

    min_area = float('inf')
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
                if area < min_area:
                    min_area = area
    
    if min_area == float('inf'):
        return 0.0
        
    return min_area / hull_area

def _regular_polygon(n, radius=1.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))

def _make_12gon_center():
    polygon = _regular_polygon(12, radius=1.0)
    center = np.array([[0.0, 0.0]])
    return np.vstack([polygon, center])

def _optimize_from_start(start_points, deadline, maxiter=3000):
    if time.time() >= deadline:
        return start_points, get_score(start_points)
    try:
        res = minimize(
            lambda flat: -get_score(flat.reshape(N, 2)),
            start_points.flatten(),
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        if res.fun < -get_score(start_points):
            return res.x.reshape(N, 2), -res.fun
        return start_points, get_score(start_points)
    except:
        return start_points, get_score(start_points)

def _push_apart_refinement(points, deadline, iterations=50):
    pts = points.copy()
    best_score = get_score(pts)
    step = 0.005
    for it in range(iterations):
        if time.time() >= deadline:
            break
        min_area = float('inf')
        min_tri = (0, 1, 2)
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    p1 = pts[i]
                    p2 = pts[j]
                    p3 = pts[k]
                    area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
                    if area < min_area:
                        min_area = area
                        min_tri = (i, j, k)
        if min_area < 1e-15:
            break
        i, j, k = min_tri
        p1, p2, p3 = pts[i].copy(), pts[j].copy(), pts[k].copy()
        centroid = (p1 + p2 + p3) / 3.0
        for idx, p in [(i, p1), (j, p2), (k, p3)]:
            direction = p - centroid
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction = direction / norm
                pts[idx] = p + step * direction
        new_score = get_score(pts)
        if new_score > best_score:
            best_score = new_score
        else:
            pts[i] = p1
            pts[j] = p2
            pts[k] = p3
            step *= 0.5
            if step < 1e-6:
                break
    return pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    starts = []
    starts.append(_regular_polygon(13, radius=1.0))
    starts.append(_make_12gon_center())
    
    rng = np.random.default_rng(12345)
    for noise_scale in [0.05, 0.1, 0.2]:
        base = _regular_polygon(13, radius=1.0)
        noise = rng.normal(0, noise_scale, (N, 2))
        starts.append(base + noise)
    
    base_12c = _make_12gon_center()
    noise = rng.normal(0, 0.05, (N, 2))
    starts.append(base_12c + noise)
    
    best_points = None
    best_score = 0.0
    
    for s in starts:
        s_score = get_score(s)
        if s_score > best_score:
            best_score = s_score
            best_points = s.copy()
    
    if time.time() < deadline:
        opt_pts, opt_score = _optimize_from_start(best_points, deadline, maxiter=5000)
        if opt_score > best_score:
            best_points = opt_pts
            best_score = opt_score
    
    for s in starts:
        if time.time() >= deadline - 20.0:
            break
        if np.allclose(s, best_points, atol=1e-6):
            continue
        opt_pts, opt_score = _optimize_from_start(s, deadline, maxiter=3000)
        if opt_score > best_score:
            best_points = opt_pts
            best_score = opt_score
    
    if time.time() < deadline - 5.0:
        refined_pts, refined_score = _push_apart_refinement(best_points, deadline, iterations=100)
        if refined_score > best_score:
            best_points = refined_pts
            best_score = refined_score
    
    return best_points

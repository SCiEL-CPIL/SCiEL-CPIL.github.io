import numpy as np
from itertools import combinations
from scipy.optimize import minimize
import time

N = 11
TRI_AREA = np.sqrt(3) / 4.0
H = np.sqrt(3) / 2.0

# Pre-compute all triangle index combinations
tri_indices = np.array(list(combinations(range(N), 3)))

def min_triangle_area(pts):
    """Compute minimum triangle area for N points."""
    p1 = pts[tri_indices[:, 0]]
    p2 = pts[tri_indices[:, 1]]
    p3 = pts[tri_indices[:, 2]]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def project_to_triangle(pt):
    """Project point to nearest point inside the triangle."""
    x, y = pt
    y = np.clip(y, 0, H)
    x_min = y / (2 * H)
    x_max = 1.0 - y / (2 * H)
    x = np.clip(x, x_min, x_max)
    return np.array([x, y])

def project_all(pts):
    return np.array([project_to_triangle(p) for p in pts])

def objective_neg_min_area(flat_pts):
    """Objective: negative of min triangle area (to minimize)."""
    pts = flat_pts.reshape(N, 2)
    pts_proj = project_all(pts)
    return -min_triangle_area(pts_proj)

def initial_grid_config():
    """Create a structured initial configuration with triangular lattice."""
    points = []
    # 4 rows: 3, 4, 3, 1 = 11 points
    # Row 0: y=0
    points.append([0.05, 0.0])
    points.append([0.5, 0.0])
    points.append([0.95, 0.0])
    # Row 1: y=H/3
    y1 = H / 3.0
    x_min1 = y1 / (2 * H)
    x_max1 = 1.0 - y1 / (2 * H)
    for i in range(4):
        x = x_min1 + (x_max1 - x_min1) * i / 3.0
        points.append([x, y1])
    # Row 2: y=2H/3
    y2 = 2 * H / 3.0
    x_min2 = y2 / (2 * H)
    x_max2 = 1.0 - y2 / (2 * H)
    for i in range(3):
        x = x_min2 + (x_max2 - x_min2) * i / 2.0
        points.append([x, y2])
    # Row 3: y=H
    points.append([0.5, H])
    return np.array(points)

def initial_random_config(rng):
    """Generate random points inside the triangle."""
    points = []
    while len(points) < N:
        u, v = rng.random(), rng.random()
        if u + v < 1.0:
            x = u + 0.5 * v
            y = H * v
            points.append([x, y])
    return np.array(points)

def local_optimize(pts0, max_iter=1000, timeout=8):
    """Local optimization using Nelder-Mead."""
    t0 = time.time()
    flat0 = pts0.flatten()
    
    try:
        result = minimize(
            objective_neg_min_area,
            flat0,
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12}
        )
    except Exception:
        return pts0
    
    if time.time() - t0 > timeout:
        return pts0
    
    best_pts = result.x.reshape(N, 2)
    best_pts = project_all(best_pts)
    return best_pts

def construct():
    t_start = time.time()
    deadline = 140.0
    
    rng = np.random.RandomState(42)
    
    best_score = -1.0
    best_pts = None
    
    # Initial configurations
    configs = [initial_grid_config()]
    for i in range(30):
        configs.append(initial_random_config(rng))
    
    for pts0 in configs:
        if time.time() > deadline:
            break
        
        pts_opt = local_optimize(pts0, max_iter=500, timeout=4)
        score = min_triangle_area(pts_opt) / TRI_AREA
        
        if score > best_score:
            best_score = score
            best_pts = pts_opt.copy()
    
    # Refine the best solution
    if best_pts is not None and time.time() < deadline - 10:
        for _ in range(3):
            if time.time() > deadline:
                break
            pts_refined = local_optimize(best_pts, max_iter=2000, timeout=20)
            score = min_triangle_area(pts_refined) / TRI_AREA
            if score > best_score:
                best_score = score
                best_pts = pts_refined.copy()
    
    if best_pts is None:
        best_pts = initial_grid_config()
    
    return best_pts

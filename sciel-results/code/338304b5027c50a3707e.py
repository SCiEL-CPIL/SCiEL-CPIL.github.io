"""Heilbronn convex regions n=14: maximize min triangle area / convex hull area."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

_START_TIME = time.time()
TIME_LIMIT = 140.0  # safety margin before 150s kill

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _triangle_areas(points):
    """Compute all triangle areas for n points. Returns array of shape (C(n,3),)."""
    n = len(points)
    areas = []
    for i, j, k in combinations(range(n), 3):
        xi, yi = points[i]
        xj, yj = points[j]
        xk, yk = points[k]
        area = 0.5 * abs((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))
        areas.append(area)
    return np.array(areas)

def _convex_hull_area(points):
    """Compute convex hull area using Andrew's monotone chain."""
    n = len(points)
    if n < 3:
        return 0.0
    pts = sorted(points, key=lambda p: (p[0], p[1]))
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0
    
    area = 0.0
    for i in range(len(hull)):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % len(hull)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0

def _score(points):
    """Score = min triangle area / convex hull area."""
    areas = _triangle_areas(points)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    return float(np.min(areas) / hull_area)

def _soft_min(areas, beta):
    """Smooth approximation of min using log-sum-exp (negative)."""
    # min(areas) ≈ -(1/beta) * log(sum(exp(-beta * areas)))
    # For numerical stability:
    min_val = np.min(areas)
    shifted = -beta * (areas - min_val)
    # shifted <= 0, so exp(shifted) <= 1
    log_sum = np.log(np.sum(np.exp(shifted)))
    return min_val - (1.0 / beta) * log_sum

def _negative_soft_score(points, beta):
    """Negative of soft-min triangle area / hull area for minimization."""
    areas = _triangle_areas(points)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 1e10
    soft_min_area = _soft_min(areas, beta)
    return -soft_min_area / hull_area

def _regular_polygon(n, radius=1.0, center=(0.5, 0.5), rotation=0.0):
    """Generate regular n-gon."""
    angles = 2 * np.pi * np.arange(n) / n + rotation
    cx, cy = center
    return np.array([[cx + radius * np.cos(a), cy + radius * np.sin(a)] for a in angles])

def _grid_initial(n):
    """Generate a grid-like initial configuration."""
    # 14 points: try 4x4 grid minus 2, or other patterns
    # Use a 4x4 grid, remove 2 points for 14
    xs = np.linspace(0.1, 0.9, 4)
    ys = np.linspace(0.1, 0.9, 4)
    pts = []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            pts.append([x, y])
    # Remove 2 corner points to get 14
    pts = pts[:14]
    return np.array(pts)

def _circle_with_perturbation(n, radius=1.0, center=(0.5, 0.5), rng=None):
    """Points on circle with slight radial perturbation."""
    if rng is None:
        rng = np.random.RandomState(42)
    angles = 2 * np.pi * np.arange(n) / n + rng.uniform(0, 2*np.pi)
    radii = radius * (1 + 0.05 * rng.randn(n))
    cx, cy = center
    return np.array([[cx + r * np.cos(a), cy + r * np.sin(a)] for r, a in zip(radii, angles)])

def _optimize_from_start(points0, max_iter=200, time_budget=None):
    """Optimize from a starting configuration using soft-min surrogate."""
    if time_budget is None:
        time_budget = _time_remaining()
    
    best_points = points0.copy()
    best_score = _score(points0)
    
    # Progressive beta schedule
    betas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    
    for beta in betas:
        if _time_remaining() < 5.0:
            break
        
        x0 = best_points.flatten()
        
        def obj(x):
            pts = x.reshape(-1, 2)
            return _negative_soft_score(pts, beta)
        
        try:
            result = minimize(
                obj, x0, method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.success or result.fun < obj(x0):
                new_points = result.x.reshape(-1, 2)
                new_score = _score(new_points)
                if new_score > best_score:
                    best_points = new_points
                    best_score = new_score
        except Exception:
            pass
    
    # Final exact polish with high beta
    if _time_remaining() > 10.0:
        beta_final = 500.0
        x0 = best_points.flatten()
        
        def obj_final(x):
            pts = x.reshape(-1, 2)
            return _negative_soft_score(pts, beta_final)
        
        try:
            result = minimize(
                obj_final, x0, method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result.fun < obj_final(x0):
                new_points = result.x.reshape(-1, 2)
                new_score = _score(new_points)
                if new_score > best_score:
                    best_points = new_points
                    best_score = new_score
        except Exception:
            pass
    
    return best_points, best_score

def construct():
    """Construct 14 points maximizing min triangle area / convex hull area."""
    rng = np.random.RandomState(14)
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # Regular polygon
    initial_configs.append(_regular_polygon(N, radius=1.0))
    
    # Grid
    initial_configs.append(_grid_initial(N))
    
    # Circle with perturbations (multiple seeds)
    for seed in range(5):
        r = np.random.RandomState(seed * 100 + 42)
        initial_configs.append(_circle_with_perturbation(N, radius=1.0, rng=r))
    
    # Random points in unit square
    for seed in range(3):
        r = np.random.RandomState(seed * 200 + 7)
        initial_configs.append(r.uniform(0.1, 0.9, size=(N, 2)))
    
    # Hexagonal-like pattern
    # Place points in rows
    hex_pts = []
    for row in range(4):
        n_in_row = 4 if row % 2 == 0 else 3
        x_start = 0.15 if row % 2 == 0 else 0.25
        for col in range(n_in_row):
            hex_pts.append([x_start + col * 0.2, 0.15 + row * 0.22])
    if len(hex_pts) >= N:
        initial_configs.append(np.array(hex_pts[:N]))
    
    best_points = None
    best_score = -np.inf
    
    for config in initial_configs:
        if _time_remaining() < 15.0:
            break
        
        # Normalize: center at origin, scale so convex hull area is reasonable
        cx, cy = config.mean(axis=0)
        config_centered = config - np.array([cx, cy])
        
        points, score = _optimize_from_start(config_centered)
        
        if score > best_score:
            best_score = score
            best_points = points
    
    # If we still have time, try more random restarts
    if best_points is None or _time_remaining() > 20.0:
        for attempt in range(10):
            if _time_remaining() < 10.0:
                break
            r = np.random.RandomState(1000 + attempt * 77)
            config = r.uniform(0.0, 1.0, size=(N, 2))
            cx, cy = config.mean(axis=0)
            config_centered = config - np.array([cx, cy])
            
            points, score = _optimize_from_start(config_centered, max_iter=100)
            if score > best_score:
                best_score = score
                best_points = points
    
    if best_points is None:
        # Fallback: regular polygon
        best_points = _regular_polygon(N, radius=1.0)
    
    return best_points
# EVOLVE-BLOCK-END

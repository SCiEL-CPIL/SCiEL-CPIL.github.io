"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
import random
from scipy.optimize import minimize
import numpy as np


def _triangle_areas(pts):
    """Compute all triangle areas for N points. Returns array of shape (N, N, N) or flat."""
    n = len(pts)
    # Use vectorized computation
    # Area of triangle (i, j, k) = 0.5 * |cross(pts[j]-pts[i], pts[k]-pts[i])|
    # For efficiency, compute pairwise differences
    # p1 = pts[:, None, :] (n,1,2)
    # p2 = pts[None, :, :] (1,n,2)
    # p3 = pts[None, None, :] (1,1,n)
    # diff1 = p2 - p1
    # diff2 = p3 - p1
    # cross = diff1[:,:,0]*diff2[:,:,1] - diff1[:,:,1]*diff2[:,:,0]
    # area = 0.5 * |cross|
    p1 = pts[:, np.newaxis, :]
    p2 = pts[np.newaxis, :, :]
    p3 = pts[np.newaxis, np.newaxis, :]
    d1 = p2 - p1
    d2 = p3 - p1
    cross = d1[:, :, 0] * d2[:, :, 1] - d1[:, :, 1] * d2[:, :, 0]
    areas = 0.5 * np.abs(cross)
    return areas


def _convex_hull_area(pts):
    """Compute convex hull area using Andrew's monotone chain."""
    n = len(pts)
    if n < 3:
        return 0.0
    pts_sorted = sorted(pts, key=lambda p: (p[0], p[1]))
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    lower = []
    for p in pts_sorted:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    upper = []
    for p in reversed(pts_sorted):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0
    
    # Shoelace formula
    area = 0.0
    m = len(hull)
    for i in range(m):
        j = (i + 1) % m
        area += hull[i][0] * hull[j][1]
        area -= hull[j][0] * hull[i][1]
    return abs(area) / 2.0


def _score(pts):
    """Compute min triangle area / convex hull area."""
    areas = _triangle_areas(pts)
    # Exclude degenerate triangles (i==j or j==k or i==k)
    # Set diagonal to inf
    n = len(pts)
    mask = np.ones((n, n, n), dtype=bool)
    for i in range(n):
        mask[i, i, :] = False
        mask[i, :, i] = False
        mask[:, i, i] = False
    valid_areas = areas[mask]
    if len(valid_areas) == 0:
        return 0.0
    min_area = np.min(valid_areas)
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    return min_area / hull_area


def _objective(x):
    """Negative score for minimization."""
    pts = x.reshape(-1, 2)
    return -_score(pts)


def _regular_polygon(n, radius=1.0):
    """Create a regular n-gon centered at origin."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return radius * np.column_stack([np.cos(angles), np.sin(angles)])


def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    best_score = 0.0
    best_pts = _regular_polygon(N)
    best_score = _score(best_pts)
    
    # Try multiple initializations and local optimizations
    rng = np.random.RandomState(42)
    
    initializations = []
    
    # Regular polygon
    initializations.append(_regular_polygon(N))
    
    # Slightly perturbed regular polygons
    for scale in [0.8, 1.2]:
        initializations.append(_regular_polygon(N, radius=scale))
    
    # Random points in unit circle
    for _ in range(5):
        angles = rng.uniform(0, 2 * np.pi, N)
        radii = rng.uniform(0.5, 1.0, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        initializations.append(pts)
    
    # Points on circle with some jitter
    for _ in range(3):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.uniform(-0.1, 0.1, N)
        radii = rng.uniform(0.9, 1.1, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        initializations.append(pts)
    
    # Two concentric rings
    for _ in range(3):
        n1 = N // 2
        n2 = N - n1
        angles1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
        angles2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + np.pi / n2
        pts1 = np.column_stack([np.cos(angles1), np.sin(angles1)])
        pts2 = 0.5 * np.column_stack([np.cos(angles2), np.sin(angles2)])
        pts = np.vstack([pts1, pts2])
        initializations.append(pts)
    
    # Grid-like arrangements
    for _ in range(3):
        # 4x4 grid with one point removed and adjusted
        xs = np.linspace(-1, 1, 4)
        ys = np.linspace(-1, 1, 4)
        grid = np.array([[x, y] for x in xs for y in ys])
        # Remove 2 random points and add 2 random points
        indices = rng.choice(16, 14, replace=False)
        pts = grid[indices]
        initializations.append(pts)
    
    # Optimize each initialization
    for init_pts in initializations:
        if time.time() > deadline - 30:
            break
        
        x0 = init_pts.flatten()
        
        # Try multiple optimizers
        for method in ['BFGS', 'Nelder-Mead']:
            if time.time() > deadline - 15:
                break
            
            try:
                result = minimize(
                    _objective,
                    x0,
                    method=method,
                    options={'maxiter': 2000, 'disp': False}
                )
                
                if time.time() > deadline - 5:
                    break
                
                if result.success or result.fun < -best_score:
                    pts = result.x.reshape(-1, 2)
                    score = _score(pts)
                    if score > best_score:
                        best_score = score
                        best_pts = pts.copy()
                        
            except Exception:
                pass
        
        # Also try L-BFGS-B
        if time.time() > deadline - 15:
            continue
        
        try:
            result = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'disp': False}
            )
            
            if time.time() > deadline - 5:
                continue
            
            if result.success or result.fun < -best_score:
                pts = result.x.reshape(-1, 2)
                score = _score(pts)
                if score > best_score:
                    best_score = score
                    best_pts = pts.copy()
                    
        except Exception:
            pass
    
    # Final polish with more iterations if time allows
    if time.time() < deadline - 10:
        try:
            x0 = best_pts.flatten()
            result = minimize(
                _objective,
                x0,
                method='BFGS',
                options={'maxiter': 5000, 'disp': False, 'gtol': 1e-10}
            )
            pts = result.x.reshape(-1, 2)
            score = _score(pts)
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            pass
    
    return best_pts
# EVOLVE-BLOCK-END

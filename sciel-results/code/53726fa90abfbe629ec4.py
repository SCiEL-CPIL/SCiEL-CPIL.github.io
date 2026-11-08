"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

# Precompute triangle indices for vectorized area computation
_combos = list(combinations(range(N), 3))
_idx_i = np.array([c[0] for c in _combos])
_idx_j = np.array([c[1] for c in _combos])
_idx_k = np.array([c[2] for c in _combos])

H = np.sqrt(3) / 2.0


def _min_triangle_area(points):
    """Compute minimum area of all C(11,3) triangles formed by points."""
    pi = points[_idx_i]
    pj = points[_idx_j]
    pk = points[_idx_k]
    v1 = pj - pi
    v2 = pk - pi
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return np.min(areas)


def _project_to_triangle(pts):
    """Project points to stay inside the equilateral triangle (0,0),(1,0),(0.5,H)."""
    for i in range(len(pts)):
        x, y = pts[i]
        # Clip to triangle bounds
        # Left edge: from (0,0) to (0.5, H): x >= y / (2*H)
        # Right edge: from (1,0) to (0.5, H): x <= 1 - y / (2*H)
        # Bottom: y >= 0
        # Top: y <= H
        
        if y < 0:
            y = 0
        if y > H:
            y = H
        min_x = y / (2 * H)
        max_x = 1 - y / (2 * H)
        if x < min_x:
            x = min_x
        if x > max_x:
            x = max_x
        pts[i] = [x, y]
    return pts


def _objective_flat(x):
    """Objective function for optimization: negative min area (to maximize)."""
    points = x.reshape(N, 2)
    # Project to ensure validity
    points = _project_to_triangle(points)
    return -_min_triangle_area(points)


def _make_symmetric_init(rng):
    """Create a symmetric initial configuration."""
    # Place points in a roughly symmetric pattern
    # 3 corners, 3 edge midpoints, 3 inner ring, 1 center
    pts = []
    # Corners (slightly inset)
    eps = 0.05
    pts.append([eps, eps * H / 0.5 * 0.5])  # near (0,0)
    pts.append([1 - eps, eps * H / 0.5 * 0.5])  # near (1,0)
    pts.append([0.5, H - eps * H])  # near top
    
    # Edge midpoints (slightly inset)
    pts.append([0.25, H * 0.25])
    pts.append([0.75, H * 0.25])
    pts.append([0.5, H * 0.5])
    
    # Inner ring
    r = 0.2
    for angle in [0, 2*np.pi/3, 4*np.pi/3]:
        cx, cy = 0.5, H/3
        x = cx + r * np.cos(angle)
        y = cy + r * np.sin(angle) * (H / 0.5)  # scale y
        pts.append([x, y])
    
    # Center
    pts.append([0.5, H/3])
    
    # Add small random perturbation
    for i in range(len(pts)):
        pts[i][0] += rng.uniform(-0.01, 0.01)
        pts[i][1] += rng.uniform(-0.01, 0.01)
    
    pts = _project_to_triangle(np.array(pts))
    return pts


def construct():
    t_start = time.time()
    deadline = t_start + 110  # Safety margin before 120s limit
    
    rng = np.random.default_rng(42)
    
    best_score = -np.inf
    best_points = None
    
    # Multi-start optimization
    n_starts = 20
    
    for start_idx in range(n_starts):
        if time.time() > deadline - 10:
            break
        
        # Generate initial configuration
        if start_idx == 0:
            init_pts = _make_symmetric_init(np.random.default_rng(7))
        else:
            # Random perturbation of symmetric config or random placement
            if start_idx % 2 == 0:
                init_pts = _make_symmetric_init(np.random.default_rng(start_idx * 100 + 1))
            else:
                # Random points inside triangle
                init_pts = []
                for _ in range(N):
                    u, v = rng.random(), rng.random()
                    if u + v < 1:
                        x = u + 0.5 * v
                        y = H * v
                    else:
                        x = 1 - u + 0.5 * v
                        y = H * (1 - u - v)
                    init_pts.append([x, y])
                init_pts = np.array(init_pts)
        
        init_pts = _project_to_triangle(init_pts)
        x0 = init_pts.flatten()
        
        # Optimize using SLSQP
        try:
            result = minimize(
                _objective_flat,
                x0,
                method='SLSQP',
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            
            if result.success or result.fun < best_score:
                pts = result.x.reshape(N, 2)
                pts = _project_to_triangle(pts)
                score = _min_triangle_area(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts.copy()
        except Exception:
            continue
    
    # If no optimization succeeded, fall back to best random
    if best_points is None:
        pts = _make_symmetric_init(np.random.default_rng(7))
        best_points = pts
    
    return np.array(best_points)

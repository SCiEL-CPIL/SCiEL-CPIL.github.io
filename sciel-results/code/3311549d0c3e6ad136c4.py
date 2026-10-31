"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by 3 of the points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = pts[i]
                b = pts[j]
                c = pts[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _inside_triangle(pt, eps=1e-9):
    """Check if point is inside or on the boundary of the equilateral triangle."""
    x, y = pt
    # Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    # barycentric-like check
    h = np.sqrt(3) / 2
    if y < -eps or y > h + eps:
        return False
    if x < -eps or x > 1 + eps:
        return False
    # Left edge: from (0,0) to (0.5, h): line y = sqrt(3)*x => x >= y/sqrt(3)
    # Right edge: from (1,0) to (0.5, h): line y = -sqrt(3)*(x-1) => x <= 1 - y/sqrt(3)
    if x < y / np.sqrt(3) - eps:
        return False
    if x > 1 - y / np.sqrt(3) + eps:
        return False
    return True

def _project_to_triangle(pt):
    """Project a point onto the closest point inside the equilateral triangle."""
    x, y = pt
    h = np.sqrt(3) / 2
    # Clamp y
    y = max(0.0, min(h, y))
    # Clamp x based on y
    x_min = y / np.sqrt(3)
    x_max = 1 - y / np.sqrt(3)
    x = max(x_min, min(x_max, x))
    return np.array([x, y])

def _objective_flat(flat_pts):
    """Objective to minimize: negative of min triangle area."""
    pts = flat_pts.reshape(-1, 2)
    # Penalize points outside triangle heavily
    penalty = 0.0
    for p in pts:
        if not _inside_triangle(p, eps=-1e-6):
            # Compute distance to triangle boundary roughly
            proj = _project_to_triangle(p)
            penalty += 10.0 * np.linalg.norm(p - proj)
    if penalty > 0:
        return 1.0 + penalty
    area = _min_triangle_area(pts)
    return -area

def _initial_placement():
    """Generate a good initial placement of 11 points."""
    h = np.sqrt(3) / 2
    pts = []
    # Centroid
    pts.append(np.array([0.5, h / 3]))
    # Near vertices (inset slightly)
    inset = 0.15
    pts.append(np.array([inset, inset * np.sqrt(3) * 0.5]))  # near (0,0)
    pts.append(np.array([1 - inset, inset * np.sqrt(3) * 0.5]))  # near (1,0)
    pts.append(np.array([0.5, h - inset * np.sqrt(3) * 0.5]))  # near top
    
    # Hexagonal-like ring of 6 points around centroid
    center = np.array([0.5, h / 3])
    radius = 0.22
    for i in range(6):
        angle = i * np.pi / 3 + np.pi / 6
        px = center[0] + radius * np.cos(angle)
        py = center[1] + radius * np.sin(angle)
        pts.append(_project_to_triangle(np.array([px, py])))
    
    return np.array(pts)

def construct():
    deadline = time.time() + 130  # Leave 20s safety margin
    best_pts = _initial_placement()
    best_area = _min_triangle_area(best_pts)
    
    # Local optimization using Nelder-Mead on flattened coordinates
    flat_init = best_pts.flatten()
    
    # Try multiple random perturbations of the initial placement
    np.random.seed(42)
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            flat_start = flat_init.copy()
        else:
            # Add small random perturbation
            perturbation = 0.05 * np.random.randn(2 * N)
            flat_start = flat_init + perturbation
            # Project back to triangle
            pts_pert = flat_start.reshape(-1, 2)
            for i in range(N):
                pts_pert[i] = _project_to_triangle(pts_pert[i])
            flat_start = pts_pert.flatten()
        
        try:
            result = minimize(
                _objective_flat,
                flat_start,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if time.time() > deadline - 10:
                break
            
            pts_opt = result.x.reshape(-1, 2)
            # Ensure all points are inside triangle
            valid = True
            for i in range(N):
                if not _inside_triangle(pts_opt[i]):
                    pts_opt[i] = _project_to_triangle(pts_opt[i])
                    valid = True
            
            area_opt = _min_triangle_area(pts_opt)
            if area_opt > best_area:
                best_area = area_opt
                best_pts = pts_opt.copy()
        except Exception:
            pass
    
    # One more round with more iterations if time allows
    if time.time() < deadline - 30:
        try:
            result = minimize(
                _objective_flat,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-9}
            )
            pts_opt = result.x.reshape(-1, 2)
            for i in range(N):
                if not _inside_triangle(pts_opt[i]):
                    pts_opt[i] = _project_to_triangle(pts_opt[i])
            area_opt = _min_triangle_area(pts_opt)
            if area_opt > best_area:
                best_area = area_opt
                best_pts = pts_opt.copy()
        except Exception:
            pass
    
    return best_pts
# EVOLVE-BLOCK-END

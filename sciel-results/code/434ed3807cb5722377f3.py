"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import minimize

# Equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = 0.5 * np.sqrt(3) / 2  # area of the equilateral triangle with side 1


def _point_in_triangle(p, eps=1e-9):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(p, V0, V1)
    d2 = sign(p, V1, V2)
    d3 = sign(p, V2, V0)
    has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
    has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
    return not (has_neg and has_pos)


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by three points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((points[j][0] - points[i][0]) * (points[k][1] - points[i][1]) -
                                 (points[k][0] - points[i][0]) * (points[j][1] - points[i][1]))
                if area < min_area:
                    min_area = area
    return min_area


def _min_triangle_area_normalized(points):
    """Return the minimum triangle area divided by the triangle area (for reporting)."""
    return _min_triangle_area(points) / TRI_AREA


def _negative_min_area_flat(x):
    """Objective function for optimization: negative of the minimum triangle area."""
    points = x.reshape(N, 2)
    # Check if all points are inside the triangle
    for p in points:
        if not _point_in_triangle(p):
            return 1.0  # Large penalty for invalid points
    
    return -_min_triangle_area(points)


def _gradient_flat(x):
    """Compute gradient of the objective (negative min area) with respect to flat x."""
    points = x.reshape(N, 2)
    
    # Find the minimum area triangle
    n = N
    min_area = float('inf')
    min_tri = None
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((points[j][0] - points[i][0]) * (points[k][1] - points[i][1]) -
                                 (points[k][0] - points[i][0]) * (points[j][1] - points[i][1]))
                if area < min_area:
                    min_area = area
                    min_tri = (i, j, k)
    
    i, j, k = min_tri
    
    # Compute gradient of the area with respect to points
    # Area = 0.5 * |(pj - pi) x (pk - pi)|
    # Let's compute the signed area and its gradient
    pi = points[i]
    pj = points[j]
    pk = points[k]
    
    # Signed area (without the 0.5 and abs)
    cross = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pk[0] - pi[0]) * (pj[1] - pi[1])
    sign = 1.0 if cross > 0 else -1.0
    
    # Gradient of signed area with respect to each point
    # d(area)/d(pi) = -sign * (pk - pj)
    # d(area)/d(pj) = sign * (pk - pi)  
    # d(area)/d(pk) = -sign * (pj - pi)
    
    grad = np.zeros_like(x)
    grad[2*i] = -sign * (pk[0] - pj[0])
    grad[2*i+1] = -sign * (pk[1] - pj[1])
    
    grad[2*j] = sign * (pk[0] - pi[0])
    grad[2*j+1] = sign * (pk[1] - pi[1])
    
    grad[2*k] = -sign * (pj[0] - pi[0])
    grad[2*k+1] = -sign * (pj[1] - pi[1])
    
    # The objective is -min_area, and min_area = 0.5 * |cross| = 0.5 * sign * cross
    # So d(-min_area)/dx = -0.5 * sign * d(cross)/dx
    grad = -0.5 * grad
    
    return grad


def _random_initial_points():
    """Generate random initial points inside the triangle."""
    points = []
    while len(points) < N:
        u = np.random.random()
        v = np.random.random()
        if u + v < 1.0:
            p = V0 + u * (V1 - V0) + v * (V2 - V0)
            points.append(p)
    return np.array(points).flatten()


def _grid_initial_points():
    """Generate initial points on a grid-like pattern inside the triangle."""
    # Use a structured approach: place points at centroids and midpoints
    points = []
    
    # Centroid
    centroid = (V0 + V1 + V2) / 3
    points.append(centroid)
    
    # Midpoints of sides
    mid01 = (V0 + V1) / 2
    mid12 = (V1 + V2) / 2
    mid20 = (V2 + V0) / 2
    points.extend([mid01, mid12, mid20])
    
    # Add more points to reach N=11
    # Use a finer grid
    for i in range(1, 4):
        for j in range(1, 4):
            if i + j < 4 and len(points) < N:
                u = i / 4.0
                v = j / 4.0
                if u + v < 1.0:
                    p = V0 + u * (V1 - V0) + v * (V2 - V0)
                    # Avoid duplicates
                    is_dup = False
                    for q in points:
                        if np.linalg.norm(p - q) < 1e-6:
                            is_dup = True
                            break
                    if not is_dup:
                        points.append(p)
    
    # Fill remaining with random points
    while len(points) < N:
        u = np.random.random()
        v = np.random.random()
        if u + v < 1.0:
            p = V0 + u * (V1 - V0) + v * (V2 - V0)
            is_dup = False
            for q in points:
                if np.linalg.norm(p - q) < 1e-6:
                    is_dup = True
                    break
            if not is_dup:
                points.append(p)
    
    return np.array(points[:N]).flatten()


def construct():
    """Construct 11 points inside the equilateral triangle to maximize the minimum triangle area."""
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    best_x = None
    best_val = float('-inf')
    
    # Try multiple initializations
    initializations = []
    
    # Grid-based initialization
    initializations.append(_grid_initial_points())
    
    # Random initializations
    for _ in range(5):
        initializations.append(_random_initial_points())
    
    # Symmetric initialization: place points in a hexagonal-like pattern
    # This is based on known optimal configurations for small n
    sym_points = []
    # Center
    sym_points.append((V0 + V1 + V2) / 3)
    # 6 points around center at radius r
    r = 0.3
    for angle in np.linspace(0, 2*np.pi, 6, endpoint=False):
        p = (V0 + V1 + V2) / 3 + r * np.array([np.cos(angle), np.sin(angle)])
        if _point_in_triangle(p):
            sym_points.append(p)
    # Add 4 more points
    for angle in np.linspace(0, 2*np.pi, 4, endpoint=False):
        p = (V0 + V1 + V2) / 3 + 0.5 * np.array([np.cos(angle), np.sin(angle)])
        if _point_in_triangle(p) and len(sym_points) < N:
            sym_points.append(p)
    
    while len(sym_points) < N:
        u = np.random.random()
        v = np.random.random()
        if u + v < 1.0:
            p = V0 + u * (V1 - V0) + v * (V2 - V0)
            sym_points.append(p)
    
    initializations.append(np.array(sym_points[:N]).flatten())
    
    # Optimize from each initialization
    for x0 in initializations:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                _negative_min_area_flat,
                x0,
                method='L-BFGS-B',
                jac=_gradient_flat,
                options={'maxiter': 1000, 'maxls': 50, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < best_val:
                x_opt = result.x
                points_opt = x_opt.reshape(N, 2)
                
                # Verify all points are inside the triangle
                valid = True
                for p in points_opt:
                    if not _point_in_triangle(p, eps=1e-8):
                        valid = False
                        break
                
                if valid:
                    val = -result.fun
                    if val > best_val:
                        best_val = val
                        best_x = x_opt.copy()
                        
        except Exception:
            continue
    
    # If no valid result from optimization, fall back to a known good configuration
    if best_x is None:
        # Use the grid initialization as fallback
        best_x = _grid_initial_points()
    
    points = best_x.reshape(N, 2)
    
    # Final validation
    for p in points:
        if not _point_in_triangle(p, eps=1e-6):
            # If any point is invalid, use a simple valid configuration
            points = _grid_initial_points().reshape(N, 2)
            break
    
    return points
# EVOLVE-BLOCK-END

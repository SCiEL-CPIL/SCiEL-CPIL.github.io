"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
from scipy.spatial import ConvexHull

def _convex_hull_area(points):
    """Compute the area of the convex hull of a set of 2D points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _all_triangle_areas(points):
    """Compute areas of all triples of points."""
    n = len(points)
    areas = []
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                areas.append(area)
                indices.append((i, j, k))
    return np.array(areas), indices

def _jacobian_for_triangle(points, i, j, k):
    """
    Compute the gradient of the area of triangle (i, j, k) with respect to all 2N coordinates.
    Area = 0.5 * | (p_j - p_i) x (p_k - p_i) |
    Let A = 0.5 * ( (x_j - x_i)(y_k - y_i) - (x_k - x_i)(y_j - y_i) )
    We assume the sign is constant locally.
    dA/dx_i = 0.5 * ( -(y_k - y_i) + (x_k - x_i) * 0 ) ... wait.
    A = 0.5 * ( x_j y_k - x_j y_i - x_i y_k + x_i y_i + x_k y_i - x_k y_j - x_i y_k + x_i y_j ) -- No, simpler:
    A = 0.5 * ( (x_j - x_i)(y_k - y_i) - (x_k - x_i)(y_j - y_i) )
    A = 0.5 * ( x_j y_k - x_j y_i - x_i y_k + x_i y_i - x_k y_j + x_k y_i + x_i y_j - x_i y_i )
    A = 0.5 * ( x_j y_k - x_j y_i - x_i y_k - x_k y_j + x_k y_i + x_i y_j )
    
    dA/dx_i = 0.5 * ( -y_k + y_i + y_j )
    dA/dy_i = 0.5 * ( x_k - x_j )
    
    dA/dx_j = 0.5 * ( y_i - y_k )
    dA/dy_j = 0.5 * ( -x_i + x_k )
    
    dA/dx_k = 0.5 * ( y_j - y_i )
    dA/dy_k = 0.5 * ( x_i - x_j )
    """
    grad = np.zeros(2 * len(points))
    xi, yi = points[i]
    xj, yj = points[j]
    xk, yk = points[k]
    
    # dA/dx_i
    grad[2*i] = 0.5 * (-yk + yi + yj)
    # dA/dy_i
    grad[2*i + 1] = 0.5 * (xk - xj)
    
    # dA/dx_j
    grad[2*j] = 0.5 * (yi - yk)
    # dA/dy_j
    grad[2*j + 1] = 0.5 * (-xi + xk)
    
    # dA/dx_k
    grad[2*k] = 0.5 * (yj - yi)
    # dA/dy_k
    grad[2*k + 1] = 0.5 * (xi - xj)
    
    return grad

def construct():
    """
    Place 13 points in a planar convex region to maximize the minimum triangle area
    divided by the convex hull area.
    
    Approach: Damped Newton-Raphson to equalize the areas of the smallest triangles.
    """
    best_score = 0.0
    best_points = None
    
    # Initial configuration: Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    init_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Normalize to unit hull area
    hull_area = _convex_hull_area(init_points)
    if hull_area > 1e-10:
        init_points = init_points / np.sqrt(hull_area)
    
    current_points = init_points.copy()
    current_score = 0.0
    
    # Damped Newton loop
    max_iter = 100
    damping = 0.1
    min_area_tol = 1e-5
    
    for it in range(max_iter):
        areas, indices = _all_triangle_areas(current_points)
        min_area = np.min(areas)
        hull_area = _convex_hull_area(current_points)
        
        if hull_area < 1e-10:
            break
            
        current_score = min_area / hull_area
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_points.copy()
            
        # Identify active triangles: those close to the minimum
        # We want to push these up.
        threshold = min_area + 0.01 * min_area + 1e-6
        active_mask = areas < threshold
        active_indices = [indices[i] for i in range(len(indices)) if active_mask[i]]
        
        if len(active_indices) == 0:
            break
            
        # Limit the number of active triangles to avoid overdetermined systems
        # If too many, pick the smallest ones
        if len(active_indices) > 26: # 2N variables
            # Sort by area and take the smallest 26
            sorted_active = sorted(active_indices, key=lambda idx: areas[indices.index(idx)])
            active_indices = sorted_active[:26]
            
        # Build Jacobian matrix J (m x 2N)
        m = len(active_indices)
        J = np.zeros((m, 2*N))
        b = np.zeros(m)
        
        for r, (i, j, k) in enumerate(active_indices):
            J[r] = _jacobian_for_triangle(current_points, i, j, k)
            # We want to increase the area.
            # Target area: slightly above current min
            # Let's target the average of the active areas + a small delta
            target_area = min_area + 0.05 * min_area 
            b[r] = target_area - areas[indices.index((i,j,k))]
            
        # Solve J dx = b using least squares
        try:
            dx, residuals, rank, s = np.linalg.lstsq(J, b, rcond=None)
        except Exception:
            break
            
        # Apply damping
        new_points = current_points + damping * dx.reshape(N, 2)
        
        # Check if the new points are valid (non-degenerate)
        new_hull_area = _convex_hull_area(new_points)
        if new_hull_area < 1e-10:
            damping *= 0.5
            continue
            
        new_areas, _ = _all_triangle_areas(new_points)
        new_min_area = np.min(new_areas)
        new_score = new_min_area / new_hull_area
        
        if new_score > current_score:
            current_points = new_points
            # Increase damping slightly if progress is good
            damping = min(1.0, damping * 1.1)
        else:
            # Decrease damping
            damping *= 0.5
            if damping < 1e-4:
                break
                
    # Final normalization
    if best_points is not None:
        hull_area = _convex_hull_area(best_points)
        if hull_area > 1e-10:
            best_points = best_points / np.sqrt(hull_area)
    else:
        best_points = init_points
        
    return best_points
# EVOLVE-BLOCK-END

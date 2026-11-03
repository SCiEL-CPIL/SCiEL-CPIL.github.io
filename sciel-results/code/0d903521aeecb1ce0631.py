"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _min_triangle_area_fast(points):
    """Compute min area of any triangle formed by the n points.
    points: (n, 2) array. Returns scalar min area."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            d = points[j] - points[i]  # (2,)
            diff_k = points - points[i]  # (n, 2)
            cross = d[0] * diff_k[:, 1] - d[1] * diff_k[:, 0]  # (n,)
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            mask[j] = False
            areas = 0.5 * np.abs(cross[mask])
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area


def _bary_to_cart(bc):
    """Convert barycentric (b,c) coordinates (n,2) to Cartesian (n,2).
    a = 1 - b - c
    x = b + 0.5*c
    y = c*sqrt(3)/2
    """
    b = bc[:, 0]
    c = bc[:, 1]
    x = b + 0.5 * c
    y = c * (np.sqrt(3) / 2)
    return np.column_stack([x, y])


def _clip_to_triangle(points):
    """Project points into the equilateral triangle."""
    x = points[:, 0].copy()
    y = points[:, 1].copy()
    
    # y bounds
    y_max = np.sqrt(3) / 2
    y = np.clip(y, 0, y_max)
    
    # x bounds based on y
    # Left edge: x >= y / sqrt(3)
    # Right edge: x <= 1 - y / sqrt(3)
    x_min = y / np.sqrt(3)
    x_max = 1 - y / np.sqrt(3)
    
    # If x is out of [x_min, x_max], project to the nearest edge
    # This is a simple clamp, which works for convex sets like a triangle
    x = np.clip(x, x_min, x_max)
    
    return np.column_stack([x, y])


def _objective_bary(bc_flat):
    """
    Objective function for SLSQP.
    Minimizes the negative of the minimum triangle area.
    Input: flat array of length 2*N, representing (b_i, c_i) for each point.
    """
    bc = bc_flat.reshape(N, 2)
    points = _bary_to_cart(bc)
    min_area = _min_triangle_area_fast(points)
    return -min_area


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin (limit is 150s)
    
    rng = np.random.default_rng(42)
    best_points = None
    best_score = -float('inf')
    
    # Constraints for SLSQP in barycentric coordinates (b, c)
    # a = 1 - b - c >= 0  =>  b + c <= 1
    # b >= 0
    # c >= 0
    
    # We define constraints for each point i
    # Inequality constraints: func(x) >= 0
    # b_i >= 0
    # c_i >= 0
    # 1 - b_i - c_i >= 0
    
    constraints = []
    for i in range(N):
        # b_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i]})
        # c_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1]})
        # 1 - b_i - c_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*i+1]})
    
    # Bounds for b and c: [0, 1]
    bounds = [(0.0, 1.0)] * (2 * N)
    
    # Multi-start SLSQP
    n_starts = 20
    
    for i in range(n_starts):
        if time.time() > deadline - 5:
            break
            
        # Generate random initial points in the triangle
        # Use rejection sampling or transformation
        # Transformation: u, v in [0,1], if u+v>1 reflect
        bc_init = np.zeros((N, 2))
        for k in range(N):
            u = rng.uniform(0, 1)
            v = rng.uniform(0, 1)
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            bc_init[k, 0] = u
            bc_init[k, 1] = v
            
        x0 = bc_init.flatten()
        
        try:
            result = minimize(
                _objective_bary,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if result.success:
                bc_opt = result.x.reshape(N, 2)
                points_opt = _bary_to_cart(bc_opt)
                
                # Safety clip just in case SLSQP had numerical issues
                points_opt = _clip_to_triangle(points_opt)
                
                score = _min_triangle_area_fast(points_opt)
                
                if score > best_score:
                    best_score = score
                    best_points = points_opt.copy()
                    
        except Exception:
            continue
    
    if best_points is None:
        # Fallback: Centered grid
        h = np.sqrt(3) / 2
        best_points = np.array([
            [0.5, 0.0], [0.0, 0.0], [1.0, 0.0],
            [0.25, h/2], [0.75, h/2], [0.5, h],
            [0.5, h/4], [0.25, h/4], [0.75, h/4],
            [0.5, 3*h/4], [0.5, 2*h/4]
        ])
        
    # Final safety check: ensure returned points are inside
    best_points = _clip_to_triangle(best_points)
    
    return best_points
# EVOLVE-BLOCK-END

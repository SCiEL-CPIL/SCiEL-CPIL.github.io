import numpy as np
from scipy.optimize import differential_evolution
import time

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 600  # 600s limit, keep margin
    
    n = 12
    sqrt3 = np.sqrt(3)
    
    # Objective function for Differential Evolution
    def objective(params):
        if time.time() > deadline - 10:
            return 1000.0 # Return high value to stop
            
        centers = params.reshape(n, 2)
        
        # 1. Check separation (non-overlap)
        # Min distance between centers of aligned unit hexagons is sqrt(3)
        min_dist = sqrt3
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < min_dist:
                    # Quadratic penalty for overlap
                    penalty += (min_dist - d) ** 2 * 1000.0
        
        # 2. Calculate required outer hex side length s
        # For a point (x,y) to be inside a hexagon of side s centered at origin (aligned):
        # |x*cos(theta) + y*sin(theta)| <= (sqrt(3)/2) * s  for theta in {0, 60, 120}
        # s >= (2/sqrt(3)) * max(|proj|)
        
        max_s = 0.0
        # Pre-compute angles and cos/sin for the 3 axes
        angles = np.deg2rad([0, 60, 120])
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)
        
        # Vertices of a unit hexagon (aligned)
        # Angles: 0, 60, 120, 180, 240, 300
        v_angles = np.deg2rad(np.arange(0, 360, 60))
        v_cos = np.cos(v_angles)
        v_sin = np.sin(v_angles)
        
        for c in centers:
            cx, cy = c
            # For each vertex of the inner hexagon
            for k in range(6):
                vx = cx + v_cos[k]
                vy = cy + v_sin[k]
                
                # Projections onto the 3 normal axes of the outer hex
                # proj_0 = vx
                # proj_60 = vx*cos(60) + vy*sin(60)
                # proj_120 = vx*cos(120) + vy*sin(120)
                proj0 = abs(vx)
                proj60 = abs(vx * 0.5 + vy * (sqrt3 / 2))
                proj120 = abs(vx * (-0.5) + vy * (sqrt3 / 2))
                
                s_req = max(proj0, proj60, proj120) * (2.0 / sqrt3)
                if s_req > max_s:
                    max_s = s_req
                    
        return max_s + penalty

    # Define bounds for the 24 variables (12 centers)
    # The centers should be within a reasonable range, e.g., [-5, 5]
    bounds = [(-5.0, 5.0)] * (n * 2)
    
    # Run Differential Evolution
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=200, 
            popsize=20, 
            tol=0.01, 
            seed=42,
            polish=True # Use Nelder-Mead to polish the result
        )
        
        best_params = result.x
        best_centers = best_params.reshape(n, 2)
        
        # Calculate the actual required s for the best centers
        # Re-use logic from objective but without penalty
        max_s_actual = 0.0
        for c in best_centers:
            cx, cy = c
            for k in range(6):
                vx = cx + v_cos[k]
                vy = cy + v_sin[k]
                proj0 = abs(vx)
                proj60 = abs(vx * 0.5 + vy * (sqrt3 / 2))
                proj120 = abs(vx * (-0.5) + vy * (sqrt3 / 2))
                s_req = max(proj0, proj60, proj120) * (2.0 / sqrt3)
                if s_req > max_s_actual:
                    max_s_actual = s_req
        
        # Add a tiny epsilon to ensure strict validity due to floating point
        s_final = max_s_actual + 1e-5
        
        # Verify validity
        # 1. Separation
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(best_centers[i] - best_centers[j])
                if d < sqrt3 - 1e-8:
                    valid = False
                    break
            if not valid:
                break
        
        # 2. Containment
        if valid:
            for c in best_centers:
                cx, cy = c
                for k in range(6):
                    vx = cx + v_cos[k]
                    vy = cy + v_sin[k]
                    # Check if vertex is inside outer hex of side s_final
                    # |proj| <= (sqrt(3)/2) * s
                    limit = (sqrt3 / 2) * s_final
                    if abs(vx) > limit + 1e-8:
                        valid = False
                        break
                    if abs(vx * 0.5 + vy * (sqrt3 / 2)) > limit + 1e-8:
                        valid = False
                        break
                    if abs(vx * (-0.5) + vy * (sqrt3 / 2)) > limit + 1e-8:
                        valid = False
                        break
                if not valid:
                    break

        if valid:
            # Construct return arrays
            inner_hex_data = np.zeros((n, 3))
            for i in range(n):
                inner_hex_data[i, 0] = best_centers[i, 0]
                inner_hex_data[i, 1] = best_centers[i, 1]
                inner_hex_data[i, 2] = 0.0 # Angle 0
            
            outer_hex_data = np.array([0.0, 0.0, 0.0])
            return inner_hex_data, outer_hex_data, s_final
            
    except Exception as e:
        # If optimization fails, fall back to seed solution
        pass

    # Fallback: Seed solution
    inner_hex_data = np.array(
        [
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
            [0, -4, 0],
        ]
    )
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = 8
    return inner_hex_data, outer_hex_data, outer_hex_side_length

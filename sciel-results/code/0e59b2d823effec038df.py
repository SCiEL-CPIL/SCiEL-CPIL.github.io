import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0  # 5s safety margin
    
    # Define 11 hexagonal lattice positions (axial to Cartesian)
    axial_coords = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (0, 2), (-2, 2), (1, -2)
    ]
    
    def axial_to_cartesian(q, r):
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return np.array([x, y])
    
    base_positions = np.array([axial_to_cartesian(q, r) for q, r in axial_coords])
    base_center = base_positions.mean(axis=0)
    base_positions = base_positions - base_center
    
    def compute_container_side(transformed_points, container_center, container_angle, inner_angle=0.0):
        """Compute minimum container side length using conservative support bound."""
        normals = []
        for k in range(6):
            angle = container_angle + k * np.pi / 3
            normals.append(np.array([np.cos(angle), np.sin(angle)]))
        
        max_proj = 0.0
        for n in normals:
            for p in transformed_points:
                proj = (p - container_center) @ n
                # Add support of inner hexagon in direction n
                # Conservative: use 1.0 (max support of unit hexagon)
                max_proj = max(max_proj, proj + 1.0)
        
        return 2.0 * max_proj / np.sqrt(3)
    
    def check_non_overlap(transformed_points):
        min_dist = np.sqrt(3) - 1e-9
        for i in range(len(transformed_points)):
            for j in range(i + 1, len(transformed_points)):
                dist = np.linalg.norm(transformed_points[i] - transformed_points[j])
                if dist < min_dist:
                    return False
        return True
    
    def objective(params):
        if time.time() > deadline:
            return 1e6
        
        m00, m01, m10, m11 = params[0:4]
        cx, cy = params[4:6]
        angle = params[6]
        
        M = np.array([[m00, m01], [m10, m11]])
        transformed = base_positions @ M.T
        
        # Check non-overlap with penalty
        min_dist = np.sqrt(3) - 1e-9
        penalty = 0.0
        for i in range(len(transformed)):
            for j in range(i + 1, len(transformed)):
                dist = np.linalg.norm(transformed[i] - transformed[j])
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
        
        side = compute_container_side(transformed, np.array([cx, cy]), angle)
        return side + penalty * 1000.0
    
    # Initial guess
    x0_list = [
        np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    ]
    
    # Add random starts
    np.random.seed(42)
    for _ in range(15):
        scale_x = np.random.uniform(0.85, 1.15)
        scale_y = np.random.uniform(0.85, 1.15)
        shear = np.random.uniform(-0.2, 0.2)
        rot = np.random.uniform(-0.3, 0.3)
        c, s = np.cos(rot), np.sin(rot)
        M_init = np.array([[c * scale_x, -s * scale_y + shear * scale_x],
                           [s * scale_x, c * scale_y]])
        x0 = np.array([M_init[0,0], M_init[0,1], M_init[1,0], M_init[1,1],
                      np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
                      np.random.uniform(-np.pi/6, np.pi/6)])
        x0_list.append(x0)
    
    best_result = None
    best_side = float('inf')
    
    for x0 in x0_list:
        if time.time() > deadline - 3.0:
            break
        
        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-7})
        
        if result.fun < best_side and result.fun < 1e4:
            best_side = result.fun
            best_result = result.x
    
    if best_result is None:
        transformed = base_positions
        best_angle = 0
        best_side = float('inf')
        for angle in np.linspace(0, np.pi/3, 7):
            side = compute_container_side(transformed, np.array([0,0]), angle)
            if side < best_side:
                best_side = side
                best_angle = angle
        inner = [[p[0], p[1], 0.0] for p in transformed]
        return inner, (0.0, 0.0), best_side, best_angle
    
    m00, m01, m10, m11 = best_result[0:4]
    cx, cy = best_result[4:6]
    angle = best_result[6]
    
    M = np.array([[m00, m01], [m10, m11]])
    transformed = base_positions @ M.T
    
    inner = [[p[0], p[1], 0.0] for p in transformed]
    side = compute_container_side(transformed, np.array([cx, cy]), angle)
    
    return inner, (cx, cy), side, angle

# EVOLVE-BLOCK-END

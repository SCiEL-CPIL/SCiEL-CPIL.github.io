import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns
        inner_hex_data: np.ndarray of shape (12,3)
        outer_hex_data: np.ndarray of shape (3,)
        outer_hex_side_length: float
    """
    
    def get_vertices(cx, cy, side, angle_deg):
        """
        Returns vertices of a hexagon with center (cx, cy), side length 'side', 
        and rotation 'angle_deg'.
        Vertices are at angles angle + 60*i.
        """
        angles = np.deg2rad(angle_deg + 60 * np.arange(6))
        xs = cx + side * np.cos(angles)
        ys = cy + side * np.sin(angles)
        return np.column_stack((xs, ys))

    def is_valid_packing(positions, side=1.0):
        """
        Checks if hexagons at 'positions' (12, 2) with side 'side' 
        do not overlap. 
        For aligned hexagons, distance >= sqrt(3) is sufficient.
        """
        n = len(positions)
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < np.sqrt(3) - 1e-9:
                    return False
        return True

    def compute_required_side(positions, side=1.0):
        """
        Computes the minimum side length 's' of the outer hexagon 
        (centered at origin, angle 0) that contains all inner hexagons.
        """
        max_s = 0.0
        for cx, cy in positions:
            verts = get_vertices(cx, cy, side, 0)
            for vx, vy in verts:
                # Calculate required s for this vertex
                # Inequalities for unit hexagon:
                # 1. sqrt(3)*x + y <= sqrt(3)
                # 2. -sqrt(3)*x + y <= sqrt(3)
                # 3. -y <= 1
                # 4. -sqrt(3)*x - y <= sqrt(3)
                # 5. sqrt(3)*x - y <= sqrt(3)
                # 6. y <= 1
                
                s1 = (np.sqrt(3) * vx + vy) / np.sqrt(3)
                s2 = (-np.sqrt(3) * vx + vy) / np.sqrt(3)
                s3 = -vy
                s4 = (-np.sqrt(3) * vx - vy) / np.sqrt(3)
                s5 = (np.sqrt(3) * vx - vy) / np.sqrt(3)
                s6 = vy
                
                local_max = max(s1, s2, s3, s4, s5, s6)
                if local_max > max_s:
                    max_s = local_max
        return max_s

    def objective(params):
        """
        Objective function for the optimizer.
        params: 24 values (x1, y1, ..., x12, y12)
        """
        positions = params.reshape(12, 2)
        
        # Check overlap
        if not is_valid_packing(positions):
            # Return a large penalty
            return 100.0
            
        s = compute_required_side(positions)
        return s

    # Initial guess: Honeycomb lattice
    # Center at (0,0)
    # Neighbors at distance sqrt(3)
    sqrt3 = np.sqrt(3)
    init_positions = np.array([
        [0.0, 0.0],
        [sqrt3, 0.0],
        [-sqrt3, 0.0],
        [sqrt3/2, 1.5],
        [-sqrt3/2, 1.5],
        [sqrt3/2, -1.5],
        [-sqrt3/2, -1.5],
        # Add 5 more to make 12. 
        # Extend the lattice.
        [sqrt3*1.5, 1.5], # (3*sqrt3/2, 1.5)
        [-sqrt3*1.5, 1.5], # (-3*sqrt3/2, 1.5)
        [sqrt3*1.5, -1.5], # (3*sqrt3/2, -1.5)
        [-sqrt3*1.5, -1.5], # (-3*sqrt3/2, -1.5)
        [0.0, 3.0] # Top center
    ])
    
    # Verify initial validity
    if not is_valid_packing(init_positions):
        # Fallback to a simpler spread if initial is bad
        init_positions = np.random.rand(12, 2) * 10 - 5

    best_result = None
    best_s = float('inf')
    
    # Use Differential Evolution for global optimization
    # Bounds: roughly -5 to 5
    bounds = [(-5, 5)] * 24
    
    start_time = time.time()
    time_limit = 600  # seconds
    
    try:
        # DE is robust for this size
        result = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            tol=1e-8,
            polish=True,
            workers=1
        )
        
        if result.success:
            best_positions = result.x.reshape(12, 2)
            best_s = result.fun
            best_result = best_positions
            
        # Refine with Nelder-Mead
        if best_result is not None:
            res_refined = minimize(
                objective, 
                best_result.flatten(), 
                method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 1000}
            )
            if res_refined.fun < best_s:
                best_s = res_refined.fun
                best_result = res_refined.x.reshape(12, 2)
                
    except Exception as e:
        pass # Fallback to initial if optimization fails

    if best_result is None:
        best_result = init_positions
        best_s = compute_required_side(best_result)

    # Construct output
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = best_result[i, 0]
        inner_hex_data[i, 1] = best_result[i, 1]
        inner_hex_data[i, 2] = 0.0  # Angle

    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_side = best_s

    return inner_hex_data, outer_hex_data, outer_side


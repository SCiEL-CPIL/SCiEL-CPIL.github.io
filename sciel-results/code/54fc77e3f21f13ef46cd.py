import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_12():
    start_time = time.time()
    deadline = 600.0

    # Constants
    N = 12
    min_dist = np.sqrt(3)
    
    # Helper to get vertices of a unit hexagon at (cx, cy) with angle 0
    def get_unit_hex_verts(cx, cy):
        # Angle 0 means vertex at (1, 0)
        angles = np.deg2rad(np.arange(0, 360, 60))
        return np.column_stack((cx + np.cos(angles), cy + np.sin(angles)))

    # Objective function
    # vars: [x1, y1, ..., x12, y12, phi]
    def objective(vars):
        if time.time() > deadline:
            return np.inf
            
        centers = vars[:24].reshape(12, 2)
        phi = vars[24]
        
        # Get all vertices
        all_verts = []
        for i in range(12):
            verts = get_unit_hex_verts(centers[i, 0], centers[i, 1])
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        
        # Outer hexagon normals
        # Outer hexagon centered at (0,0), angle phi
        # Normals are at phi + 30 + 60k
        normal_angles = np.deg2rad(phi + 30 + 60 * np.arange(6))
        normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
        
        # Max projection
        projs = all_verts @ normals.T
        max_proj = np.max(projs)
        
        # s = (2 / sqrt(3)) * max_proj
        s = (2.0 / np.sqrt(3)) * max_proj
        return s

    # Constraints
    # Pairwise distance >= min_dist
    # Inequality: dist^2 - min_dist^2 >= 0
    def make_constraint(i, j):
        def constraint(vars):
            c1 = vars[2*i:2*i+2]
            c2 = vars[2*j:2*j+2]
            d2 = np.sum((c1 - c2)**2)
            return d2 - min_dist**2
        return constraint

    constraints = []
    for i in range(12):
        for j in range(i+1, 12):
            constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})

    # Initial guess: 4-3-4-1 cluster
    # Row 0 (y=0): 4 hexes at x = 0, sqrt(3), 2sqrt(3), 3sqrt(3)
    # Row 1 (y=1.5): 3 hexes at x = sqrt(3)/2, 3sqrt(3)/2, 5sqrt(3)/2
    # Row 2 (y=3): 4 hexes at x = 0, sqrt(3), 2sqrt(3), 3sqrt(3)
    # Row 3 (y=4.5): 1 hex at x = sqrt(3)/2
    
    sqrt3 = np.sqrt(3)
    x_coords = []
    y_coords = []
    
    # Center the cluster roughly at origin
    # x range: 0 to 3sqrt3 ~ 5.2
    # y range: 0 to 4.5
    # Center x: 2.6, Center y: 2.25
    
    row0_x = [0, sqrt3, 2*sqrt3, 3*sqrt3]
    row1_x = [sqrt3/2, 3*sqrt3/2, 5*sqrt3/2]
    row2_x = [0, sqrt3, 2*sqrt3, 3*sqrt3]
    row3_x = [sqrt3/2]
    
    for x in row0_x: x_coords.append(x); y_coords.append(0)
    for x in row1_x: x_coords.append(x); y_coords.append(1.5)
    for x in row2_x: x_coords.append(x); y_coords.append(3.0)
    for x in row3_x: x_coords.append(x); y_coords.append(4.5)
    
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    
    # Shift to center of mass
    x_coords -= np.mean(x_coords)
    y_coords -= np.mean(y_coords)
    
    x0 = np.column_stack((x_coords, y_coords)).flatten()
    x0 = np.append(x0, 0.0) # phi
    
    # Run optimization
    try:
        res = minimize(objective, x0, method='SLSQP', constraints=constraints, 
                       options={'maxiter': 500, 'ftol': 1e-10, 'disp': False})
        
        if res.success:
            best_vars = res.x
        else:
            best_vars = x0
            
        # Try a few more runs with different initial angles for outer hex
        # Or just use the best found
        
        # Extract result
        centers = best_vars[:24].reshape(12, 2)
        phi = best_vars[24]
        
        # Calculate final s
        s_final = objective(best_vars)
        
        # Construct return arrays
        # Inner: (12, 3) [x, y, angle]
        # Angle for inner hexagons is 0
        inner_hex_data = np.column_stack((centers, np.zeros(12)))
        
        # Outer: (3,) [x, y, angle]
        # Center is (0,0)
        outer_hex_data = np.array([0.0, 0.0, np.rad2deg(phi)])
        
        return inner_hex_data, outer_hex_data, s_final

    except Exception as e:
        # Fallback to initial guess
        centers = x0[:24].reshape(12, 2)
        phi = x0[24]
        s_final = objective(x0)
        inner_hex_data = np.column_stack((centers, np.zeros(12)))
        outer_hex_data = np.array([0.0, 0.0, np.rad2deg(phi)])
        return inner_hex_data, outer_hex_data, s_final


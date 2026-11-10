import numpy as np
from scipy.optimize import minimize_scalar
import time

def hexagon_packing_12():
    start_time = time.time()
    
    # Helper to calculate s for a given set of points and rotation
    def calc_s(points, theta):
        c = np.cos(theta)
        s = np.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        rot_centers = points @ rot.T
        
        # Hexagon vertices relative to center (side 1, angle 0)
        hex_angles = np.deg2rad(np.arange(0, 360, 60))
        hex_vertices = np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])
        
        # Outer hexagon normals (angle 0)
        normal_angles = np.deg2rad(30 + np.arange(0, 360, 60))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        max_proj = -np.inf
        for i in range(len(points)):
            center = rot_centers[i]
            verts = center + (hex_vertices @ rot.T)
            projs = verts @ normals.T
            hex_max_proj = np.max(projs)
            if hex_max_proj > max_proj:
                max_proj = hex_max_proj
        
        s_val = max_proj * 2.0 / np.sqrt(3)
        return s_val

    # Define candidate shapes (axial coordinates)
    shapes = []
    
    # Shape 1: Seed
    seed_axial = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, -1), (-1, 1),
        (2, -1), (-2, 1),
        (2, 0), (-2, 0),
        (1, -2), (-1, 2)
    ]
    shapes.append(seed_axial)
    
    # Shape 2: 3-4-3-2 (Rows)
    # Row 0 (y=0): 4 hexes
    # Row 1 (y=sqrt(3)): 4 hexes (offset)
    # Row 2 (y=2*sqrt(3)): 3 hexes
    # Row 3 (y=3*sqrt(3)): 1 hex? No, 12 total.
    # Let's try 2-4-4-2
    # Axial to Cartesian: x = 1.5q, y = sqrt(3)(r + q/2)
    # We want to generate Cartesian points directly for flexibility?
    # No, stick to axial for easy generation.
    
    # Shape 3: 1 + 6 + 5 (remove one from 2nd ring to break symmetry, see if it packs tighter)
    # 2nd ring: (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
    # Remove (0,2)?
    shape3 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-1, 2), (-2, 1), (-2, 0), (1, -2) # 6 neighbors + 5 from 2nd ring
    ]
    shapes.append(shape3)
    
    best_s = float('inf')
    best_points = None
    best_theta = 0
    
    for axial_coords in shapes:
        points = []
        for q, r in axial_coords:
            x = 1.5 * q
            y = np.sqrt(3) * (r + q / 2.0)
            points.append([x, y])
        points = np.array(points)
        
        # Center
        centroid = np.mean(points, axis=0)
        points = points - centroid
        
        # Optimize rotation
        def objective(theta):
            return calc_s(points, theta)
            
        res = minimize_scalar(objective, bounds=(0, np.pi/3), method='bounded', options={'xatol': 1e-6})
        theta_opt = res.x
        s_opt = objective(theta_opt)
        
        if s_opt < best_s:
            best_s = s_opt
            best_points = points
            best_theta = theta_opt
            
        # Time check
        if time.time() - start_time > 500:
            break

    # Construct result
    c = np.cos(best_theta)
    s = np.sin(best_theta)
    rot = np.array([[c, -s], [s, c]])
    final_points = best_points @ rot.T
    
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = final_points[i, 0]
        inner_hex_data[i, 1] = final_points[i, 1]
        inner_hex_data[i, 2] = np.rad2deg(best_theta)
        
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, best_s

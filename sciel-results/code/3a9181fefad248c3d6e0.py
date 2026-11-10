import numpy as np
from scipy.optimize import minimize_scalar

def hexagon_packing_12():
    # 12 axial coordinates for a compact cluster
    axial_coords = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, -1), (-1, 1),
        (2, -1), (-2, 1),
        (2, 0), (-2, 0),
        (1, -2), (-1, 2)
    ]
    
    # Convert to Cartesian
    points = []
    for q, r in axial_coords:
        x = 1.5 * q
        y = np.sqrt(3) * (r + q / 2.0)
        points.append([x, y])
    points = np.array(points)
    
    # Scale to ensure separation (gap > 1e-6)
    # Distance between adjacent centers is sqrt(3).
    # We need distance > sqrt(3).
    # Scaling by 1.01 gives 1.01 * sqrt(3) > sqrt(3).
    scale = 1.01
    points = points * scale
    
    # Function to calculate minimal enclosing hexagon side length for centers
    def calc_s_centers(theta):
        c = np.cos(theta)
        s = np.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        rotated_points = points @ rot.T
        
        # Projections on 3 axes: 0, 60, 120 degrees
        # Axis vectors
        u0 = np.array([1.0, 0.0])
        u1 = np.array([0.5, np.sqrt(3)/2])
        u2 = np.array([-0.5, np.sqrt(3)/2])
        
        max_proj = -np.inf
        for u in [u0, u1, u2]:
            projs = rotated_points @ u
            w = np.max(projs) - np.min(projs)
            if w > max_proj:
                max_proj = w
                
        s_centers = max_proj / np.sqrt(3)
        return s_centers

    # Minimize s_centers + 1
    def objective(theta):
        return calc_s_centers(theta) + 1.0

    # Optimize theta in [0, pi/3]
    res = minimize_scalar(objective, bounds=(0, np.pi/3), method='bounded')
    theta_opt = res.x
    
    s_opt = objective(theta_opt)
    
    # Construct the result
    # Rotate points by theta_opt
    c = np.cos(theta_opt)
    s = np.sin(theta_opt)
    rot = np.array([[c, -s], [s, c]])
    final_points = points @ rot.T
    
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = final_points[i, 0]
        inner_hex_data[i, 1] = final_points[i, 1]
        inner_hex_data[i, 2] = 0.0  # Angle 0
        
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, s_opt

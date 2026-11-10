import numpy as np
import time

def hexagon_packing_12():
    start_time = time.time()
    deadline = 600.0

    # Helper to convert axial to cartesian
    def axial_to_cartesian(q, r):
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    # Helper to get vertices of a unit hexagon at (cx, cy)
    def get_hex_vertices(cx, cy, angle_deg=0):
        angles = np.deg2rad(angle_deg + np.arange(6) * 60)
        x = cx + np.cos(angles)
        y = cy + np.sin(angles)
        return np.column_stack((x, y))

    # Define candidate configurations (axial coordinates)
    # Config 1: Star missing (2, -2)
    config_1 = [
        (0,0), 
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (0,2), (-2,2), (-2,0), (0,-2)
    ]
    
    # Config 2: Another variation
    config_2 = [
        (0,0),
        (1,0), (-1,0),
        (0,1), (0,-1),
        (1,-1), (-1,1),
        (2,-1), (-2,1),
        (2,0), (-2,0),
        (1,-2), (-1,2)
    ]

    # Function to calculate minimal enclosing hexagon side length s
    def calc_s(config):
        # Convert to Cartesian
        points = np.array([axial_to_cartesian(q, r) for q, r in config])
        
        # Center the cluster at origin
        centroid = np.mean(points, axis=0)
        points = points - centroid
        
        # Precompute vertices of the unit hexagons relative to their centers
        # Hexagon side 1, angle 0.
        # Vertices at angles 0, 60, 120, 180, 240, 300.
        hex_angles = np.deg2rad(np.arange(0, 360, 60))
        hex_vertices = np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])
        
        # Function to calculate s for a given rotation theta
        def calc_s_theta(theta):
            c = np.cos(theta)
            s = np.sin(theta)
            rot = np.array([[c, -s], [s, c]])
            
            # Rotate centers
            rot_centers = points @ rot.T
            
            # Calculate max projection onto the 6 normals of the outer hexagon
            # Outer hexagon normals (outward) for angle 0:
            # Vertices at 0, 60, ... Normals are at 30, 90, ...?
            # Let's check.
            # Regular hexagon with flat top (angle 0 usually means vertex at 0? Or flat edge at 0?)
            # Problem def: "vertices (cx + d cos(a + 60i deg) ...)"
            # If a=0, vertices are at 0, 60, 120...
            # So the "pointy" top/bottom? No, vertex at 0 is on the right.
            # The edges are between vertex i and i+1.
            # Edge normal is perpendicular to the edge.
            # Midpoint of edge 0-1: angle 30 deg.
            # So normals are at 30, 90, 150, 210, 270, 330 degrees.
            
            normal_angles = np.deg2rad(30 + np.arange(0, 360, 60))
            normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
            
            max_proj = -np.inf
            
            # For each inner hexagon
            for i in range(len(config)):
                center = rot_centers[i]
                # Vertices of this hexagon (angle 0 relative to global? 
                # The problem allows angle_deg for inner hexagons. 
                # We can set them to 0. 
                # If we rotate the cluster, we can also rotate the individual hexagons 
                # to maintain their orientation relative to the cluster or global.
                # The seed set angle 0. 
                # If we rotate the centers, should we rotate the hexagons too?
                # Yes, to maintain the packing structure.
                # So the vertices are center + R(theta) * hex_vertices.
                # Or simply, the vertices are at center + hex_vertices rotated by theta?
                # Yes.
                
                verts = center + (hex_vertices @ rot.T)
                
                # Project vertices onto normals
                projs = verts @ normals.T
                # Max projection for this hexagon
                hex_max_proj = np.max(projs)
                
                if hex_max_proj > max_proj:
                    max_proj = hex_max_proj
            
            # s * sqrt(3)/2 = max_proj
            s_val = max_proj * 2.0 / np.sqrt(3)
            return s_val

        # Optimize theta
        # The shape has 6-fold symmetry?
        # We search [0, 60 degrees).
        
        # Use a simple grid search followed by local refinement
        best_s = np.inf
        best_theta = 0
        
        # Coarse grid
        thetas = np.linspace(0, np.pi/3, 100)
        for theta in thetas:
            s_val = calc_s_theta(theta)
            if s_val < best_s:
                best_s = s_val
                best_theta = theta
        
        # Refine around best_theta
        # Use minimize_scalar from scipy if available, otherwise manual
        try:
            from scipy.optimize import minimize_scalar
            res = minimize_scalar(calc_s_theta, bounds=(best_theta - np.pi/60, best_theta + np.pi/60), method='bounded', options={'xatol': 1e-8})
            best_theta = res.x
            best_s = res.fun
        except ImportError:
            pass
            
        # Construct the result
        c = np.cos(best_theta)
        s = np.sin(best_theta)
        rot = np.array([[c, -s], [s, c]])
        final_points = points @ rot.T
        
        inner_hex_data = np.zeros((12, 3))
        for i in range(12):
            inner_hex_data[i, 0] = final_points[i, 0]
            inner_hex_data[i, 1] = final_points[i, 1]
            inner_hex_data[i, 2] = np.rad2deg(best_theta) # Rotate the hexagons as well
        
        # Outer hexagon centered at 0,0.
        # Angle 0.
        outer_hex_data = np.array([0.0, 0.0, 0.0])
        
        return inner_hex_data, outer_hex_data, best_s

    best_result = None
    best_score = -np.inf

    configs = [config_1, config_2]
    
    for config in configs:
        if time.time() > deadline - 60:
            break
        try:
            inner, outer, s = calc_s(config)
            score = 1.0 / s
            if score > best_score:
                best_score = score
                best_result = (inner, outer, s)
        except Exception as e:
            pass

    if best_result is None:
        # Fallback to a simple valid configuration if all fail
        # Just return the seed configuration
        config = config_2
        points = np.array([axial_to_cartesian(q, r) for q, r in config])
        centroid = np.mean(points, axis=0)
        points = points - centroid
        inner = np.zeros((12, 3))
        for i in range(12):
            inner[i, 0] = points[i, 0]
            inner[i, 1] = points[i, 1]
            inner[i, 2] = 0.0
        outer = np.array([0.0, 0.0, 0.0])
        # Estimate s
        max_r = np.max(np.sqrt(points[:, 0]**2 + points[:, 1]**2))
        s = max_r * 2.0 / np.sqrt(3) + 1.0
        best_result = (inner, outer, s)

    return best_result

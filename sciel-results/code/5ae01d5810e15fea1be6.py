import numpy as np
import time
import math
from scipy.optimize import minimize

def hexagon_packing_11():
    start_time = time.time()
    deadline = 630.0
    safety = 15.0

    # Pre-compute base hexagon vertices (angle 0, side 1)
    hex_verts_base = np.array([
        [np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))] for a in range(0, 360, 60)
    ])

    def get_vertices(center, angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        rot = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                        [np.sin(angle_rad), np.cos(angle_rad)]])
        return center + hex_verts_base @ rot.T

    def get_normals(vertices):
        # Returns unit normals for the 6 edges (outward)
        normals = []
        for i in range(6):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % 6]
            edge_vec = v2 - v1
            n = np.array([edge_vec[1], -edge_vec[0]])
            # Ensure outward pointing: dot with (midpoint - center) > 0
            center = np.mean(vertices, axis=0)
            mid = (v1 + v2) / 2
            if np.dot(n, mid - center) < 0:
                n = -n
            n = n / np.linalg.norm(n)
            normals.append(n)
        return np.array(normals)

    def max_penetration(v1, v2):
        # SAT for two convex polygons. Returns max penetration depth (0 if disjoint).
        n1 = get_normals(v1)
        n2 = get_normals(v2)
        all_normals = np.vstack([n1, n2])
        
        max_pen = 0.0
        for n in all_normals:
            proj1 = v1 @ n
            min1, max1 = np.min(proj1), np.max(proj1)
            proj2 = v2 @ n
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # If there is a gap, they are disjoint
            if max1 < min2 or max2 < min1:
                return 0.0
            
            overlap = np.minimum(max1, max2) - np.maximum(min1, min2)
            if overlap > max_pen:
                max_pen = overlap
        return max_pen

    def compute_side(centers, angles, theta_outer_deg):
        # Compute the minimum side length of the outer hexagon containing all inner hexagons
        # for a given outer hexagon orientation.
        # The outer hexagon is centered at (0,0).
        theta_outer_rad = np.deg2rad(theta_outer_deg)
        # Normals of the outer hexagon edges
        # Vertices at 0, 60, ... -> Normals at 30, 90, ...
        # If rotated by theta, normals at theta + 30, theta + 90, ...
        outer_normals = []
        for i in range(6):
            a = theta_outer_rad + np.deg2rad(30 + 60 * i)
            outer_normals.append([np.cos(a), np.sin(a)])
        outer_normals = np.array(outer_normals)
        
        # Get all inner vertices
        all_verts = []
        for i in range(len(centers)):
            verts = get_vertices(centers[i], angles[i])
            all_verts.append(verts)
        all_verts_flat = np.vstack(all_verts)
        
        # Project onto outer normals
        proj = all_verts_flat @ outer_normals.T
        max_proj = np.max(proj, axis=0)
        
        # Apothem = side * sqrt(3) / 2
        # max_proj <= apothem
        # side >= 2 * max_proj / sqrt(3)
        return (2.0 / np.sqrt(3.0)) * np.max(max_proj)

    def objective(params):
        if time.time() > deadline - safety:
            return 1e10

        # params: [x1, y1, a1, ..., x11, y11, a11, theta_outer]
        data = params[:33].reshape(11, 3)
        theta_outer_deg = params[33]
        
        centers = data[:, :2]
        angles = data[:, 2]
        
        # Compute vertices for all inner hexagons
        all_vertices = []
        for i in range(11):
            verts = get_vertices(centers[i], angles[i])
            all_vertices.append(verts)
            
        # Check overlaps
        pen_sum = 0.0
        for i in range(11):
            for j in range(i + 1, 11):
                pen = max_penetration(all_vertices[i], all_vertices[j])
                if pen > 0:
                    # Quadratic penalty for penetration
                    pen_sum += pen * pen
                    
        # Calculate side length
        side = compute_side(centers, angles, theta_outer_deg)
        
        # Penalty for overlaps
        weight = 1000.0
        return side + weight * pen_sum

    # --- Initial Configuration Generation ---
    
    # Base: Center + 6 ring (perfectly touching)
    d = np.sqrt(3)
    ring1_centers = []
    ring1_angles = []
    for i in range(6):
        angle_pos = np.deg2rad(30 + 60 * i)
        cx = d * np.cos(angle_pos)
        cy = d * np.sin(angle_pos)
        ring1_centers.append([cx, cy])
        ring1_angles.append(0.0)
        
    ring1_centers = np.array(ring1_centers)
    ring1_angles = np.array(ring1_angles)
    
    base_centers = np.vstack([np.array([[0, 0]]), ring1_centers])
    base_angles = np.concatenate([np.array([0.0]), ring1_angles])
    
    def make_candidate(r2_angles_deg, r2_dist, r2_rot=0.0, outer_angle=0.0):
        r2_centers = []
        r2_angles = []
        for a in r2_angles_deg:
            rad = np.deg2rad(a)
            cx = r2_dist * np.cos(rad)
            cy = r2_dist * np.sin(rad)
            r2_centers.append([cx, cy])
            r2_angles.append(r2_rot)
            
        r2_centers = np.array(r2_centers)
        r2_angles = np.array(r2_angles)
        
        centers = np.vstack([base_centers, r2_centers])
        angles = np.concatenate([base_angles, r2_angles])
        
        init_params = []
        for i in range(11):
            init_params.extend([centers[i][0], centers[i][1], angles[i]])
        init_params.append(outer_angle)
        return np.array(init_params)

    candidates = []
    
    # Candidate 1: 4 hexagons in gaps (0, 60, 180, 240)
    # This is a standard hexagonal cluster configuration
    candidates.append(make_candidate([0, 60, 180, 240], 2.5))
    
    # Candidate 2: 4 hexagons in gaps (0, 120, 180, 300)
    candidates.append(make_candidate([0, 120, 180, 300], 2.5))
    
    # Candidate 3: 4 hexagons in gaps (60, 120, 240, 300)
    candidates.append(make_candidate([60, 120, 240, 300], 2.5))
    
    # Candidate 4: Asymmetric / different rotation
    candidates.append(make_candidate([0, 90, 180, 270], 2.5))
    
    # Candidate 5: Linear-ish arrangement (inspiration)
    # Try to mimic the 4-3-4 grid but rotated/optimized
    # Grid coords:
    # Row 0: 4 hexes
    # Row 1: 3 hexes
    # Row 2: 4 hexes
    # Center is (1.5, 1.5*sqrt(3)/2 * 2?) No, let's just generate a grid and center it.
    grid_coords = []
    h = math.sqrt(3) / 2
    # Row 0 (y=0)
    for i in range(4):
        grid_coords.append([i * 3.0, 0.0])
    # Row 1 (y=2h)
    for i in range(3):
        grid_coords.append([1.5 + i * 3.0, 2 * h])
    # Row 2 (y=4h)
    for i in range(4):
        grid_coords.append([i * 3.0, 4 * h])
        
    grid_coords = np.array(grid_coords)
    center = np.mean(grid_coords, axis=0)
    grid_coords -= center
    
    # Add slight expansion to avoid initial touching penalties if needed, or rely on optimizer
    # The grid has touching hexes.
    grid_angles = np.zeros(11)
    grid_params = []
    for i in range(11):
        grid_params.extend([grid_coords[i][0], grid_coords[i][1], grid_angles[i]])
    grid_params.append(0.0)
    candidates.append(np.array(grid_params))
    
    # Random restarts
    np.random.seed(42)
    for _ in range(40):
        centers = base_centers.copy()
        angles = base_angles.copy()
        
        r2_centers = []
        r2_angles = []
        for i in range(4):
            angle_pos = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(2.2, 3.2)
            cx = r * np.cos(angle_pos)
            cy = r * np.sin(angle_pos)
            r2_centers.append([cx, cy])
            r2_angles.append(np.random.uniform(0, 60))
            
        r2_centers = np.array(r2_centers)
        r2_angles = np.array(r2_angles)
        
        centers = np.vstack([centers, r2_centers])
        angles = np.concatenate([angles, r2_angles])
        
        init_params = []
        for i in range(11):
            init_params.extend([centers[i][0], centers[i][1], angles[i]])
        init_params.append(np.random.uniform(-30, 30))
        
        candidates.append(np.array(init_params))

    best_result = None
    best_score = float('inf')
    
    for init in candidates:
        if time.time() > deadline - safety:
            break
            
        try:
            # Use Nelder-Mead for robustness with non-smooth constraints
            res = minimize(objective, init, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-9, 'disp': False})
            
            if res.fun < best_score:
                # Check validity
                data = res.x[:33].reshape(11, 3)
                theta_outer_deg = res.x[33]
                
                centers = data[:, :2]
                angles = data[:, 2]
                
                all_vertices = []
                for i in range(11):
                    verts = get_vertices(centers[i], angles[i])
                    all_vertices.append(verts)
                    
                valid = True
                for i in range(11):
                    for j in range(i + 1, 11):
                        pen = max_penetration(all_vertices[i], all_vertices[j])
                        if pen > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    # Calculate final side
                    side = compute_side(centers, angles, theta_outer_deg)
                    
                    if side < best_score:
                        best_score = side
                        # Format output
                        inner_hex_data = np.column_stack([centers, angles])
                        outer_hex_data = np.array([0, 0, theta_outer_deg])
                        best_result = (inner_hex_data, outer_hex_data, best_score)
                        
        except Exception:
            pass

    if best_result is None:
        # Fallback: Return a valid, though not optimal, configuration
        # Use the grid configuration with a large enough outer hex
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
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 10.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length

    return best_result

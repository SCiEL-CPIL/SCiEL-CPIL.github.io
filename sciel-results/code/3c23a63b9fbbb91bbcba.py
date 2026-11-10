import numpy as np
import time
from scipy.optimize import minimize
import itertools

def hexagon_packing_11():
    start_time = time.time()
    deadline = 630.0
    
    # Unit hexagon vertices (angle 0)
    hex_verts_base = np.array([
        [np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))] for a in range(0, 360, 60)
    ])
    
    def get_vertices(center, angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        rot = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                        [np.sin(angle_rad), np.cos(angle_rad)]])
        return center + hex_verts_base @ rot.T

    def get_normals(vertices):
        # Returns unit normals for the 6 edges
        normals = []
        for i in range(6):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % 6]
            edge_vec = v2 - v1
            n = np.array([edge_vec[1], -edge_vec[0]])
            n = n / np.linalg.norm(n)
            normals.append(n)
        return np.array(normals)

    def max_penetration(vertices1, vertices2):
        # SAT for two convex polygons
        # Returns max penetration depth (0 if disjoint)
        max_pen = 0.0
        
        normals1 = get_normals(vertices1)
        normals2 = get_normals(vertices2)
        
        all_normals = np.vstack([normals1, normals2])
        
        for n in all_normals:
            # Project vertices1
            proj1 = vertices1 @ n
            min1 = np.min(proj1)
            max1 = np.max(proj1)
            
            # Project vertices2
            proj2 = vertices2 @ n
            min2 = np.min(proj2)
            max2 = np.max(proj2)
            
            # Check for gap
            if max1 < min2 or max2 < min1:
                return 0.0 # Disjoint
            
            # Calculate overlap
            overlap = np.minimum(max1, max2) - np.maximum(min1, min2)
            if overlap > max_pen:
                max_pen = overlap
                
        return max_pen

    def objective(params):
        if time.time() > deadline - 10:
            return 1e10
            
        # params: [x1, y1, a1, ..., x11, y11, a11, theta_outer]
        # a_i in degrees
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
                    pen_sum += pen * pen
                    
        # Calculate side length of outer hexagon
        # Outer hexagon is centered at origin, angle theta_outer_deg
        # We need the support function of the union of inner hexagons
        
        theta_outer_rad = np.deg2rad(theta_outer_deg)
        outer_normals = []
        for i in range(6):
            a = theta_outer_rad + np.deg2rad(60 * i)
            outer_normals.append([np.cos(a), np.sin(a)])
        outer_normals = np.array(outer_normals)
        
        # Collect all vertices
        all_verts_flat = np.vstack(all_vertices)
        
        # Project all vertices onto outer normals
        proj = all_verts_flat @ outer_normals.T
        max_proj = np.max(proj, axis=0)
        
        # The side length S of a regular hexagon with circumradius R (distance from center to vertex)
        # is S = R * sqrt(3).
        # The support function h(n) gives the distance from center to the supporting line.
        # For a regular hexagon centered at origin, the distance from center to edge (apothem) is ap = S * sqrt(3) / 2.
        # The distance from center to vertex is R = 2 * ap / sqrt(3) = S.
        # Wait.
        # Vertices are at distance R.
        # Edge midpoint distance is R * cos(30) = R * sqrt(3)/2.
        # So h(n) for the outer hexagon in direction n (which is a normal to an edge) is the apothem.
        # So apothem_outer = max_proj.
        # S = 2 * apothem_outer / sqrt(3).
        
        side = (2.0 / np.sqrt(3.0)) * np.max(max_proj)
        
        # Penalty for overlaps
        weight = 1000.0
        return side + weight * pen_sum

    # Initial candidates
    d = np.sqrt(3)
    base_centers = np.array([[0, 0]])
    base_angles = np.array([0.0])
    
    # Ring 1: 6 hexagons
    ring1_centers = []
    ring1_angles = []
    for i in range(6):
        angle_pos = np.deg2rad(30 + 60 * i)
        cx = d * np.cos(angle_pos)
        cy = d * np.sin(angle_pos)
        ring1_centers.append([cx, cy])
        ring1_angles.append(0.0) # Aligned to share edges with center
        
    ring1_centers = np.array(ring1_centers)
    ring1_angles = np.array(ring1_angles)
    
    base_centers = np.vstack([base_centers, ring1_centers])
    base_angles = np.concatenate([base_angles, ring1_angles])
    
    # Ring 2: 4 hexagons
    # Place them in the gaps. The gaps are at angles 0, 60, 120, 180, 240, 300.
    # Let's pick 4 of these 6 positions.
    # Distance?
    # The vertex of the center hex is at (1,0).
    # The vertex of the neighbor at 30 deg is at (1.5, 0.866) + (0.5, 0.866) = (2, 1.732)?
    # Let's just estimate.
    # Try distance 3.0.
    candidates = []
    
    gap_angles = [0, 60, 120, 180, 240, 300]
    
    # Try different subsets of 4 gaps
    for subset in itertools.combinations(range(6), 4):
        r2_centers = []
        r2_angles = []
        for idx in subset:
            angle_pos = np.deg2rad(gap_angles[idx])
            # Heuristic distance
            cx = 3.0 * np.cos(angle_pos)
            cy = 3.0 * np.sin(angle_pos)
            r2_centers.append([cx, cy])
            # Orient them to point outward or align?
            # Let's try aligning with the gap direction
            r2_angles.append(gap_angles[idx])
            
        r2_centers = np.array(r2_centers)
        r2_angles = np.array(r2_angles)
        
        centers = np.vstack([base_centers, r2_centers])
        angles = np.concatenate([base_angles, r2_angles])
        
        # Flatten: x, y, angle for each
        init_params = []
        for i in range(11):
            init_params.extend([centers[i][0], centers[i][1], angles[i]])
        init_params.append(0.0) # Outer angle
        
        candidates.append(np.array(init_params))
        
    # Add random candidates
    np.random.seed(42)
    for _ in range(20):
        centers = base_centers.copy()
        angles = base_angles.copy()
        
        r2_centers = []
        r2_angles = []
        for i in range(4):
            angle_pos = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(2.5, 4.0)
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
        if time.time() > deadline - 60:
            break
            
        try:
            res = minimize(objective, init, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False})
            
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
                    theta_outer_rad = np.deg2rad(theta_outer_deg)
                    outer_normals = []
                    for i in range(6):
                        a = theta_outer_rad + np.deg2rad(60 * i)
                        outer_normals.append([np.cos(a), np.sin(a)])
                    outer_normals = np.array(outer_normals)
                    
                    all_verts_flat = np.vstack(all_vertices)
                    proj = all_verts_flat @ outer_normals.T
                    max_proj = np.max(proj, axis=0)
                    side = (2.0 / np.sqrt(3.0)) * np.max(max_proj)
                    
                    if side < best_score:
                        best_score = side
                        # Format output
                        inner_hex_data = np.column_stack([centers, angles])
                        outer_hex_data = np.array([0, 0, theta_outer_deg])
                        best_result = (inner_hex_data, outer_hex_data, best_score)
                        
        except Exception as e:
            pass

    if best_result is None:
        # Fallback
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
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length

    return best_result

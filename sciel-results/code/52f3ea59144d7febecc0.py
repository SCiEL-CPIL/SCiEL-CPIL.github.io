    import numpy as np
    from scipy.optimize import minimize, minimize_scalar
    import time

    N = 12

    def hex_vertices(cx, cy, s, angle_deg):
        angle = np.radians(angle_deg)
        angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180 + angle
        xs = cx + s * np.cos(angles)
        ys = cy + s * np.sin(angles)
        return np.column_stack([xs, ys])

    def find_min_hex(vertices):
        vertices = np.array(vertices)
        c = np.mean(vertices, axis=0)
        
        def apothem_for_angle(theta_rad):
            # Normals for a regular hexagon with orientation theta
            # Vertices at theta, theta+60...
            # Normals to sides are at theta+30, theta+90, theta+150
            n_angles = np.deg2rad([30, 90, 150]) + theta_rad
            normals = np.column_stack([np.cos(n_angles), np.sin(n_angles)])
            
            diff = vertices - c
            projections = np.abs(diff @ normals.T)
            max_proj = np.max(projections)
            return max_proj

        result = minimize_scalar(apothem_for_angle, bounds=(0, np.pi/3), method='bounded', options={'xatol': 1e-6})
        theta_opt = result.x
        R_opt = result.fun
        side = 2 * R_opt / np.sqrt(3)
        
        return side, c[0], c[1], np.rad2deg(theta_opt)

    def construct():
        # Define lattice vectors for unit hexagons (flat-top, side 1)
        # Distance between centers is sqrt(3)
        d = np.sqrt(3)
        v1 = np.array([d, 0])
        v2 = np.array([d/2, 1.5]) # d * sin(60) = sqrt(3) * 1.5? No. sin(60)=sqrt(3)/2. d*sqrt(3)/2 = 3/2 = 1.5. Correct.
        
        best_side = float('inf')
        best_result = None
        
        # Candidate 1: 3x4 Grid
        # 3 steps in v1, 4 steps in v2? Or 4 in v1, 3 in v2?
        # Let's try 4 in v1, 3 in v2.
        positions_1 = []
        for i in range(4):
            for j in range(3):
                positions_1.append(i*v1 + j*v2)
        
        # Candidate 2: 3 in v1, 4 in v2
        positions_2 = []
        for i in range(3):
            for j in range(4):
                positions_2.append(i*v1 + j*v2)
                
        # Candidate 3: 2x6
        positions_3 = []
        for i in range(6):
            for j in range(2):
                positions_3.append(i*v1 + j*v2)
                
        # Candidate 4: 6x2
        positions_4 = []
        for i in range(2):
            for j in range(6):
                positions_4.append(i*v1 + j*v2)
                
        # Candidate 5: Compact cluster
        # Center at 0,0
        # Ring 1: 6 hexes at distance d, angles 0, 60, ...
        # Ring 2: add 5 more to make 12?
        # This is complex to generate generically. Stick to grids.
        
        candidates = [positions_1, positions_2, positions_3, positions_4]
        
        for pos_list in candidates:
            # Center the configuration
            center = np.mean(pos_list, axis=0)
            pos_list = [p - center for p in pos_list]
            
            # Get all vertices
            all_verts = []
            for p in pos_list:
                v = hex_vertices(p[0], p[1], 1.0, 0.0)
                all_verts.append(v)
            all_verts = np.vstack(all_verts)
            
            side, cx, cy, angle = find_min_hex(all_verts)
            
            if side < best_side:
                best_side = side
                # Store the inner positions and angles (all 0)
                inner = [[p[0], p[1], 0.0] for p in pos_list]
                best_result = (inner, (cx, cy), side, angle)
                
        return best_result
    
import numpy as np
import time

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Unit hexagon vertices (pointy-top, angle 0 degrees)
    hex_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
    unit_hex_vertices = np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])
    
    # Hexagonal lattice for touching pointy-top hexagons
    d = np.sqrt(3.0)  # distance between centers of touching hexagons
    a1 = np.array([d, 0.0])
    a2 = np.array([d * 0.5, d * np.sqrt(3.0) / 2.0])
    
    def hex_to_cartesian(q, r):
        return q * a1 + r * a2
    
    # Generate multiple candidate configurations
    configs = []
    
    # Hexagonal cluster radius 1 (7 hex) + 4 extensions
    base7 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    configs.append(base7 + [(2,0), (2,-1), (1,1), (-2,2)])
    configs.append(base7 + [(-2,0), (-2,1), (-1,-1), (2,-2)])
    configs.append(base7 + [(1,2), (2,1), (2,0), (1,-1)])
    configs.append(base7 + [(-1,-2), (-2,-1), (-2,0), (-1,1)])
    
    # Row-based configurations
    for row_counts in [[3,5,3], [2,4,5], [4,3,4], [3,4,4], [4,4,3], [5,3,3], [3,3,5], [2,5,4], [5,2,4], [4,5,2]]:
        config = []
        for r, count in enumerate(row_counts):
            start_q = -count // 2
            for i in range(count):
                config.append((start_q + i, r))
        if len(config) == N:
            configs.append(config)
    
    # Two-row configs
    for c1, c2 in [(5,6), (6,5), (4,7), (7,4), (3,8), (8,3)]:
        config = []
        for i in range(c1):
            config.append((i - c1//2, 0))
        for i in range(c2):
            config.append((i - c2//2, 1))
        configs.append(config)
    
    # Three-row more varied
    for c1, c2, c3 in [(1,9,1), (9,1,1), (1,1,9), (2,7,2), (7,2,2), (2,2,7), (6,3,2), (3,6,2), (2,3,6), (6,2,3), (3,2,6), (2,6,3)]:
        config = []
        for r, count in enumerate([c1, c2, c3]):
            start_q = -count // 2
            for i in range(count):
                config.append((start_q + i, r))
        if len(config) == N:
            configs.append(config)
    
    # Linear
    config_lin = [(i - 5, 0) for i in range(11)]
    configs.append(config_lin)
    
    best_side = float('inf')
    best_inner = None
    best_center = None
    best_angle = 0.0
    
    for config in configs:
        if time.time() > deadline:
            break
        if len(config) != N:
            continue
        
        centers = np.array([hex_to_cartesian(q, r) for q, r in config])
        
        # Try both pointy-top and flat-top inner hexagons
        for inner_rot in [0.0, 30.0]:
            if time.time() > deadline:
                break
            
            if inner_rot == 0.0:
                verts_template = unit_hex_vertices
            else:
                rot = np.radians(inner_rot)
                R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
                verts_template = unit_hex_vertices @ R.T
            
            all_vertices = np.vstack([verts_template + c for c in centers])
            
            # Try multiple enclosing hexagon orientations
            for phi_deg in range(0, 60, 3):
                if time.time() > deadline:
                    break
                phi = np.radians(phi_deg)
                normals = np.column_stack([
                    np.cos(phi + k * np.pi / 3),
                    np.sin(phi + k * np.pi / 3)
                ] for k in range(6))
                
                projections = all_vertices @ normals.T
                proj_max = projections.max(axis=0)
                proj_min = projections.min(axis=0)
                ranges = proj_max - proj_min
                s_candidate = np.max(ranges) / np.sqrt(3.0)
                
                if s_candidate >= best_side:
                    continue
                
                # Find center via least squares
                targets = (proj_max + proj_min) / 2.0
                c_sol, residuals, rank, sv = np.linalg.lstsq(normals, targets, rcond=None)
                
                # Verify and compute actual required s
                max_viol = 0.0
                for v in all_vertices:
                    for i in range(6):
                        val = np.dot(v - c_sol, normals[i])
                        if val > 0:
                            max_viol = max(max_viol, val)
                
                s_actual = max_viol / (np.sqrt(3.0) / 2.0)
                
                if s_actual < best_side:
                    best_side = s_actual
                    best_inner = [[float(c[0]), float(c[1]), inner_rot] for c in centers]
                    best_center = (float(c_sol[0]), float(c_sol[1]))
                    best_angle = float(phi_deg)
    
    if best_inner is None:
        # Fallback
        centers = np.array([hex_to_cartesian(q, r) for q, r in configs[0]])
        all_vertices = np.vstack([unit_hex_vertices + c for c in centers])
        x_min, x_max = all_vertices[:, 0].min(), all_vertices[:, 0].max()
        y_min, y_max = all_vertices[:, 1].min(), all_vertices[:, 1].max()
        c = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
        s = max(x_max - x_min, y_max - y_min) / np.sqrt(3.0) * 1.2
        best_inner = [[float(c[0]), float(c[1]), 0.0] for c in centers]
        best_center = (float(c[0]), float(c[1]))
        best_angle = 0.0
        best_side = s
    
    return best_inner, best_center, float(best_side), float(best_angle)


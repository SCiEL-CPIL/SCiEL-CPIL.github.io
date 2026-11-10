import numpy as np
import time

def hexagon_packing_11():
    start_time = time.time()
    deadline = 600 - 10  # 590s safety margin

    # Configuration: 1 center, 6 ring 1, 4 ring 2
    centers = [
        (0, 0),
        (2, 0), (-2, 0),
        (1, np.sqrt(3)), (-1, np.sqrt(3)),
        (1, -np.sqrt(3)), (-1, -np.sqrt(3)),
        (3, np.sqrt(3)), (-3, np.sqrt(3)),
        (3, -np.sqrt(3)), (-3, -np.sqrt(3))
    ]
    
    inner_hex_data = np.array([[x, y, 0.0] for x, y in centers])
    
    # Generate vertices
    vertices = []
    for cx, cy in centers:
        for i in range(6):
            angle = np.deg2rad(60 * i)
            vx = cx + np.cos(angle)
            vy = cy + np.sin(angle)
            vertices.append((vx, vy))
    vertices = np.array(vertices)
    
    best_S = float('inf')
    best_a = 0.0
    
    # Coarse search
    for a_deg in np.linspace(0, 60, 300):
        if time.time() > deadline:
            break
        a_rad = np.deg2rad(a_deg)
        normal_angles = a_rad + np.deg2rad(30) + np.deg2rad(60) * np.arange(6)
        
        # Project all vertices onto all normals
        # verts (N, 2), normals (6, 2)
        # proj (N, 6)
        cos_n = np.cos(normal_angles)
        sin_n = np.sin(normal_angles)
        proj = vertices[:, 0:1] * cos_n[None, :] + vertices[:, 1:2] * sin_n[None, :]
        max_proj = np.max(proj)
        
        S = 2 * max_proj / np.sqrt(3)
        if S < best_S:
            best_S = S
            best_a = a_deg
            
    # Fine search around best_a
    if time.time() < deadline:
        range_fine = 2.0
        step_fine = 0.01
        a_start = max(0, best_a - range_fine)
        a_end = min(60, best_a + range_fine)
        for a_deg in np.linspace(a_start, a_end, int((a_end - a_start) / step_fine)):
            if time.time() > deadline:
                break
            a_rad = np.deg2rad(a_deg)
            normal_angles = a_rad + np.deg2rad(30) + np.deg2rad(60) * np.arange(6)
            cos_n = np.cos(normal_angles)
            sin_n = np.sin(normal_angles)
            proj = vertices[:, 0:1] * cos_n[None, :] + vertices[:, 1:2] * sin_n[None, :]
            max_proj = np.max(proj)
            S = 2 * max_proj / np.sqrt(3)
            if S < best_S:
                best_S = S
                best_a = a_deg

    outer_hex_data = np.array([0.0, 0.0, best_a])
    
    return inner_hex_data, outer_hex_data, float(best_S)

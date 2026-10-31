import numpy as np
import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0  # 120s limit, keep margin
    
    # Define unit hexagon vertices (pointy top, side 1)
    # Vertices at angles 30, 90, 150, 210, 270, 330? No, standard pointy top:
    # (0, 1), (sqrt(3)/2, 1/2), (sqrt(3)/2, -1/2), (0, -1), (-sqrt(3)/2, -1/2), (-sqrt(3)/2, 1/2)
    # Let's use flat top for consistency with lattice?
    # Flat top vertices: (1, 0), (1/2, sqrt(3)/2), (-1/2, sqrt(3)/2), (-1, 0), (-1/2, -sqrt(3)/2), (1/2, -sqrt(3)/2)
    # Let's stick to one orientation for inner hexagons, say Flat Top (angle 0).
    
    hex_verts = np.array([
        [1.0, 0.0],
        [0.5, math.sqrt(3)/2],
        [-0.5, math.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -math.sqrt(3)/2],
        [0.5, -math.sqrt(3)/2]
    ])
    
    # Define candidate center configurations
    # Lattice vectors for hexagonal packing of flat-top hexagons
    # Distance between centers is sqrt(3)
    # Vector u = (sqrt(3), 0)
    # Vector v = (sqrt(3)/2, 1.5)
    
    u = np.array([math.sqrt(3), 0.0])
    v = np.array([math.sqrt(3)/2, 1.5])
    
    candidates = []
    
    # Config 1: 4-3-4
    # Row 0 (y=0): 4 hexes
    # Row 1 (y=1.5): 3 hexes
    # Row 2 (y=3.0): 4 hexes
    c1 = []
    # Center the rows
    # Row 0: x = -1.5, -0.5, 0.5, 1.5 (in units of sqrt(3)/2? No, let's use lattice coords)
    # Lattice coords (i, j) -> pos = i*u + j*v
    # Let's just place them manually in x,y
    # Spacing x = sqrt(3)
    # Spacing y = 1.5
    # Offset x for alternating rows = sqrt(3)/2
    
    s3 = math.sqrt(3)
    
    # 4-3-4
    # y=0: x = -1.5*s3, -0.5*s3, 0.5*s3, 1.5*s3
    # y=1.5: x = -s3, 0, s3
    # y=3.0: x = -1.5*s3, -0.5*s3, 0.5*s3, 1.5*s3
    centers_1 = []
    for x in [-1.5*s3, -0.5*s3, 0.5*s3, 1.5*s3]:
        centers_1.append((x, 0.0))
    for x in [-s3, 0.0, s3]:
        centers_1.append((x, 1.5))
    for x in [-1.5*s3, -0.5*s3, 0.5*s3, 1.5*s3]:
        centers_1.append((x, 3.0))
    candidates.append(centers_1)
    
    # Config 2: 3-4-3-1 (Asymmetric, might pack tighter in one direction)
    # y=0: 3
    # y=1.5: 4
    # y=3.0: 3
    # y=4.5: 1
    centers_2 = []
    for x in [-s3, 0.0, s3]:
        centers_2.append((x, 0.0))
    for x in [-1.5*s3, -0.5*s3, 0.5*s3, 1.5*s3]:
        centers_2.append((x, 1.5))
    for x in [-s3, 0.0, s3]:
        centers_2.append((x, 3.0))
    centers_2.append((0.0, 4.5))
    candidates.append(centers_2)
    
    # Config 3: Linear (Fallback)
    centers_3 = []
    for k in range(-5, 6):
        centers_3.append((k * s3, 0.0))
    candidates.append(centers_3)
    
    # Pre-compute all vertices for each candidate (unrotated)
    # We will rotate the whole cluster
    all_vertices_list = []
    for centers in candidates:
        verts = []
        for cx, cy in centers:
            for vx, vy in hex_verts:
                verts.append((cx + vx, cy + vy))
        all_vertices_list.append(np.array(verts))
        
    # Outer hexagon normals (Pointy Top, centered at 0,0)
    # Normals at 0, 60, 120, 180, 240, 300 degrees
    # Wait, if outer is Pointy Top, vertices are at 90, 150...
    # Normals are perpendicular to sides.
    # Sides are at 0, 60... ?
    # Let's just use 6 normals at 0, 60, 120, 180, 240, 300.
    # The condition is max projection <= S * sqrt(3)/2
    normals = np.array([
        [math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] for i in range(6)
    ])
    
    best_S = float('inf')
    best_theta = 0.0
    best_centers = candidates[0]
    
    # Search for optimal rotation theta
    # Symmetry: 30 degrees is enough for regular hexagon
    # Step size: 0.1 degrees
    for i in range(301):
        theta_deg = i * 0.1
        if time.time() > deadline - 5:
            break
        theta = math.radians(theta_deg)
        c = math.cos(theta)
        s = math.sin(theta)
        R = np.array([[c, -s], [s, c]])
        
        current_min_S = 0.0
        for verts in all_vertices_list:
            # Rotate vertices
            rot_verts = verts @ R.T
            
            # Project onto normals
            # projections = rot_verts @ normals.T
            # max_proj = np.max(projections, axis=0)
            # S_required = max(max_proj) * 2 / sqrt(3)
            
            # Efficient calculation
            max_proj = 0.0
            for n in normals:
                proj = np.dot(rot_verts, n)
                m = np.max(proj)
                if m > max_proj:
                    max_proj = m
            
            S_req = max_proj * 2.0 / math.sqrt(3)
            if S_req > current_min_S:
                current_min_S = S_req
                
        if current_min_S < best_S:
            best_S = current_min_S
            best_theta = theta
            # We need to know which candidate was best
            # Actually, the loop structure above calculates min_S for ALL candidates?
            # No, the inner loop `for verts in all_vertices_list` calculates S for EACH candidate and takes the MAX?
            # No, I want the MIN S among candidates.
            # Let's restructure.
            
            pass # Logic error in draft above.
    
    # Correct Logic:
    best_S = float('inf')
    best_theta = 0.0
    best_center_idx = 0
    
    for i in range(301):
        theta_deg = i * 0.1
        if time.time() > deadline - 5:
            break
        theta = math.radians(theta_deg)
        c = math.cos(theta)
        s = math.sin(theta)
        R = np.array([[c, -s], [s, c]])
        
        min_S_for_theta = float('inf')
        min_idx_for_theta = 0
        
        for idx, verts in enumerate(all_vertices_list):
            rot_verts = verts @ R.T
            
            max_proj = 0.0
            for n in normals:
                proj = np.dot(rot_verts, n)
                m = np.max(proj)
                if m > max_proj:
                    max_proj = m
            
            S_req = max_proj * 2.0 / math.sqrt(3)
            if S_req < min_S_for_theta:
                min_S_for_theta = S_req
                min_idx_for_theta = idx
                
        if min_S_for_theta < best_S:
            best_S = min_S_for_theta
            best_theta = theta
            best_center_idx = min_idx_for_theta
            
    # Construct result
    centers = candidates[best_center_idx]
    c = math.cos(best_theta)
    s = math.sin(best_theta)
    
    inner = []
    for cx, cy in centers:
        # Rotate center
        rx = cx * c - cy * s
        ry = cx * s + cy * c
        inner.append([rx, ry, 0.0])
        
    return inner, (0.0, 0.0), best_S, 0.0
# EVOLVE-BLOCK-END

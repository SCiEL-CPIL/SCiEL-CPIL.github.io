import numpy as np
import time

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Configuration: 7-hexagon flower + 4 hexagons attached
    # This ensures a compact shape.
    base_axial = [(0,0)]
    ring1 = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    base_axial.extend(ring1)
    extra = [(2,0), (2,-1), (3,-1), (3,-2)]
    base_axial.extend(extra)
    
    def axial_to_cartesian(q, r):
        x = np.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        return x, y

    points = [axial_to_cartesian(q, r) for q, r in base_axial]
    
    # Center the cluster
    cx = np.mean([p[0] for p in points])
    cy = np.mean([p[1] for p in points])
    centered_points = [(p[0]-cx, p[1]-cy) for p in points]
    
    # Scale to ensure disjointness.
    # Distance between centers in grid is sqrt(3).
    # Vertex distance is 1.
    # To touch vertices, distance must be 2.
    # Scale factor = 2 / sqrt(3).
    # Add small buffer for safety.
    scale_factor = (2.0 / np.sqrt(3.0)) * 1.01
    scaled_points = [(x * scale_factor, y * scale_factor) for x, y in centered_points]
    
    # Generate vertices for all inner hexagons
    hex_verts_rel = [(np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))) for a in range(0, 360, 60)]
    
    all_vertices = []
    for px, py in scaled_points:
        for vx, vy in hex_verts_rel:
            all_vertices.append((px+vx, py+vy))
            
    all_vertices = np.array(all_vertices)
    
    best_S = float('inf')
    best_angle = 0.0
    
    # Search for optimal outer hexagon rotation
    for angle_deg in np.arange(0, 60, 1):
        if time.time() > deadline:
            break
        angle_rad = np.deg2rad(angle_deg)
        
        max_proj = 0.0
        for k in range(6):
            a = angle_rad + np.deg2rad(k*60)
            nx = np.cos(a)
            ny = np.sin(a)
            projs = all_vertices[:, 0] * nx + all_vertices[:, 1] * ny
            curr_max = np.max(projs)
            if curr_max > max_proj:
                max_proj = curr_max
        
        # Add safety margin to S to avoid floating point containment errors
        S_required = (2.0 / np.sqrt(3.0)) * max_proj * 1.001
        
        if S_required < best_S:
            best_S = S_required
            best_angle = angle_deg
            
    if best_S == float('inf'):
        best_S = 10.0
        best_angle = 0.0
        
    inner = [[x, y, 0.0] for x, y in scaled_points]
    
    return inner, (0.0, 0.0), best_S, best_angle

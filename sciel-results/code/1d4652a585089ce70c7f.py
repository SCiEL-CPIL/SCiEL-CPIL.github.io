import numpy as np
import time

N = 11

# Axial coordinates for a compact cluster of 11 hexagons
# Center + 6 neighbors + 4 more
axial_coords = [
    (0, 0),
    (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
    (2, 0), (2, -1), (1, 1), (0, 2)
]

def construct():
    start_time = time.time()
    deadline = 150.0
    
    # 1. Generate centers
    centers = []
    for q, r in axial_coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
    
    # 2. Generate vertices for all inner hexagons
    # Unit hexagon vertices (side 1)
    angles = np.arange(6) * np.pi / 3.0
    hex_vertices_rel = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    
    all_vertices = []
    for cx, cy in centers:
        verts = hex_vertices_rel + np.array([cx, cy])
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    # 3. Define objective function for outer hexagon rotation
    def calc_side_length(phi):
        # Normals for outer hexagon
        normal_angles = phi + np.arange(6) * np.pi / 3.0
        normals = np.stack([np.cos(normal_angles), np.sin(normal_angles)], axis=1)
        
        # Project all vertices onto normals
        projections = all_vertices @ normals.T
        max_projections = np.max(np.abs(projections), axis=0)
        max_proj = np.max(max_projections)
        
        # Apothem A = max_proj
        # Side S = 2 * A / sqrt(3)
        return 2.0 * max_proj / np.sqrt(3.0)
    
    # 4. Optimize phi
    best_phi = 0.0
    best_S = float('inf')
    
    # Grid search
    phis = np.linspace(0, np.pi / 3.0, 100)
    for phi in phis:
        S = calc_side_length(phi)
        if S < best_S:
            best_S = S
            best_phi = phi
            
    # Refine
    if time.time() < deadline - 10:
        step = np.pi / 3.0 / 100
        fine_phis = np.linspace(best_phi - 2*step, best_phi + 2*step, 100)
        for phi in fine_phis:
            S = calc_side_length(phi)
            if S < best_S:
                best_S = S
                best_phi = phi

    # 5. Construct return values
    inner = [[cx, cy, 0.0] for cx, cy in centers]
    center = (0.0, 0.0)
    side = best_S
    angle = np.degrees(best_phi)
    
    return inner, center, side, angle

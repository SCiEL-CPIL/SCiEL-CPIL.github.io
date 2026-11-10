import numpy as np
import time

def _hex_vertices(cx, cy, d, a_deg):
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.array([cx + d * np.cos(angles), cy + d * np.sin(angles)]).T

def hexagon_packing_12():
    start_time = time.time()
    
    # Generate 12 hexagon centers in a 3x4 hex grid
    # Add small gap to ensure separation > 1e-6
    gap = 1e-4
    dx = np.sqrt(3) + gap
    dy = 1.5 + gap/2 # Adjust dy slightly too to maintain hex symmetry?
    # Actually, just scaling the whole cluster might be safer.
    # Let's generate standard centers and then scale them from the centroid.
    
    centers = []
    for r in range(3):
        for c in range(4):
            x = c * np.sqrt(3) + (r % 2) * (np.sqrt(3) / 2)
            y = r * 1.5
            centers.append((x, y))
            
    centers = np.array(centers)
    mean_center = np.mean(centers, axis=0)
    centers = centers - mean_center
    
    # Scale centers to create gap
    # The distance between neighbors is sqrt(3).
    # We want distance > sqrt(3).
    # Scale factor 1 + gap/sqrt(3)
    scale_factor = 1 + 1e-4 / np.sqrt(3)
    centers = centers * scale_factor

    min_s = np.inf
    best_angle = 0
    
    for outer_angle_deg in np.linspace(0, 30, 100):
        normal_angles = np.deg2rad(outer_angle_deg + 30 + 60 * np.arange(6))
        normals = np.array([np.cos(normal_angles), np.sin(normal_angles)]).T
        
        all_vertices = []
        for cx, cy in centers:
            verts = _hex_vertices(cx, cy, 1.0, 0.0)
            all_vertices.append(verts)
        
        all_vertices = np.vstack(all_vertices)
        
        projections = all_vertices @ normals
        max_projs = np.max(projections, axis=0)
        R = np.max(max_projs)
        s = 2 * R / np.sqrt(3)
        
        if s < min_s:
            min_s = s
            best_angle = outer_angle_deg
            
        if time.time() - start_time > 590:
            break

    inner = np.zeros((12, 3))
    for i, (cx, cy) in enumerate(centers):
        inner[i] = [cx, cy, 0.0]
        
    outer = np.array([0.0, 0.0, best_angle])
    
    return inner, outer, min_s

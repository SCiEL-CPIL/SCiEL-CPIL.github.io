import numpy as np
from itertools import combinations
import time

def hexagon_packing_11():
    start_time = time.time()
    deadline = 600.0
    
    scale = 1.001
    a = np.array([1.5 * scale, np.sqrt(3)/2 * scale])
    b = np.array([1.5 * scale, -np.sqrt(3)/2 * scale])
    
    base_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)
    ]
    
    def hex_neighbors(i, j):
        return [
            (i+1, j), (i-1, j),
            (i, j+1), (i, j-1),
            (i+1, j-1), (i-1, j+1)
        ]
        
    ring2 = set()
    for i, j in base_coords[1:]:
        for ni, nj in hex_neighbors(i, j):
            if (ni, nj) not in base_coords:
                ring2.add((ni, nj))
                
    ring2_list = list(ring2)
    
    def get_vertices(coords):
        verts = []
        for i, j in coords:
            cx = i * a[0] + j * b[0]
            cy = i * a[1] + j * b[1]
            for k in range(6):
                angle = np.deg2rad(60*k)
                vx = cx + np.cos(angle)
                vy = cy + np.sin(angle)
                verts.append([vx, vy])
        return np.array(verts)
        
    base_verts = get_vertices(base_coords)
    ring2_verts = {idx: get_vertices([coord]) for idx, coord in enumerate(ring2_list)}
    
    best_S = float('inf')
    best_config = None
    
    for combo in combinations(range(len(ring2_list)), 4):
        if time.time() > deadline - 10:
            break
            
        extra_coords = [ring2_list[i] for i in combo]
        all_coords = base_coords + extra_coords
        extra_verts = np.vstack([ring2_verts[i] for i in combo])
        all_verts = np.vstack([base_verts, extra_verts])
        
        thetas = np.linspace(0, np.pi/3, 601)
        angles = thetas[:, None] + np.deg2rad(30 + 60*np.arange(6))[None, :]
        normals = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
        
        projections = np.einsum('ij,ik->ijk', normals, all_verts)
        max_proj = np.max(projections, axis=2)
        min_proj = np.min(projections, axis=2)
        widths = max_proj - min_proj
        S_vals = np.max(widths, axis=1) / np.sqrt(3)
        
        min_S_idx = np.argmin(S_vals)
        min_S = S_vals[min_S_idx]
        
        if min_S < best_S:
            best_S = min_S
            best_theta = thetas[min_S_idx]
            best_combo = combo
            best_verts = all_verts
            best_coords = all_coords
            best_max_proj = max_proj[min_S_idx]
            best_min_proj = min_proj[min_S_idx]
            best_normals = normals[min_S_idx]
            
    if best_config is None and best_S == float('inf'):
         # Fallback
         return np.zeros((11,3)), np.array([0,0,0]), 10.0

    midpoints = (best_max_proj + best_min_proj) / 2
    c, _, _, _ = np.linalg.lstsq(best_normals, midpoints, rcond=None)
    
    inner_hex_data = np.zeros((11, 3))
    for idx, (i, j) in enumerate(best_coords):
        cx = i * a[0] + j * b[0]
        cy = i * a[1] + j * b[1]
        inner_hex_data[idx] = [cx, cy, 0]
        
    outer_hex_data = np.array([c[0], c[1], np.rad2deg(best_theta)])
    
    return inner_hex_data, outer_hex_data, best_S

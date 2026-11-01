import numpy as np
import time
import math

N = 12

def construct():
    start_time = time.time()
    time_limit = 100.0
    
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    hex_verts = np.array([np.cos(angles), np.sin(angles)]).T
    
    outer_normals = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2]
    ])
    
    def axial_to_cartesian(q, r):
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return np.array([x, y])
    
    centers = []
    for r in range(3):
        for q in range(4):
            c = axial_to_cartesian(q, r)
            centers.append(c)
            
    centers = np.array(centers)
    mean_c = np.mean(centers, axis=0)
    centers = centers - mean_c
    
    def calc_S(centers):
        pts = centers[:, np.newaxis, :] + hex_verts[np.newaxis, :, :]
        pts = pts.reshape(-1, 2)
        projections = pts @ outer_normals.T
        max_proj = np.max(np.abs(projections))
        S = (2.0 / np.sqrt(3.0)) * max_proj
        return S
    
    def check_collisions(centers, moved_idx):
        for j in range(N):
            if j == moved_idx:
                continue
            d = np.linalg.norm(centers[moved_idx] - centers[j])
            if d < np.sqrt(3.0) - 1e-9:
                return False
        return True
    
    current_centers = centers.copy()
    current_S = calc_S(current_centers)
    best_centers = current_centers.copy()
    best_S = current_S
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        idx = np.random.randint(N)
        old_c = current_centers[idx].copy()
        
        step = 0.5 * T
        new_c = old_c + np.random.normal(0, step, 2)
        
        if check_collisions(current_centers, idx) is False: # This checks current, not new.
             # Need to check new_c against others
             pass
             
        # Correct collision check for new_c
        valid = True
        for j in range(N):
            if j == idx:
                continue
            d = np.linalg.norm(new_c - current_centers[j])
            if d < np.sqrt(3.0) - 1e-9:
                valid = False
                break
        
        if valid:
            # Tentatively place
            current_centers[idx] = new_c
            # Re-center
            mean_c = np.mean(current_centers, axis=0)
            current_centers = current_centers - mean_c
            
            new_S = calc_S(current_centers)
            
            if new_S < current_S:
                current_S = new_S
                if new_S < best_S:
                    best_S = new_S
                    best_centers = current_centers.copy()
            else:
                delta = new_S - current_S
                if np.random.rand() < np.exp(-delta / T):
                    current_S = new_S
                else:
                    # Revert
                    current_centers[idx] = old_c
                    mean_c = np.mean(current_centers, axis=0)
                    current_centers = current_centers - mean_c
                    # Recalc S? No, it's the same as before revert.
            
        T *= cooling
        if T < T_min:
            T = T_min
            
    inner = []
    for c in best_centers:
        inner.append([c[0], c[1], 0.0])
        
    return inner, (0.0, 0.0), best_S, 0.0

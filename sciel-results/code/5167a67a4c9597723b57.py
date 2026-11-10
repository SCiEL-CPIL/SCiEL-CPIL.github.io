import numpy as np
from scipy.optimize import minimize
import time

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 600  # 600s safety margin
    
    # Initial configuration
    # 1 center, 6 around, 4 more
    centers = []
    centers.append([0, 0])
    for k in range(6):
        angle = k * np.pi / 3
        centers.append([np.sqrt(3) * np.cos(angle), np.sqrt(3) * np.sin(angle)])
    for k in range(4):
        angle = k * np.pi / 2
        centers.append([2 * np.sqrt(3) * np.cos(angle), 2 * np.sqrt(3) * np.sin(angle)])
        
    x0 = np.array(centers).flatten()
    
    # Pre-compute hexagon vertices relative to center
    # Vertices for unit hexagon at origin
    base_vertices = np.array([
        [np.cos(k * np.pi / 3), np.sin(k * np.pi / 3)] for k in range(6)
    ])
    
    # Pre-compute edge normals for unit hexagon at origin
    # Edge k connects vertex k and k+1
    base_normals = []
    for k in range(6):
        p1 = base_vertices[k]
        p2 = base_vertices[(k+1)%6]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        normal /= np.linalg.norm(normal)
        base_normals.append(normal)
    base_normals = np.array(base_normals)
    
    def get_all_vertices(centers):
        # centers: (11, 2)
        # Returns (66, 2)
        return centers[:, np.newaxis, :] + base_vertices[np.newaxis, :, :]
        
    def check_overlap_fast(centers):
        # centers: (11, 2)
        # Returns True if any overlap
        for i in range(11):
            for j in range(i+1, 11):
                d = np.linalg.norm(centers[i] - centers[j])
                if d > 2.0:
                    continue
                if d < np.sqrt(3) - 1e-6:
                    return True
                
                # Detailed check
                v1 = centers[i] + base_vertices
                v2 = centers[j] + base_vertices
                
                # Check 6 normals of hex 1
                for n in base_normals:
                    proj1 = v1 @ n
                    proj2 = v2 @ n
                    if np.max(proj1) < np.min(proj2) - 1e-8 or np.max(proj2) < np.min(proj1) - 1e-8:
                        break
                else:
                    # Check 6 normals of hex 2 (same as hex 1 since same shape/orientation)
                    # Actually, if they are same orientation, normals are same.
                    # So if they passed hex 1 normals, they might still overlap?
                    # No, if they are separated on ANY normal of the Minkowski sum, they are separated.
                    # The normals of the Minkowski sum are the union of normals of both.
                    # Since they are identical orientation, the set of normals is the same.
                    # So checking 6 normals is sufficient for identical orientations.
                    return True
        return False

    def objective(x):
        centers = x.reshape(11, 2)
        
        if check_overlap_fast(centers):
            return 1000.0
        
        all_vertices = get_all_vertices(centers)
        
        # Outer hexagon normals (angle 0)
        # Midpoints at 30, 90, ...
        outer_normals = np.array([
            [np.cos(np.pi/6 + k * np.pi/3), np.sin(np.pi/6 + k * np.pi/3)] for k in range(6)
        ])
        
        projs = all_vertices @ outer_normals.T
        max_proj = np.max(projs)
        S = max_proj * 2 / np.sqrt(3)
        
        return S

    best_result = None
    best_score = float('inf')
    
    # Optimization loop
    while time.time() < deadline - 10:
        # Add small noise to avoid local minima
        x_start = best_result if best_result is not None else x0
        if best_result is not None:
            x_start = x_start + np.random.normal(0, 0.05, 22)
            
        result = minimize(objective, x_start, method='Powell', options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if result.fun < best_score:
            best_score = result.fun
            best_result = result.x
            
    if best_result is not None:
        centers = best_result.reshape(11, 2)
        inner_hex_data = np.zeros((11, 3))
        inner_hex_data[:, 0] = centers[:, 0]
        inner_hex_data[:, 1] = centers[:, 1]
        inner_hex_data[:, 2] = 0
        
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = best_score
    else:
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
        outer_hex_side_length = 8

    return inner_hex_data, outer_hex_data, outer_hex_side_length

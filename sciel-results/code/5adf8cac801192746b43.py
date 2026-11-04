import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
d = math.sqrt(3.0)

def construct():
    start_time = time.time()
    deadline = 140.0
    safety = 10.0
    
    # Generate ring 1 and ring 2 centers in axial coordinates
    # Axial to Cartesian: x = d*(q + r/2), y = d*(sqrt(3)/2)*r
    ring1_axial = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    ring2_axial = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                ring2_axial.append((q, r))
    
    ring1_centers = np.array([[d*(q + r/2.0), d*(math.sqrt(3)/2.0)*r] for q, r in ring1_axial])
    ring2_centers = np.array([[d*(q + r/2.0), d*(math.sqrt(3)/2.0)*r] for q, r in ring2_axial])
    
    # Precompute all 495 combinations of 4 from 12 ring-2 positions
    import itertools
    all_combos = list(itertools.combinations(range(12), 4))
    
    best_S = float('inf')
    best_config = None
    best_outer_angle = 0.0
    
    # Coarse grid search over outer angle
    theta_coarse = np.linspace(0, math.pi/3, 61)  # 61 points over 60 degrees
    
    for combo in all_combos:
        if time.time() > deadline - safety:
            break
        
        # Build centers: center + ring1 + selected ring2
        centers = np.vstack([[0, 0], ring1_centers, ring2_centers[list(combo)]])
        
        # Compute all 66 vertices (11 hexagons * 6 vertices)
        # Vertex angles: 0, 60, 120, 180, 240, 300 degrees
        vert_angles = np.arange(6) * math.pi / 3
        vert_offsets = np.stack([np.cos(vert_angles), np.sin(vert_angles)], axis=-1)
        
        # All points: (66, 2)
        points = np.tile(centers[:, np.newaxis, :], (1, 6, 1)) + vert_offsets[np.newaxis, :, :]
        points = points.reshape(-1, 2)
        
        # For each outer angle, compute S
        for theta in theta_coarse:
            if time.time() > deadline - safety:
                break
            
            # Outer hexagon normals
            normal_angles = theta + np.arange(6) * math.pi / 3
            normals = np.stack([np.cos(normal_angles), np.sin(normal_angles)], axis=-1)
            
            # Project all points onto normals
            dots = points @ normals.T  # (66, 6)
            max_proj = np.max(np.abs(dots))
            S = (2.0 / math.sqrt(3.0)) * max_proj
            
            if S < best_S:
                best_S = S
                best_config = centers.copy()
                best_outer_angle = theta
    
    # Refinement: local optimization around best outer angle
    if best_config is not None and time.time() < deadline - safety:
        # Try to refine by slightly adjusting the configuration
        # We can try small perturbations of the outer angle
        theta_fine = np.linspace(best_outer_angle - 0.05, best_outer_angle + 0.05, 101)
        
        for theta in theta_fine:
            if time.time() > deadline - safety:
                break
            
            normal_angles = theta + np.arange(6) * math.pi / 3
            normals = np.stack([np.cos(normal_angles), np.sin(normal_angles)], axis=-1)
            
            vert_angles = np.arange(6) * math.pi / 3
            vert_offsets = np.stack([np.cos(vert_angles), np.sin(vert_angles)], axis=-1)
            points = np.tile(best_config[:, np.newaxis, :], (1, 6, 1)) + vert_offsets[np.newaxis, :, :]
            points = points.reshape(-1, 2)
            
            dots = points @ normals.T
            max_proj = np.max(np.abs(dots))
            S = (2.0 / math.sqrt(3.0)) * max_proj
            
            if S < best_S:
                best_S = S
                best_outer_angle = theta
    
    # Verify disjointness and containment
    def _hex_vertices(cx, cy, angle, size=1.0):
        verts = []
        for k in range(6):
            a = angle + k * math.pi / 3
            verts.append((cx + size * math.cos(a), cy + size * math.sin(a)))
        return np.array(verts)
    
    def _poly_overlap(v1, v2, eps=1e-9):
        polys = [v1, v2]
        for v in polys:
            for i in range(len(v)):
                p1 = v[i]
                p2 = v[(i + 1) % len(v)]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                proj1 = np.dot(v1, normal)
                proj2 = np.dot(v2, normal)
                if proj1.max() < proj2.min() + eps or proj2.max() < proj1.min() + eps:
                    return False
        return True
    
    def _in_outer(point, outer_center, outer_side, outer_angle):
        p = point - outer_center
        k_angles = np.arange(6) * math.pi / 3 + outer_angle
        normals = np.stack([np.cos(k_angles), np.sin(k_angles)], axis=-1)
        dots = p @ normals.T
        apothem = outer_side * math.sqrt(3) / 2.0
        return np.all(np.abs(dots) <= apothem + 1e-9)
    
    if best_config is not None:
        # Check validity
        valid = True
        verts_list = [_hex_vertices(best_config[i][0], best_config[i][1], 0.0) for i in range(N)]
        
        for i in range(N):
            for j in range(i + 1, N):
                if _poly_overlap(verts_list[i], verts_list[j], 1e-6):
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            for i in range(N):
                for v in verts_list[i]:
                    if not _in_outer(v, (0.0, 0.0), best_S, best_outer_angle):
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            inner = [[best_config[i][0], best_config[i][1], 0.0] for i in range(N)]
            return inner, (0.0, 0.0), best_S, math.degrees(best_outer_angle)
    
    # Fallback: valid (loose) arrangement
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0

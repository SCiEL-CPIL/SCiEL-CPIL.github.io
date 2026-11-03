"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def hex_vertices(cx, cy, rot_deg):
        rot = np.radians(rot_deg)
        angles = rot + np.arange(6) * np.pi / 3
        return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])
    
    def min_container_radius(points, phi_deg):
        phi = np.radians(phi_deg)
        max_proj = 0.0
        for k in range(6):
            alpha = phi + np.pi/6 + k * np.pi/3
            nx, ny = np.cos(alpha), np.sin(alpha)
            for p in points:
                proj = nx * p[0] + ny * p[1]
                if proj > max_proj:
                    max_proj = proj
        return max_proj * 2.0 / np.sqrt(3.0)
    
    def evaluate_config(positions, rotations, num_phi=720):
        all_points = []
        for (cx, cy), rot in zip(positions, rotations):
            verts = hex_vertices(cx, cy, rot)
            for v in verts:
                all_points.append((v[0], v[1]))
        
        best_r = float('inf')
        best_phi = 0.0
        for i in range(num_phi):
            phi = i * 360.0 / num_phi
            r = min_container_radius(all_points, phi)
            if r < best_r:
                best_r = r
                best_phi = phi
        return best_r, best_phi
    
    # Generate candidate configurations
    candidates = []
    
    # 1. Linear row
    pos = [(i * np.sqrt(3), 0.0) for i in range(-5, 6)]
    rot = [0.0] * 11
    candidates.append((pos, rot))
    
    # 2. Flower + 4 outer
    pos = [(0.0, 0.0)]
    rot = [0.0]
    for k in range(6):
        ang = np.radians(k * 60)
        pos.append((np.sqrt(3) * np.cos(ang), np.sqrt(3) * np.sin(ang)))
        rot.append(0.0)
    for k, ang_deg in enumerate([0, 60, 120, 180]):
        ang = np.radians(ang_deg)
        pos.append((2 * np.sqrt(3) * np.cos(ang), 2 * np.sqrt(3) * np.sin(ang)))
        rot.append(0.0)
    candidates.append((pos, rot))
    
    # 3. Grid 3-4-3-1
    pos = []
    rot = []
    for i in range(3):
        pos.append((i * np.sqrt(3), 0.0))
        rot.append(0.0)
    for i in range(4):
        pos.append((i * np.sqrt(3) + np.sqrt(3)/2, 1.5))
        rot.append(0.0)
    for i in range(3):
        pos.append((i * np.sqrt(3), 3.0))
        rot.append(0.0)
    pos.append((np.sqrt(3), 4.5))
    rot.append(0.0)
    candidates.append((pos, rot))
    
    # 4. Grid 4-3-4
    pos = []
    rot = []
    for i in range(4):
        pos.append((i * np.sqrt(3), 0.0))
        rot.append(0.0)
    for i in range(3):
        pos.append((i * np.sqrt(3) + np.sqrt(3)/2, 1.5))
        rot.append(0.0)
    for i in range(4):
        pos.append((i * np.sqrt(3), 3.0))
        rot.append(0.0)
    candidates.append((pos, rot))
    
    # 5. Compact 3-3-3 + 2 caps
    pos = []
    rot = []
    for i in range(3):
        pos.append((i * np.sqrt(3), 0.0))
        rot.append(0.0)
    for i in range(3):
        pos.append((i * np.sqrt(3) + np.sqrt(3)/2, 1.5))
        rot.append(0.0)
    for i in range(3):
        pos.append((i * np.sqrt(3) + np.sqrt(3)/2, -1.5))
        rot.append(0.0)
    pos.append((np.sqrt(3), 3.0))
    rot.append(0.0)
    pos.append((np.sqrt(3), -3.0))
    rot.append(0.0)
    candidates.append((pos, rot))
    
    # 6. 5-2-4 arrangement
    pos = []
    rot = []
    for i in range(5):
        pos.append((i * np.sqrt(3), 0.0))
        rot.append(0.0)
    for i in range(2):
        pos.append((i * np.sqrt(3) + 2 * np.sqrt(3)/2, 1.5))
        rot.append(0.0)
    for i in range(4):
        pos.append((i * np.sqrt(3) + np.sqrt(3)/2, -1.5))
        rot.append(0.0)
    candidates.append((pos, rot))
    
    # Evaluate all
    best_r = float('inf')
    best_pos = None
    best_rot = None
    best_phi = 0.0
    
    for pos, rot in candidates:
        if time.time() > deadline - 5.0:
            break
        r, phi = evaluate_config(pos, rot, num_phi=720)
        if r < best_r:
            best_r = r
            best_pos = pos
            best_rot = rot
            best_phi = phi
    
    if best_pos is None:
        best_pos = [(i * np.sqrt(3), 0.0) for i in range(-5, 6)]
        best_rot = [0.0] * 11
        best_r, best_phi = evaluate_config(best_pos, best_rot, num_phi=720)
    
    # Center the cluster
    all_points = []
    for (cx, cy), rot in zip(best_pos, best_rot):
        verts = hex_vertices(cx, cy, rot)
        for v in verts:
            all_points.append((v[0], v[1]))
    all_points = np.array(all_points)
    centroid = np.mean(all_points, axis=0)
    
    inner = []
    for (cx, cy), rot in zip(best_pos, best_rot):
        inner.append([cx - centroid[0], cy - centroid[1], rot])
    
    # Recompute with centered points
    all_centered = all_points - centroid
    best_r_final = float('inf')
    best_phi_final = 0.0
    for i in range(720):
        phi = i * 360.0 / 720
        r = min_container_radius([(p[0], p[1]) for p in all_centered], phi)
        if r < best_r_final:
            best_r_final = r
            best_phi_final = phi
    
    return inner, (0.0, 0.0), best_r_final, best_phi_final
# EVOLVE-BLOCK-END

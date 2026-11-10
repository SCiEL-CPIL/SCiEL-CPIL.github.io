# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _get_all_vertices(positions, angles, side=1.0):
    """positions: (N,2), angles: (N,) in degrees. Returns (N*6, 2)."""
    verts = []
    for i in range(len(positions)):
        verts.append(_hex_vertices(positions[i,0], positions[i,1], side, angles[i]))
    return np.vstack(verts)

def _min_outer_side(vertices):
    """
    Compute minimum side length s of a regular hexagon centered at origin (angle 0)
    that contains all given vertices.
    """
    outward_normals = np.array([
        [np.cos(np.radians(30 + 60*k)), np.sin(np.radians(30 + 60*k))]
        for k in range(6)
    ])
    dots = vertices @ outward_normals.T
    max_dot = np.max(dots, axis=1)
    apothem_min = np.max(max_dot)
    s_min = 2.0 * apothem_min / np.sqrt(3.0)
    return s_min

def _check_separation(pos1, ang1, pos2, ang2, side=1.0):
    """
    Check if two unit hexagons are separated using SAT.
    Returns True if separated (no overlap), False if overlapping.
    """
    v1 = _hex_vertices(pos1[0], pos1[1], side, ang1)
    v2 = _hex_vertices(pos2[0], pos2[1], side, ang2)
    
    normals = []
    for v in [v1, v2]:
        for i in range(6):
            edge = v[(i+1)%6] - v[i]
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
                normals.append(n)
    
    for n in normals:
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 < min2 - 1e-8 or max2 < min1 - 1e-8:
            return True
    
    return False

def _overlap_penalty(pos, ang, side=1.0, eps=1e-6):
    """
    Compute a penalty for overlaps between all pairs of hexagons.
    """
    penalty = 0.0
    n = len(pos)
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pos[i] - pos[j])
            if d > 2.0 * side + 1e-3:
                continue
            v1 = _hex_vertices(pos[i,0], pos[i,1], side, ang[i])
            v2 = _hex_vertices(pos[j,0], pos[j,1], side, ang[j])
            
            normals = []
            for v in [v1, v2]:
                for k in range(6):
                    edge = v[(k+1)%6] - v[k]
                    nn = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(nn)
                    if norm > 1e-12:
                        nn = nn / norm
                        normals.append(nn)
            
            min_overlap = np.inf
            for nrm in normals:
                proj1 = v1 @ nrm
                proj2 = v2 @ nrm
                overlap = min(np.max(proj1), np.max(proj2)) - max(np.min(proj1), np.min(proj2))
                if overlap > 0:
                    min_overlap = min(min_overlap, overlap)
            
            if min_overlap < np.inf:
                penalty += min_overlap ** 2
    
    return penalty

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580.0  # 600s limit, leave margin
    
    n = 12
    side = 1.0
    
    # Initial configuration: hexagonal grid pattern
    init_positions = []
    init_angles = []
    for row in range(4):
        for col in range(3):
            x = col * 2 * side * 1.5
            y = row * side * np.sqrt(3)
            if row % 2 == 1:
                x += side * 1.5
            init_positions.append([x, y])
            init_angles.append(0.0)
    
    init_positions = np.array(init_positions)
    init_angles = np.array(init_angles)
    
    centroid = np.mean(init_positions, axis=0)
    init_positions -= centroid
    
    best_pos = init_positions.copy()
    best_ang = init_angles.copy()
    best_s = _min_outer_side(_get_all_vertices(best_pos, best_ang, side))
    
    # Simulated Annealing parameters
    T = 1.0
    T_min = 1e-6
    alpha = 0.999
    
    pos = best_pos.copy()
    ang = best_ang.copy()
    
    step = 0
    while time.time() < deadline:
        # Pick a random hexagon to move
        i = random.randint(0, n - 1)
        
        # Save old state
        old_pos = pos[i].copy()
        old_ang = ang[i]
        
        # Generate new state
        pos[i] = old_pos + np.random.normal(0, 0.1, 2)
        ang[i] = old_ang + np.random.normal(0, 5.0)
        
        # Evaluate new state
        vertices_new = _get_all_vertices(pos, ang, side)
        s_new = _min_outer_side(vertices_new)
        overlap_new = _overlap_penalty(pos, ang, side)
        objective_new = s_new + 10.0 * overlap_new
        
        # Evaluate old state (we need this for comparison)
        vertices_old = _get_all_vertices(np.concatenate([pos[:i], [old_pos], pos[i+1:]]), 
                                         np.concatenate([ang[:i], [old_ang], ang[i+1:]]), side)
        s_old = _min_outer_side(vertices_old)
        overlap_old = _overlap_penalty(np.concatenate([pos[:i], [old_pos], pos[i+1:]]), 
                                       np.concatenate([ang[:i], [old_ang], ang[i+1:]]), side)
        objective_old = s_old + 10.0 * overlap_old
        
        # Accept or reject
        delta = objective_new - objective_old
        if delta < 0 or random.random() < np.exp(-delta / T):
            # Accept
            pass
        else:
            # Reject, revert
            pos[i] = old_pos
            ang[i] = old_ang
        
        # Update best
        if objective_new < best_s + 10.0 * _overlap_penalty(best_pos, best_ang, side):
            # Verify separation
            valid = True
            for a in range(n):
                for b in range(a+1, n):
                    if not _check_separation(pos[a], ang[a], pos[b], ang[b], side):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                best_pos = pos.copy()
                best_ang = ang.copy()
                best_s = s_new
        
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
        
        step += 1
        if step % 1000 == 0:
            elapsed = time.time() - start_time
            if elapsed > deadline - 60.0:
                break
    
    # Final validation and return
    inner_hex_data = np.column_stack([best_pos, best_ang])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, best_s

# EVOLVE-BLOCK-END

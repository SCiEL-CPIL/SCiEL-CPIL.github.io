# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, s, a_deg):
    """Get 6 vertices of a regular hexagon."""
    a = np.radians(a_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _check_separation(v1, v2, eps=1e-6):
    """Check if two hexagons are separated (no intersection)."""
    for verts in [v1, v2]:
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i + 1) % 6]
            edge = p2 - p1
            n = np.array([-edge[1], edge[0]])
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-15:
                continue
            n = n / n_norm
            proj1 = verts @ n
            proj2 = v2 @ n
            if proj1.max() < proj2.min() - eps or proj2.max() < proj1.min() - eps:
                return True
    return False

def _compute_min_side(inner_positions):
    """Compute minimum outer hexagon side length needed to contain all inner hexagons."""
    inner_all_verts = []
    for pos, angle in inner_positions:
        verts = _hex_vertices(pos[0], pos[1], 1.0, angle)
        inner_all_verts.append(verts)
    
    all_points = np.vstack(inner_all_verts)
    
    best_side = float('inf')
    for outer_angle_deg in range(0, 60, 3):
        a = np.radians(outer_angle_deg)
        vertices = np.column_stack([np.cos(a + np.radians(60 * np.arange(6))), 
                                     np.sin(a + np.radians(60 * np.arange(6)))])
        max_proj = 0
        for i in range(6):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % 6]
            edge = p2 - p1
            n = np.array([-edge[1], edge[0]])
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-15:
                continue
            n = n / n_norm
            mid = (p1 + p2) / 2
            if np.dot(n, -mid) < 0:
                n = -n
            projs = all_points @ n
            max_proj = max(max_proj, np.max(projs))
        
        side = max_proj * 2 / np.sqrt(3)
        best_side = min(best_side, side)
    
    return best_side

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 600s limit
    
    sqrt3 = np.sqrt(3)
    
    # Honeycomb arrangement
    positions = np.array([
        [0.0, 0.0, 0.0],
        [sqrt3, 0.0, 0.0],
        [-sqrt3, 0.0, 0.0],
        [sqrt3/2, 1.5, 0.0],
        [-sqrt3/2, 1.5, 0.0],
        [sqrt3/2, -1.5, 0.0],
        [-sqrt3/2, -1.5, 0.0],
        [2*sqrt3, 0.0, 0.0],
        [-2*sqrt3, 0.0, 0.0],
        [sqrt3, 3.0, 0.0],
        [-sqrt3, -3.0, 0.0],
    ])
    
    # Local optimization
    best_side = _compute_min_side(positions)
    best_positions = positions.copy()
    step_size = 0.05
    
    while time.time() < deadline and step_size > 0.001:
        improved = False
        for i in range(11):
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    if dx == 0 and dy == 0:
                        continue
                    new_positions = positions.copy()
                    new_positions[i][0] += dx
                    new_positions[i][1] += dy
                    
                    valid = True
                    for j in range(11):
                        if j == i:
                            continue
                        v1 = _hex_vertices(new_positions[i][0], new_positions[i][1], 1.0, new_positions[i][2])
                        v2 = _hex_vertices(positions[j][0], positions[j][1], 1.0, positions[j][2])
                        if not _check_separation(v1, v2):
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    side = _compute_min_side(new_positions)
                    if side < best_side - 1e-8:
                        best_side = side
                        best_positions = new_positions
                        positions = new_positions
                        improved = True
        
        if not improved:
            step_size *= 0.8
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_side = max(best_side, np.sqrt(11))
    
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i, 0] = best_positions[i][0]
        inner_hex_data[i, 1] = best_positions[i][1]
        inner_hex_data[i, 2] = best_positions[i][2]
    
    return inner_hex_data, outer_hex_data, outer_side

# EVOLVE-BLOCK-END

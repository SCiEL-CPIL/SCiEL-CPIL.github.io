# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle in degrees."""
    a = math.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = a + math.radians(60 * i)
        verts[i, 0] = cx + d * math.cos(theta)
        verts[i, 1] = cy + d * math.sin(theta)
    return verts

def _check_separation(hex1, hex2):
    """Check if two hexagons (each (6,2) arrays) are separated by SAT."""
    for verts in (hex1, hex2):
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i + 1) % 6]
            edge = p2 - p1
            n = np.array([-edge[1], edge[0]])
            n = n / (np.linalg.norm(n) + 1e-15)
            proj1 = hex1 @ n
            proj2 = hex2 @ n
            if proj1.max() <= proj2.min() + 1e-6 or proj2.max() <= proj1.min() + 1e-6:
                return True
    return False

def _check_inside(hex_verts, outer_verts):
    """Check if all vertices of hex are inside outer hexagon."""
    for v in hex_verts:
        for i in range(6):
            p1 = outer_verts[i]
            p2 = outer_verts[(i + 1) % 6]
            cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
            if cross < -1e-6:
                return False
    return True

def _is_valid(inner_data, outer_data, s):
    """Full validity check."""
    outer_verts = _hex_vertices(outer_data[0], outer_data[1], s, outer_data[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False
    inner_verts_list = []
    for i in range(12):
        v = _hex_vertices(inner_data[i, 0], inner_data[i, 1], 1.0, inner_data[i, 2])
        inner_verts_list.append(v)
        if not _check_inside(v, outer_verts):
            return False
    for i in range(12):
        for j in range(i + 1, 12):
            if not _check_separation(inner_verts_list[i], inner_verts_list[j]):
                return False
    return True

def _get_all_inner_vertices(inner_data):
    """Get all 72 vertices from all 12 inner hexagons."""
    all_verts = []
    for i in range(12):
        v = _hex_vertices(inner_data[i, 0], inner_data[i, 1], 1.0, inner_data[i, 2])
        all_verts.append(v)
    return all_verts

def _min_s_for_all_vertices(all_inner_verts, outer_center, angle_deg):
    """Find minimum s such that all inner vertices fit in outer hexagon."""
    a = math.radians(angle_deg)
    normals = []
    for i in range(6):
        theta = a + math.radians(30 + 60 * i)
        normals.append(np.array([math.cos(theta), math.sin(theta)]))
    
    max_proj = 0
    for verts in all_inner_verts:
        for v in verts:
            rel = v - outer_center
            for n in normals:
                proj = abs(rel @ n)
                if proj > max_proj:
                    max_proj = proj
    return max_proj / (math.sqrt(3) / 2)

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill

    # Correct close-packed lattice for unit hexagons (side 1)
    # Neighbor distance is sqrt(3)
    # Using basis vectors for a hex grid
    e1 = np.array([1.5, math.sqrt(3)/2])
    e2 = np.array([0, math.sqrt(3)])
    
    # Generate 12 points in a 3x4 grid on this lattice
    positions = []
    for i in range(3):
        for j in range(4):
            pos = i * e1 + j * e2
            positions.append([pos[0], pos[1], 0.0])
            
    inner = np.array(positions)
    
    # Center the configuration
    com = np.mean(inner[:, :2], axis=0)
    inner[:, :2] -= com
    
    # Find optimal outer angle
    best_s = float('inf')
    best_inner = inner.copy()
    best_outer = np.array([0, 0, 0.0])
    
    all_verts = _get_all_inner_vertices(inner)
    
    for angle_deg in np.linspace(0, 60, 61):
        s_cand = _min_s_for_all_vertices(all_verts, np.array([0, 0]), angle_deg)
        if s_cand < best_s:
            best_s = s_cand
            best_inner = inner.copy()
            best_outer = np.array([0, 0, angle_deg])
            
    # Local optimization
    current_inner = best_inner.copy()
    current_outer = best_outer.copy()
    current_s = best_s
    
    np.random.seed(42)
    step_size = 0.1
    no_improve_count = 0
    max_no_improve = 20
    T = 0.5
    
    for iteration in range(10000):
        if time.time() > deadline:
            break
        
        idx = np.random.randint(0, 12)
        
        dx = np.random.uniform(-step_size, step_size)
        dy = np.random.uniform(-step_size, step_size)
        
        trial_inner = current_inner.copy()
        trial_inner[idx, 0] += dx
        trial_inner[idx, 1] += dy
        
        valid = True
        trial_verts = _hex_vertices(trial_inner[idx, 0], trial_inner[idx, 1], 1.0, trial_inner[idx, 2])
        for j in range(12):
            if j == idx:
                continue
            other_verts = _hex_vertices(trial_inner[j, 0], trial_inner[j, 1], 1.0, trial_inner[j, 2])
            if not _check_separation(trial_verts, other_verts):
                valid = False
                break
        
        if valid:
            trial_all_verts = _get_all_inner_vertices(trial_inner)
            best_trial_s = float('inf')
            best_trial_angle = current_outer[2]
            for angle_offset in np.linspace(-2, 2, 5):
                trial_angle = current_outer[2] + angle_offset
                s_cand = _min_s_for_all_vertices(trial_all_verts, np.array([0, 0]), trial_angle)
                if s_cand < best_trial_s:
                    best_trial_s = s_cand
                    best_trial_angle = trial_angle
            
            delta_s = current_s - best_trial_s
            if delta_s > 0:
                trial_outer = np.array([0, 0, best_trial_angle])
                if _is_valid(trial_inner, trial_outer, best_trial_s):
                    current_inner = trial_inner
                    current_outer = trial_outer
                    current_s = best_trial_s
                    no_improve_count = 0
                    T = max(T * 0.99, 0.01)
            else:
                if np.random.random() < math.exp(-delta_s / T):
                    trial_outer = np.array([0, 0, best_trial_angle])
                    if _is_valid(trial_inner, trial_outer, best_trial_s):
                        current_inner = trial_inner
                        current_outer = trial_outer
                        current_s = best_trial_s
                        T = max(T * 0.99, 0.01)
                else:
                    no_improve_count += 1
        else:
            no_improve_count += 1
            
        if no_improve_count >= max_no_improve:
            step_size *= 0.9
            no_improve_count = 0
            T *= 0.9
            if step_size < 0.001:
                break
                
    if not _is_valid(current_inner, current_outer, current_s):
        # Fallback to the initial lattice which should be valid
        current_inner = best_inner
        current_outer = best_outer
        current_s = best_s

    return current_inner, current_outer, current_s

# EVOLVE-BLOCK-END

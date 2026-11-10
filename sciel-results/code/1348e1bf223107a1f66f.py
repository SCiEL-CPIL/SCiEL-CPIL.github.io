# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

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

def _penalty_objective(x, penalty_weight=100.0):
    """
    Penalty-based objective: minimize s + penalty for violations.
    x = [cx1..cx12, cy1..cy12, angle1..angle12, s, outer_angle]
    Total: 12*2 + 12 + 1 + 1 = 38 variables
    """
    cx = x[0:12]
    cy = x[12:24]
    angles = x[24:36]
    s = x[36]
    outer_angle = x[37]
    
    # Compute inner hexagon vertices
    inner_verts_list = []
    for i in range(12):
        v = _hex_vertices(cx[i], cy[i], 1.0, angles[i])
        inner_verts_list.append(v)
    
    # Compute outer hexagon vertices
    outer_verts = _hex_vertices(0, 0, s, outer_angle)
    
    # Penalty for inner vertices outside outer hexagon
    containment_penalty = 0.0
    for i in range(12):
        for v in inner_verts_list[i]:
            for j in range(6):
                p1 = outer_verts[j]
                p2 = outer_verts[(j + 1) % 6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < 0:
                    containment_penalty += cross * cross
    
    # Penalty for inner hexagon overlaps (using SAT)
    overlap_penalty = 0.0
    for i in range(12):
        for j in range(i + 1, 12):
            hex1 = inner_verts_list[i]
            hex2 = inner_verts_list[j]
            for verts in (hex1, hex2):
                for k in range(6):
                    p1 = verts[k]
                    p2 = verts[(k + 1) % 6]
                    edge = p2 - p1
                    n = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(n)
                    if norm < 1e-15:
                        continue
                    n = n / norm
                    proj1 = hex1 @ n
                    proj2 = hex2 @ n
                    # If projections overlap, that's a potential violation
                    # We want them separated, so if they overlap, add penalty
                    gap = min(proj2.min() - proj1.max(), proj1.min() - proj2.max())
                    if gap < 0:
                        overlap_penalty += gap * gap
    
    total_penalty = containment_penalty + overlap_penalty
    return s + penalty_weight * total_penalty

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
    
    # Find optimal outer angle for initial lattice
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
            
    # Build initial x for penalty optimization
    x0 = np.zeros(38)
    x0[0:12] = best_inner[:, 0]
    x0[12:24] = best_inner[:, 1]
    x0[24:36] = best_inner[:, 2]
    x0[36] = best_s
    x0[37] = best_outer[2]
    
    # Penalty optimization with L-BFGS-B
    best_result = None
    best_obj = float('inf')
    best_x = x0.copy()
    
    # Try multiple penalty weights and restarts
    penalty_weights = [50.0, 100.0, 200.0, 500.0]
    
    for pw in penalty_weights:
        if time.time() > deadline:
            break
        
        try:
            res = minimize(
                _penalty_objective,
                x0,
                args=(pw,),
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if res.success or res.fun < best_obj:
                x_opt = res.x
                cx = x_opt[0:12]
                cy = x_opt[12:24]
                angles = x_opt[24:36]
                s_opt = x_opt[36]
                outer_angle = x_opt[37]
                
                # Build inner data
                inner_data = np.column_stack([cx, cy, angles])
                outer_data = np.array([0, 0, outer_angle])
                
                # Verify validity
                if _is_valid(inner_data, outer_data, s_opt):
                    if s_opt < best_obj:
                        best_obj = s_opt
                        best_x = x_opt.copy()
                        best_result = (inner_data, outer_data, s_opt)
                else:
                    # Try to fix by adjusting s slightly upward
                    for s_adj in [s_opt * 1.001, s_opt * 1.005, s_opt * 1.01, s_opt * 1.02]:
                        if _is_valid(inner_data, outer_data, s_adj):
                            if s_adj < best_obj:
                                best_obj = s_adj
                                best_x = x_opt.copy()
                                best_result = (inner_data, outer_data, s_adj)
                            break
        except Exception:
            continue
    
    # If penalty optimization didn't find anything better, use the lattice
    if best_result is None or best_obj > best_s:
        best_result = (best_inner, best_outer, best_s)
    
    # Additional local refinement with smaller steps
    if time.time() < deadline - 30:
        current_inner, current_outer, current_s = best_result
        np.random.seed(42)
        step_size = 0.05
        T = 0.1
        
        for iteration in range(3000):
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
                for angle_offset in np.linspace(-1, 1, 5):
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
                        T = max(T * 0.99, 0.005)
                else:
                    if np.random.random() < math.exp(-delta_s / T):
                        trial_outer = np.array([0, 0, best_trial_angle])
                        if _is_valid(trial_inner, trial_outer, best_trial_s):
                            current_inner = trial_inner
                            current_outer = trial_outer
                            current_s = best_trial_s
                            T = max(T * 0.99, 0.005)
            
            if step_size < 0.001:
                break
            step_size *= 0.999
        
        if current_s < best_obj:
            best_result = (current_inner, current_outer, current_s)
    
    inner_final, outer_final, s_final = best_result
    
    if not _is_valid(inner_final, outer_final, s_final):
        # Fallback to the initial lattice which should be valid
        inner_final = best_inner
        outer_final = best_outer
        s_final = best_s
    
    return inner_final, outer_final, s_final

# EVOLVE-BLOCK-END

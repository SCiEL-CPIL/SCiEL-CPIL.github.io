# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize

def hexagon_vertices(cx, cy, s, angle_deg):
    """Return vertices of a regular hexagon with center (cx,cy), side s, angle angle_deg."""
    angles = np.radians(angle_deg + 60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def get_edge_normals(verts):
    """Get outward-pointing edge normals for a hexagon given its vertices."""
    normals = []
    for i in range(6):
        p1 = verts[i]
        p2 = verts[(i + 1) % 6]
        edge = p2 - p1
        n = np.array([-edge[1], edge[0]])
        center = np.mean(verts, axis=0)
        mid = (p1 + p2) / 2
        if np.dot(n, mid - center) < 0:
            n = -n
        n = n / np.linalg.norm(n)
        normals.append(n)
    return np.array(normals)

def calculate_min_side(inner_hex_data, outer_angle_deg):
    """
    Calculate the minimum side length S of an outer regular hexagon centered at origin
    with given angle that contains all inner hexagons.
    """
    normal_angles = [outer_angle_deg + 30 + 60 * k for k in range(6)]
    normals = np.array([[np.cos(np.radians(a)), np.sin(np.radians(a))] for a in normal_angles])
    
    # Precompute base vertices for unit hexagon at angle 0
    base_angles = np.radians(60 * np.arange(6))
    base_verts = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    
    max_proj = 0.0
    for i in range(11):
        x, y, _ = inner_hex_data[i]
        verts = base_verts + np.array([x, y])
        projs = verts @ normals.T
        max_p = np.max(projs)
        if max_p > max_proj:
            max_proj = max_p
            
    side = 2.0 * max_proj / math.sqrt(3)
    return side

def min_gap_between_hexes(hex1, hex2):
    """
    Calculate the minimum separation distance between two unit hexagons.
    Positive if separated, 0 if touching, negative if overlapping.
    """
    x1, y1, a1 = hex1
    x2, y2, a2 = hex2
    
    v1 = hexagon_vertices(x1, y1, 1.0, a1)
    v2 = hexagon_vertices(x2, y2, 1.0, a2)
    
    n1 = get_edge_normals(v1)
    n2 = get_edge_normals(v2)
    
    all_normals = np.vstack([n1, n2])
    
    min_gap = float('inf')
    for n in all_normals:
        proj1 = v1 @ n
        proj2 = v2 @ n
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Gap on this axis
        # If intervals [min1, max1] and [min2, max2] are separated:
        # gap = max(min1, min2) - min(max1, max2) ? No.
        # Let's use the derived formula: gap = max(min1, min2) - min(max1, max2) is wrong for separation.
        # Separation distance is:
        if max1 < min2:
            gap = min2 - max1
        elif max2 < min1:
            gap = min1 - max2
        else:
            # Overlapping or touching
            # The "depth" of overlap is negative gap.
            # For touching, gap is 0.
            # We can define gap as the minimum distance between any point in proj1 and proj2.
            # If they overlap, the distance is 0? No, we want a penalty for overlap.
            # Let's define gap = min(max1, max2) - max(min1, min2).
            # If separated: e.g., [0,1] and [2,3]. min(1,3)=1, max(0,2)=2. 1-2 = -1.
            # So gap = -1 means overlap of 1? No, separation is 1.
            # Let's stick to:
            # sep = max(min1, min2) - min(max1, max2)
            # If sep > 0, they are separated by sep.
            # If sep <= 0, they overlap or touch.
            gap = max(min1, min2) - min(max1, max2)
            
        if gap < min_gap:
            min_gap = gap
            
    return min_gap

def objective(pos_flat, outer_angle_deg):
    """
    Objective function for optimization.
    pos_flat: 22 coordinates (11 hexagons * 2).
    Returns: outer_side_length + penalty.
    """
    pos = pos_flat.reshape(11, 2)
    inner_hex_data = np.zeros((11, 3))
    inner_hex_data[:, 0] = pos[:, 0]
    inner_hex_data[:, 1] = pos[:, 1]
    inner_hex_data[:, 2] = 0.0 # Keep angle 0
    
    side = calculate_min_side(inner_hex_data, outer_angle_deg)
    
    penalty = 0.0
    for i in range(11):
        for j in range(i + 1, 11):
            gap = min_gap_between_hexes(inner_hex_data[i], inner_hex_data[j])
            if gap < 0:
                # Overlap
                penalty += (-gap) ** 2
    
    return side + 10.0 * penalty

def hexagon_packing_11():
    start_time = time.time()
    deadline = 630.0
    safety_margin = 10.0
    
    h = math.sqrt(3) / 2
    coords = []
    for i in range(4):
        coords.append([i * 3.0, 0.0])
    for i in range(3):
        coords.append([1.5 + i * 3.0, 2 * h])
    for i in range(4):
        coords.append([i * 3.0, 4 * h])
        
    cx = np.mean([c[0] for c in coords])
    cy = np.mean([c[1] for c in coords])
    for c in coords:
        c[0] -= cx
        c[1] -= cy
        
    # Initial best
    inner_hex_data = np.zeros((11, 3))
    for i, c in enumerate(coords):
        inner_hex_data[i, 0] = c[0]
        inner_hex_data[i, 1] = c[1]
        inner_hex_data[i, 2] = 0.0
        
    best_side = float('inf')
    best_outer_angle = 0.0
    best_pos = inner_hex_data.copy()
    
    # 1. Find best outer angle for the initial grid
    for outer_angle_deg in np.arange(0, 30.01, 0.5):
        if time.time() - start_time > deadline - safety_margin:
            break
        side = calculate_min_side(inner_hex_data, outer_angle_deg)
        if side < best_side:
            best_side = side
            best_outer_angle = outer_angle_deg
            
    # Refine angle
    if time.time() - start_time < deadline - safety_margin:
        for outer_angle_deg in np.arange(max(0, best_outer_angle - 1.0), min(30, best_outer_angle + 1.0) + 0.01, 0.1):
            if time.time() - start_time > deadline - safety_margin:
                break
            side = calculate_min_side(inner_hex_data, outer_angle_deg)
            if side < best_side:
                best_side = side
                best_outer_angle = outer_angle_deg
                
    # 2. Optimize positions using Scipy
    if time.time() - start_time < deadline - safety_margin:
        initial_pos_flat = inner_hex_data[:, :2].flatten()
        
        # Use Nelder-Mead for robustness
        # maxiter is limited by time, but we check time inside objective? No, objective is pure.
        # We will run minimize with a time check in the callback or just limit iterations.
        # Nelder-Mead can be slow. Let's try a few iterations.
        
        options = {
            'maxiter': 2000,
            'xatol': 1e-8,
            'fatol': 1e-8
        }
        
        try:
            res = minimize(objective, initial_pos_flat, args=(best_outer_angle,), method='Nelder-Mead', options=options)
            if res.success or res.fun < best_side:
                new_pos = res.x.reshape(11, 2)
                new_inner = np.zeros((11, 3))
                new_inner[:, 0] = new_pos[:, 0]
                new_inner[:, 1] = new_pos[:, 1]
                new_inner[:, 2] = 0.0
                
                # Verify validity and calculate exact side
                valid = True
                for i in range(11):
                    for j in range(i + 1, 11):
                        gap = min_gap_between_hexes(new_inner[i], new_inner[j])
                        if gap <= 1e-6: # Must be separated by MORE than 1e-6
                             # Wait, the condition is "apart by more than 1e-6".
                             # So gap > 1e-6 is required.
                             # If gap <= 1e-6, it's invalid.
                             valid = False
                             break
                    if not valid: break
                             
                if valid:
                    new_side = calculate_min_side(new_inner, best_outer_angle)
                    if new_side < best_side:
                        best_side = new_side
                        best_pos = new_inner
        except Exception as e:
            pass
            
    # 3. Final local search with small steps to ensure we are at a local minimum
    # and to fix any tiny validity issues
    if time.time() - start_time < deadline - safety_margin:
        current_best = best_side
        current_pos = best_pos.copy()
        
        step_sizes = [0.01, 0.001, 0.0001]
        for step in step_sizes:
            if time.time() - start_time > deadline - safety_margin:
                break
            improved = True
            while improved and time.time() - start_time < deadline - safety_margin:
                improved = False
                for i in range(11):
                    if time.time() - start_time > deadline - safety_margin:
                        break
                    for dx in [-step, step]:
                        for dy in [-step, step]:
                            test_pos = current_pos.copy()
                            test_pos[i, 0] += dx
                            test_pos[i, 1] += dy
                            
                            valid = True
                            for j in range(11):
                                if i == j:
                                    continue
                                gap = min_gap_between_hexes(test_pos[i], test_pos[j])
                                if gap <= 1e-6:
                                    valid = False
                                    break
                            
                            if not valid:
                                continue
                            
                            new_side = calculate_min_side(test_pos, best_outer_angle)
                            if new_side < current_best - 1e-10:
                                current_pos = test_pos
                                current_best = new_side
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        if current_best < best_side:
            best_side = current_best
            best_pos = current_pos

    outer_hex_data = np.array([0.0, 0.0, best_outer_angle])
    
    return best_pos, outer_hex_data, best_side

# EVOLVE-BLOCK-END

import numpy as np
import time

def _hex_vertices(cx, cy, s, a_deg):
    """Compute 6 vertices of a regular hexagon."""
    a = np.radians(a_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _hex_separated(v1, v2, tol=1e-6):
    """Check if two hexagons are separated using all 12 edge normals."""
    for i in range(6):
        e = v1[(i+1)%6] - v1[i]
        n = np.array([-e[1], e[0]])
        norm = np.linalg.norm(n)
        if norm > 1e-10:
            n = n / norm
            p1 = v1 @ n
            p2 = v2 @ n
            if np.max(p1) < np.min(p2) - tol or np.max(p2) < np.min(p1) - tol:
                return True
    for i in range(6):
        e = v2[(i+1)%6] - v2[i]
        n = np.array([-e[1], e[0]])
        norm = np.linalg.norm(n)
        if norm > 1e-10:
            n = n / norm
            p1 = v1 @ n
            p2 = v2 @ n
            if np.max(p1) < np.min(p2) - tol or np.max(p2) < np.min(p1) - tol:
                return True
    return False

def _is_valid_packing(positions, angles):
    """Check if all pairs of hexagons are separated."""
    verts_list = []
    for i in range(11):
        verts_list.append(_hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i]))
    for i in range(11):
        for j in range(i+1, 11):
            if not _hex_separated(verts_list[i], verts_list[j]):
                return False
    return True

def _compute_outer_side(positions, angles, outer_angle):
    """
    Compute the minimum side length of a regular hexagon (centered at origin)
    with the given angle that contains all inner hexagon vertices.
    """
    all_verts = []
    for i in range(11):
        v = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    a = np.radians(outer_angle)
    max_abs_proj = 0.0
    for i in range(6):
        theta = a + np.radians(30 + 60 * i)
        n = np.array([np.cos(theta), np.sin(theta)])
        proj = all_verts @ n
        max_abs_proj = max(max_abs_proj, np.max(np.abs(proj)))
    
    return 2.0 * max_abs_proj / np.sqrt(3)

def _build_hex7_cluster():
    """Build a hex-7 cluster (center + 6 neighbors)."""
    sqrt3 = np.sqrt(3)
    pos = np.zeros((7, 2))
    pos[0] = [0, 0]
    pos[1] = [2, 0]
    pos[2] = [1, sqrt3]
    pos[3] = [-1, sqrt3]
    pos[4] = [-2, 0]
    pos[5] = [-1, -sqrt3]
    pos[6] = [1, -sqrt3]
    return pos

def _add_satellite(pos_cluster, index, distance, angle_deg, offset_dir=0):
    """Add a satellite at a given distance and angle from center."""
    a = np.radians(angle_deg)
    x = distance * np.cos(a)
    y = distance * np.sin(a)
    return np.array([x, y])

def _build_config_v1():
    """Config 1: Hex-7 + 4 satellites at corners (tighter)."""
    sqrt3 = np.sqrt(3)
    pos = np.zeros((11, 2))
    pos[:7] = _build_hex7_cluster()
    # 4 satellites at ~3.5 distance, 45 degrees apart
    for i in range(4):
        angle = 45 + 90 * i
        pos[7+i] = _add_satellite(None, 0, 3.5, angle)
    return pos, np.zeros(11)

def _build_config_v2():
    """Config 2: Hex-7 + 4 satellites in a diamond pattern."""
    sqrt3 = np.sqrt(3)
    pos = np.zeros((11, 2))
    pos[:7] = _build_hex7_cluster()
    # Diamond: top, bottom, left, right at distance 3.8
    pos[7] = [0, 3.8]
    pos[8] = [0, -3.8]
    pos[9] = [3.8, 0]
    pos[10] = [-3.8, 0]
    return pos, np.zeros(11)

def _build_config_v3():
    """Config 3: Staggered rows."""
    sqrt3 = np.sqrt(3)
    pos = np.zeros((11, 2))
    # Row 0: 4 hexagons
    for i in range(4):
        pos[i] = [i*2 - 3, 0]
    # Row 1: 3 hexagons (offset)
    for i in range(3):
        pos[4+i] = [i*2 - 2, sqrt3]
    # Row 2: 4 hexagons
    for i in range(4):
        pos[7+i] = [i*2 - 3, 2*sqrt3]
    return pos, np.zeros(11)

def _build_config_v4():
    """Config 4: Compact asymmetric."""
    sqrt3 = np.sqrt(3)
    pos = np.zeros((11, 2))
    pos[0] = [0, 0]
    pos[1] = [2, 0]
    pos[2] = [1, sqrt3]
    pos[3] = [-1, sqrt3]
    pos[4] = [-2, 0]
    pos[5] = [-1, -sqrt3]
    pos[6] = [1, -sqrt3]
    # 4 more in a tight cluster around
    pos[7] = [3, sqrt3/2]
    pos[8] = [-3, sqrt3/2]
    pos[9] = [3, -sqrt3/2]
    pos[10] = [-3, -sqrt3/2]
    return pos, np.zeros(11)

def _build_config_v5():
    """Config 5: Ring of 10 around center."""
    pos = np.zeros((11, 2))
    pos[0] = [0, 0]
    for i in range(10):
        angle = 36 * i
        pos[i+1] = [2.8 * np.cos(np.radians(angle)), 2.8 * np.sin(np.radians(angle))]
    return pos, np.zeros(11)

def _build_config_v6():
    """Config 6: Two rows of 5 + 1 center."""
    sqrt3 = np.sqrt(3)
    pos = np.zeros((11, 2))
    # Center
    pos[0] = [0, 0]
    # Top row: 5 hexagons
    for i in range(5):
        pos[1+i] = [i*2 - 4, sqrt3 * 1.2]
    # Bottom row: 5 hexagons
    for i in range(5):
        pos[6+i] = [i*2 - 4, -sqrt3 * 1.2]
    return pos, np.zeros(11)

def _build_config_v7():
    """Config 7: Hex-7 + 4 at hexagon vertices (tightest possible)."""
    sqrt3 = np.sqrt(3)
    pos = np.zeros((11, 2))
    pos[:7] = _build_hex7_cluster()
    # Place at the 4 "corner" positions of the hex-7 cluster
    # These are at distance sqrt(12) ≈ 3.464 from center
    d = np.sqrt(12)
    pos[7] = [d * np.cos(np.radians(30)), d * np.sin(np.radians(30))]
    pos[8] = [d * np.cos(np.radians(150)), d * np.sin(np.radians(150))]
    pos[9] = [d * np.cos(np.radians(210)), d * np.sin(np.radians(210))]
    pos[10] = [d * np.cos(np.radians(330)), d * np.sin(np.radians(330))]
    return pos, np.zeros(11)

def _build_config_v8():
    """Config 8: Optimized hex-7 + 4 satellites with slight perturbation."""
    sqrt3 = np.sqrt(3)
    pos = np.zeros((11, 2))
    pos[:7] = _build_hex7_cluster()
    # Satellites at distance 3.4, angles 45, 135, 225, 315
    for i in range(4):
        angle = 45 + 90 * i
        pos[7+i] = [3.4 * np.cos(np.radians(angle)), 3.4 * np.sin(np.radians(angle))]
    return pos, np.zeros(11)

def _build_config_v9():
    """Config 9: Compact 3-3-3-2 arrangement."""
    sqrt3 = np.sqrt(3)
    pos = np.zeros((11, 2))
    # Row 0: 3 hexagons
    for i in range(3):
        pos[i] = [i*2 - 2, 0]
    # Row 1: 3 hexagons (offset)
    for i in range(3):
        pos[3+i] = [i*2 - 1, sqrt3]
    # Row 2: 3 hexagons
    for i in range(3):
        pos[6+i] = [i*2 - 2, 2*sqrt3]
    # Row 3: 2 hexagons (offset)
    for i in range(2):
        pos[9+i] = [i*2 - 1, 3*sqrt3]
    return pos, np.zeros(11)

def _build_config_v10():
    """Config 10: Very tight hex-7 + 4 at minimum distance."""
    sqrt3 = np.sqrt(3)
    pos = np.zeros((11, 2))
    pos[:7] = _build_hex7_cluster()
    # Try distance 3.2 for satellites
    for i in range(4):
        angle = 45 + 90 * i
        pos[7+i] = [3.2 * np.cos(np.radians(angle)), 3.2 * np.sin(np.radians(angle))]
    return pos, np.zeros(11)

def _build_all_configs():
    """Build all initial configurations."""
    configs = []
    for builder in [_build_config_v1, _build_config_v2, _build_config_v3, 
                     _build_config_v4, _build_config_v5, _build_config_v6,
                     _build_config_v7, _build_config_v8, _build_config_v9,
                     _build_config_v10]:
        try:
            pos, angles = builder()
            configs.append((pos, angles))
        except Exception:
            continue
    return configs

def _optimize_config(pos_init, angles_init, deadline):
    """
    Optimize a single configuration using multi-stage local search.
    """
    best_side = float('inf')
    best_pos = pos_init.copy()
    best_angles = angles_init.copy()
    best_outer_angle = 0.0
    
    # First, check if initial config is valid
    if not _is_valid_packing(pos_init, angles_init):
        # Try to fix by spreading out
        scale = 1.1
        while scale < 2.0 and not _is_valid_packing(pos_init * scale, angles_init):
            scale += 0.05
        if not _is_valid_packing(pos_init * scale, angles_init):
            return float('inf'), None, None, 0.0
        best_pos = pos_init * scale
    
    # Phase 1: Global rotation search
    n_rot = 12
    rot_range = np.linspace(0, 60, n_rot, endpoint=False)
    
    for rot in rot_range:
        if time.time() > deadline - 60:
            break
        angles = np.full(11, rot)
        if not _is_valid_packing(best_pos, angles):
            continue
        # Try different outer angles
        for outer_a in np.linspace(0, 60, 12, endpoint=False):
            side = _compute_outer_side(best_pos, angles, outer_a)
            if side < best_side:
                best_side = side
                best_angles = angles.copy()
                best_outer_angle = outer_a
    
    if best_side == float('inf'):
        if _is_valid_packing(best_pos, angles_init):
            best_side = _compute_outer_side(best_pos, angles_init, 0)
            best_outer_angle = 0.0
            best_angles = angles_init.copy()
    
    if best_side == float('inf'):
        return float('inf'), None, None, 0.0
    
    # Phase 2: Local refinement with adaptive steps
    step = 0.1
    for iteration in range(40):
        if time.time() > deadline - 30:
            break
        improved = False
        for i in range(11):
            if time.time() > deadline - 30:
                break
            for dx, dy in [(step,0),(-step,0),(0,step),(0,-step),
                          (step*0.7,step*0.7),(-step*0.7,step*0.7),
                          (step*0.7,-step*0.7),(-step*0.7,-step*0.7)]:
                new_pos = best_pos.copy()
                new_pos[i] += [dx, dy]
                if _is_valid_packing(new_pos, best_angles):
                    side = _compute_outer_side(new_pos, best_angles, best_outer_angle)
                    if side < best_side - 1e-9:
                        best_pos = new_pos
                        best_side = side
                        improved = True
                        break
        if not improved:
            step *= 0.7
            if step < 1e-4:
                break
    
    # Phase 3: Refine outer angle
    if time.time() < deadline - 15:
        for outer_a in np.linspace(best_outer_angle - 5, best_outer_angle + 5, 40):
            if outer_a < 0 or outer_a >= 60:
                continue
            side = _compute_outer_side(best_pos, best_angles, outer_a)
            if side < best_side:
                best_side = side
                best_outer_angle = outer_a
    
    # Phase 4: Second round of position refinement with smaller steps
    if time.time() < deadline - 10:
        step = 0.05
        for iteration in range(20):
            if time.time() > deadline - 10:
                break
            improved = False
            for i in range(11):
                if time.time() > deadline - 10:
                    break
                for dx, dy in [(step,0),(-step,0),(0,step),(0,-step),
                              (step*0.7,step*0.7),(-step*0.7,step*0.7),
                              (step*0.7,-step*0.7),(-step*0.7,-step*0.7)]:
                    new_pos = best_pos.copy()
                    new_pos[i] += [dx, dy]
                    if _is_valid_packing(new_pos, best_angles):
                        side = _compute_outer_side(new_pos, best_angles, best_outer_angle)
                        if side < best_side - 1e-9:
                            best_pos = new_pos
                            best_side = side
                            improved = True
                            break
            if not improved:
                step *= 0.7
                if step < 1e-5:
                    break
    
    return best_side, best_pos, best_angles, best_outer_angle

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 580
    
    best_side = float('inf')
    best_pos = None
    best_angles = None
    best_outer_angle = 0.0
    
    configs = _build_all_configs()
    
    for pos_init, angles_init in configs:
        if time.time() > deadline - 30:
            break
        try:
            side, pos, angles, outer_a = _optimize_config(pos_init, angles_init, deadline)
            if side < best_side:
                best_side = side
                best_pos = pos
                best_angles = angles
                best_outer_angle = outer_a
        except Exception:
            continue
    
    # Fallback if no valid config found
    if best_pos is None or best_side >= float('inf'):
        sqrt3 = np.sqrt(3)
        pos_valid = np.zeros((11, 2))
        pos_valid[0] = [0, 0]
        pos_valid[1] = [2, 0]
        pos_valid[2] = [1, sqrt3]
        pos_valid[3] = [-1, sqrt3]
        pos_valid[4] = [-2, 0]
        pos_valid[5] = [-1, -sqrt3]
        pos_valid[6] = [1, -sqrt3]
        pos_valid[7] = [4, 0]
        pos_valid[8] = [-4, 0]
        pos_valid[9] = [2, 3*sqrt3]
        pos_valid[10] = [-2, 3*sqrt3]
        best_pos = pos_valid
        best_angles = np.zeros(11)
        best_outer_angle = 0.0
        best_side = _compute_outer_side(best_pos, best_angles, best_outer_angle)
    
    # Final validation
    if not _is_valid_packing(best_pos, best_angles):
        sqrt3 = np.sqrt(3)
        pos_valid = np.zeros((11, 2))
        pos_valid[0] = [0, 0]
        pos_valid[1] = [2, 0]
        pos_valid[2] = [1, sqrt3]
        pos_valid[3] = [-1, sqrt3]
        pos_valid[4] = [-2, 0]
        pos_valid[5] = [-1, -sqrt3]
        pos_valid[6] = [1, -sqrt3]
        pos_valid[7] = [5, 0]
        pos_valid[8] = [-5, 0]
        pos_valid[9] = [2.5, 4*sqrt3/2]
        pos_valid[10] = [-2.5, 4*sqrt3/2]
        best_pos = pos_valid
        best_angles = np.zeros(11)
        best_outer_angle = 0.0
        best_side = _compute_outer_side(best_pos, best_angles, best_outer_angle)
    
    inner_hex_data = np.column_stack([best_pos, best_angles])
    outer_hex_data = np.array([0.0, 0.0, best_outer_angle])
    
    return inner_hex_data, outer_hex_data, float(best_side)

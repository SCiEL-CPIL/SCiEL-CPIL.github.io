import math
import time
import random
import numpy as np
from scipy.optimize import minimize_scalar

N = 12
DEADLINE = 115.0  # Safety margin

# Precompute unit hexagon vertices and edge normals (angle 0)
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i+1)%6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i+1)%6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx*_nx + _ny*_ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

# Fixed outer hexagon normals (angle 0)
_OUTER_NORMALS = np.array([
    [math.cos(math.pi/6 + i*math.pi/3), math.sin(math.pi/6 + i*math.pi/3)] 
    for i in range(6)
])

def construct():
    t0 = time.time()
    
    # Precompute arrays
    _ux = _UNIT_HEX_VERTS[:, 0]
    _uy = _UNIT_HEX_VERTS[:, 1]
    _enx = _EDGE_NORMALS[:, 0]
    _eny = _EDGE_NORMALS[:, 1]
    _onx = _OUTER_NORMALS[:, 0]
    _ony = _OUTER_NORMALS[:, 1]
    sqrt3 = math.sqrt(3)

    def get_vertices(positions, angles_rad):
        """
        Returns (N, 6, 2) array of vertices.
        """
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        # (N, 6)
        rx = _ux[None, :] * c[:, None] - _uy[None, :] * s[:, None]
        ry = _ux[None, :] * s[:, None] + _uy[None, :] * c[:, None]
        verts_x = rx + positions[:, 0:1]
        verts_y = ry + positions[:, 1:2]
        return np.stack((verts_x, verts_y), axis=-1)

    def compute_side_length(positions, angles_rad):
        """
        Computes the minimum side length of a regular hexagon (fixed orientation 0)
        enclosing all inner hexagons.
        """
        verts = get_vertices(positions, angles_rad) # (N, 6, 2)
        # Flatten to (N*6, 2)
        v_flat = verts.reshape(-1, 2)
        
        # Center of mass
        center = np.mean(v_flat, axis=0)
        offsets = v_flat - center
        
        # Project onto 6 normals
        # offsets: (72, 2), _onx: (6,)
        proj_x = offsets[:, 0] * _onx[None, :]
        proj_y = offsets[:, 1] * _ony[None, :]
        proj = np.abs(proj_x + proj_y) # (72, 6)
        
        max_projs = np.max(proj, axis=0) # (6,)
        # Side length S = 2/sqrt(3) * max_proj
        side = (2.0 / sqrt3) * np.max(max_projs)
        
        return side, center

    def hexagons_disjoint_fast(h1, h2):
        """
        h1, h2: (x, y, angle_rad)
        Returns True if disjoint.
        """
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        if dist_sq > 4.5: # (2*apothem)^2 = 3, so 4.5 is safe upper bound for overlap check
            return True
        
        a1 = h1[2]
        a2 = h2[2]
        c1, s1 = math.cos(a1), math.sin(a1)
        c2, s2 = math.cos(a2), math.sin(a2)
        
        v1x = h1[0] + _ux * c1 - _uy * s1
        v1y = h1[1] + _ux * s1 + _uy * c1
        v2x = h2[0] + _ux * c2 - _uy * s2
        v2y = h2[1] + _ux * s2 + _uy * c2
        
        en1x = _enx * c1 - _eny * s1
        en1y = _enx * s1 + _eny * c1
        en2x = _enx * c2 - _eny * s2
        en2y = _enx * s2 + _eny * c2
        
        for i in range(6):
            nx, ny = en1x[i], en1y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return True
        for i in range(6):
            nx, ny = en2x[i], en2y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return True
        return False

    def is_valid_configuration(positions, angles_rad):
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i, 0], positions[i, 1], angles_rad[i])
                h2 = (positions[j, 0], positions[j, 1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    return False
        return True

    # Initial Configuration: Honeycomb
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2.0, 1.5])
    lattice_coords = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, -1), (-1, 1), (2, -1), (-2, 1),
        (1, 1), (-1, -1), (2, 0),
    ]
    base_positions = np.zeros((N, 2))
    base_angles = np.zeros(N)
    for idx, (i, j) in enumerate(lattice_coords):
        base_positions[idx] = i * a1 + j * a2
        base_angles[idx] = 0.0

    best_side = float('inf')
    best_positions = base_positions.copy()
    best_angles = base_angles.copy()
    best_center = np.zeros(2)

    # Evaluate initial
    side, center = compute_side_length(base_positions, base_angles)
    if is_valid_configuration(base_positions, base_angles):
        best_side = side
        best_positions = base_positions.copy()
        best_angles = base_angles.copy()
        best_center = center

    # SA Parameters
    T = 0.5
    T_min = 0.00001
    cooling = 0.99995
    step_size = 0.5
    step_decay = 0.99999
    
    current_positions = base_positions.copy()
    current_angles = base_angles.copy()
    current_side, current_center = compute_side_length(current_positions, current_angles)
    
    sa_time_budget = DEADLINE * 0.8
    step_count = 0
    max_steps = 500000
    accept_count = 0
    total_count = 0

    while time.time() - t0 < sa_time_budget and step_count < max_steps:
        step_count += 1
        total_count += 1
        
        idx = random.randint(0, N - 1)
        new_positions = current_positions.copy()
        new_angles = current_angles.copy()
        
        dx = random.gauss(0, step_size)
        dy = random.gauss(0, step_size)
        da = random.gauss(0, 10.0) # Degrees
        
        new_positions[idx, 0] += dx
        new_positions[idx, 1] += dy
        new_angles[idx] += da
        
        # Check collision only for moved hexagon
        valid = True
        angles_rad_new = np.radians(new_angles)
        h_moved = (new_positions[idx, 0], new_positions[idx, 1], angles_rad_new[idx])
        for j in range(N):
            if j == idx:
                continue
            h_other = (new_positions[j, 0], new_positions[j, 1], angles_rad_new[j])
            if not hexagons_disjoint_fast(h_moved, h_other):
                valid = False
                break
        
        if not valid:
            T *= cooling
            if T < T_min: T = T_min
            step_size *= step_decay
            if step_size < 0.01: step_size = 0.01
            continue
            
        new_side, new_center = compute_side_length(new_positions, new_angles)
        
        if new_side < current_side:
            current_positions = new_positions
            current_angles = new_angles
            current_side = new_side
            current_center = new_center
            accept_count += 1
            if new_side < best_side:
                best_side = new_side
                best_positions = new_positions.copy()
                best_angles = new_angles.copy()
                best_center = new_center
        else:
            delta = new_side - current_side
            if delta < 0: delta = 0
            if random.random() < math.exp(-delta / T):
                current_positions = new_positions
                current_angles = new_angles
                current_side = new_side
                current_center = new_center
                accept_count += 1
                if new_side < best_side:
                    best_side = new_side
                    best_positions = new_positions.copy()
                    best_angles = new_angles.copy()
                    best_center = new_center
        
        # Adaptive step size
        if total_count % 1000 == 0:
            acc_rate = accept_count / total_count
            if acc_rate < 0.1:
                step_size *= 0.8
                if step_size < 0.01: step_size = 0.01
            elif acc_rate > 0.5:
                step_size *= 1.2
                if step_size > 1.5: step_size = 1.5
            accept_count = 0
            total_count = 0
            
        T *= cooling
        if T < T_min: T = T_min
        step_size *= step_decay
        if step_size < 0.01: step_size = 0.01

    # Polish Phase
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 5.0:
        # Coordinate Descent Polish
        def polish_coordinate_descent(pos, ang, time_limit):
            angles_rad = np.radians(ang)
            current_side, _ = compute_side_length(pos, angles_rad)
            best_side_cd = current_side
            best_pos_cd = pos.copy()
            best_ang_cd = ang.copy()
            
            t_start = time.time()
            step = 0.1
            min_step = 0.001
            
            while time.time() - t_start < time_limit and step > min_step:
                improved = False
                for idx in range(N):
                    for dim in range(3):
                        for direction in [1, -1]:
                            new_pos = best_pos_cd.copy()
                            new_ang = best_ang_cd.copy()
                            
                            if dim == 0:
                                new_pos[idx, 0] += direction * step
                            elif dim == 1:
                                new_pos[idx, 1] += direction * step
                            else:
                                new_ang[idx] += direction * step
                            
                            angles_rad_new = np.radians(new_ang)
                            valid = True
                            h_moved = (new_pos[idx, 0], new_pos[idx, 1], angles_rad_new[idx])
                            for j in range(N):
                                if j == idx: continue
                                h_other = (new_pos[j, 0], new_pos[j, 1], angles_rad_new[j])
                                if not hexagons_disjoint_fast(h_moved, h_other):
                                    valid = False
                                    break
                            
                            if not valid:
                                continue
                            
                            new_side, _ = compute_side_length(new_pos, angles_rad_new)
                            
                            if new_side < best_side_cd - 1e-10:
                                best_pos_cd = new_pos
                                best_ang_cd = new_ang
                                best_side_cd = new_side
                                improved = True
                                break
                            
                        if time.time() - t_start > time_limit:
                            return best_side_cd, best_pos_cd, best_ang_cd
                    else:
                        continue
                    break
                else:
                    step *= 0.8
                    continue
                break
            
            return best_side_cd, best_pos_cd, best_ang_cd

        cd_time = min(remaining_time * 0.5, 20.0)
        cd_side, cd_pos, cd_ang = polish_coordinate_descent(best_positions, best_angles, cd_time)
        
        if cd_side < best_side:
            best_side = cd_side
            best_positions = cd_pos
            best_angles = cd_ang
            _, best_center = compute_side_length(best_positions, np.radians(best_angles))

    # Final verification
    angles_rad_final = np.radians(best_angles)
    if not is_valid_configuration(best_positions, angles_rad_final):
        # Fallback to base
        best_positions = base_positions.copy()
        best_angles = base_angles.copy()
        best_side, best_center = compute_side_length(base_positions, base_angles)

    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_center[0]), float(best_center[1]))
    side = float(best_side)
    angle = 0.0 # Fixed orientation
    
    return inner, center, side, angle

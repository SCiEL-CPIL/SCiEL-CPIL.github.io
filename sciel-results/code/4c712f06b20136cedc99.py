"""Hexagon packing: minimize enclosing regular hexagon side length.
Combines simulated annealing for inner placement with LP for outer hexagon."""
import math
import time
import numpy as np
from scipy.optimize import linprog

N = 11
_START = time.time()
_DEADLINE = _START + 145.0
_SQRT3 = math.sqrt(3)

def _hex_verts(cx, cy, ang_rad):
    """Vertices of unit regular hexagon, center (cx,cy), rotation ang_rad."""
    a = ang_rad
    r = 1.0
    xs = cx + r * np.cos(a + np.arange(6) * np.pi / 3.0)
    ys = cy + r * np.sin(a + np.arange(6) * np.pi / 3.0)
    return np.column_stack([xs, ys])

def _all_vertices(positions, angles):
    """Compute all 6N vertices for given positions and angles."""
    verts = []
    for i in range(N):
        verts.append(_hex_verts(positions[i, 0], positions[i, 1], angles[i]))
    return np.vstack(verts)

def _min_enclosing_hex(verts, outer_angle_deg):
    """Find min side length of regular hexagon (rotation outer_angle_deg) containing all verts.
    Returns (cx, cy, side) or None."""
    a0 = math.radians(outer_angle_deg)
    # Edge normals at outer_angle + 30 + 60k
    normals = np.array([(math.cos(a0 + math.radians(30 + 60 * k)),
                         math.sin(a0 + math.radians(30 + 60 * k))) for k in range(6)])
    M = len(verts)
    A = []
    b = []
    for i in range(M):
        vx, vy = verts[i]
        for nx, ny in normals:
            vd = vx * nx + vy * ny
            A.append([-nx, -ny, -1.0])
            b.append(-vd)
            A.append([nx, ny, -1.0])
            b.append(vd)
    A.append([0.0, 0.0, -1.0])
    b.append(0.0)
    c_obj = [0.0, 0.0, 1.0]
    try:
        res = linprog(c_obj, A_ub=np.array(A), b_ub=np.array(b),
                      bounds=[(None, None), (None, None), (0, None)],
                      method='highs')
        if res.success:
            cx, cy, R = res.x
            side = R / (_SQRT3 / 2)
            return cx, cy, side
    except Exception:
        pass
    return None

def _objective_fast(positions, angles, outer_angle_deg):
    """Fast objective: LP for enclosing hex side."""
    verts = _all_vertices(positions, angles)
    res = _min_enclosing_hex(verts, outer_angle_deg)
    if res is None:
        return float('inf')
    return res[2]

def _initial_positions(seed):
    """Honeycomb-like initial layout."""
    rng = np.random.RandomState(seed)
    sqrt3 = _SQRT3
    scale = 1.000001
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)
    pos = [(0, 0)]
    ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    pos.extend(ring1)
    ring2 = [(2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0),
             (-1, -1), (0, -2), (1, -2), (2, -1)]
    ring2_sorted = sorted(ring2, key=lambda p: p[0] ** 2 + p[1] ** 2)
    pos.extend(ring2_sorted[:4])
    positions = np.zeros((N, 2))
    for i in range(N):
        positions[i, 0] = pos[i][0] * u[0] + pos[i][1] * v[0] + rng.normal(0, 0.05)
        positions[i, 1] = pos[i][0] * u[1] + pos[i][1] * v[1] + rng.normal(0, 0.05)
    angles = rng.uniform(0, 2 * np.pi, N)
    return positions, angles

def _simulated_annealing(positions0, angles0, t_start, max_iters, outer_angles):
    """Simulated annealing over positions and angles."""
    positions = positions0.copy()
    angles = angles0.copy()
    best_side = float('inf')
    best_positions = positions.copy()
    best_angles = angles.copy()
    best_outer_angle = 0.0
    best_center = (0, 0)
    
    T = 0.5
    T_min = 0.001
    cool_rate = 0.995
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 10.0:
            break
        
        # Evaluate current
        for oa in outer_angles:
            side = _objective_fast(positions, angles, oa)
            if side < best_side:
                best_side = side
                best_positions = positions.copy()
                best_angles = angles.copy()
                best_outer_angle = oa
                verts = _all_vertices(positions, angles)
                res = _min_enclosing_hex(verts, oa)
                if res:
                    best_center = (res[0], res[1])
        
        # Generate perturbation
        new_positions = positions.copy()
        new_angles = angles.copy()
        
        # Perturb positions
        idx = np.random.randint(N)
        new_positions[idx] += np.random.normal(0, T, 2)
        
        # Sometimes perturb angle
        if np.random.rand() < 0.3:
            aidx = np.random.randint(N)
            new_angles[aidx] += np.random.normal(0, T * 0.5)
            new_angles[aidx] = new_angles[aidx] % (2 * np.pi)
        
        # Evaluate new
        new_side = float('inf')
        for oa in outer_angles:
            s = _objective_fast(new_positions, new_angles, oa)
            if s < new_side:
                new_side = s
        
        # Accept or reject
        if new_side < best_side or np.random.rand() < math.exp(-(new_side - best_side) / max(T, 1e-10)):
            positions = new_positions
            angles = new_angles
            if new_side < best_side:
                best_side = new_side
                best_positions = positions.copy()
                best_angles = angles.copy()
                # Find best outer angle for this config
                for oa in outer_angles:
                    s = _objective_fast(positions, angles, oa)
                    if s <= best_side + 1e-9:
                        best_outer_angle = oa
                        verts = _all_vertices(positions, angles)
                        res = _min_enclosing_hex(verts, oa)
                        if res:
                            best_center = (res[0], res[1])
                        break
        
        T = max(T * cool_rate, T_min)
    
    return best_positions, best_angles, best_side, best_center, best_outer_angle

def _fine_tune(positions, angles, outer_angle, t_start, max_iters):
    """Local refinement with small steps around best solution."""
    best_side = _objective_fast(positions, angles, outer_angle)
    best_pos = positions.copy()
    best_ang = angles.copy()
    
    step = 0.1
    for it in range(max_iters):
        if time.time() > _DEADLINE - 5.0:
            break
        improved = False
        for i in range(N):
            for dim in range(2):
                for sign in [1, -1]:
                    new_pos = best_pos.copy()
                    new_pos[i, dim] += sign * step
                    s = _objective_fast(new_pos, best_ang, outer_angle)
                    if s < best_side:
                        best_side = s
                        best_pos = new_pos
                        improved = True
                # Angle
                new_ang = best_ang.copy()
                new_ang[i] += sign * step * 0.5
                s = _objective_fast(best_pos, new_ang, outer_angle)
                if s < best_side:
                    best_side = s
                    best_ang = new_ang
                    improved = True
        if not improved:
            step *= 0.7
            if step < 0.001:
                break
    
    # Final LP for center
    verts = _all_vertices(best_pos, best_ang)
    res = _min_enclosing_hex(verts, outer_angle)
    center = (0, 0)
    if res:
        center = (res[0], res[1])
    
    return best_pos, best_ang, best_side, center, outer_angle

def construct():
    t_start = time.time()
    
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Outer angle candidates (search space due to hex symmetry, 0-60 deg)
    outer_angles_coarse = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0]
    
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Try multiple seeds with simulated annealing
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    iters_per_seed = 200
    
    for s in seeds:
        if time.time() > _DEADLINE - 15.0:
            break
        
        positions0, angles0 = _initial_positions(s)
        
        # Simulated annealing
        pos, ang, side, center, oang = _simulated_annealing(
            positions0, angles0, t_start, iters_per_seed, outer_angles_coarse)
        
        if side < best_side:
            best_side = side
            best_positions = pos
            best_angles = ang
            best_center = center
            best_outer_angle = oang
        
        # Fine tune
        if time.time() > _DEADLINE - 10.0:
            break
        pos2, ang2, side2, center2, oang2 = _fine_tune(
            pos, ang, oang, t_start, 100)
        
        if side2 < best_side:
            best_side = side2
            best_positions = pos2
            best_angles = ang2
            best_center = center2
            best_outer_angle = oang2
    
    # Final fine-grained outer angle search
    if time.time() < _DEADLINE - 5.0 and best_positions is not None:
        for oa in np.arange(max(0, best_outer_angle - 2.0), min(60.1, best_outer_angle + 2.0), 0.5):
            if time.time() > _DEADLINE - 5.0:
                break
            s = _objective_fast(best_positions, best_angles, oa)
            if s < best_side:
                best_side = s
                best_outer_angle = oa
                verts = _all_vertices(best_positions, best_angles)
                res = _min_enclosing_hex(verts, oa)
                if res:
                    best_center = (res[0], res[1])
    
    if best_positions is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Verify and compute final S
    verts = _all_vertices(best_positions, best_angles)
    res = _min_enclosing_hex(verts, best_outer_angle)
    if res:
        S_final = res[2]
        best_center = (res[0], res[1])
    else:
        S_final = best_side
    
    inner = [[float(best_positions[i, 0]), float(best_positions[i, 1]),
              float(math.degrees(best_angles[i]))] for i in range(N)]
    
    return inner, (float(best_center[0]), float(best_center[1])), float(S_final), float(best_outer_angle)

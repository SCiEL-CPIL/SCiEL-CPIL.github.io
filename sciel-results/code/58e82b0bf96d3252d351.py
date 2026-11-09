import numpy as np
import time
import scipy.optimize as opt

N = 11
SQRT3 = np.sqrt(3.0)
APOTHEM = SQRT3 / 2.0
EPS = 1e-4
MIN_DIST = SQRT3 + 0.05

def _hexagon_vertices(center, angle_deg=0.0):
    angle_rad = np.deg2rad(angle_deg)
    angles = np.arange(6) * np.pi / 3.0 + angle_rad
    x = center[0] + np.cos(angles)
    y = center[1] + np.sin(angles)
    return np.column_stack((x, y))

def _outer_hexagon_normals(angle_deg=0.0):
    angle_rad = np.deg2rad(angle_deg)
    angles = np.arange(6) * np.pi / 3.0 + np.pi / 6.0 + angle_rad
    nx = np.cos(angles)
    ny = np.sin(angles)
    return np.column_stack((nx, ny))

def _calculate_required_side_and_center(positions, angles, outer_angle_deg=0.0):
    normals = _outer_hexagon_normals(outer_angle_deg)
    n0 = normals[0]
    n1 = normals[1]
    n2 = normals[2]
    
    max_proj = [float('-inf'), float('-inf'), float('-inf')]
    min_proj = [float('inf'), float('inf'), float('inf')]
    
    for i in range(len(positions)):
        verts = _hexagon_vertices(positions[i], angles[i])
        
        projs0 = verts @ n0
        projs1 = verts @ n1
        projs2 = verts @ n2
        
        max_proj[0] = max(max_proj[0], np.max(projs0))
        min_proj[0] = min(min_proj[0], np.min(projs0))
        
        max_proj[1] = max(max_proj[1], np.max(projs1))
        min_proj[1] = min(min_proj[1], np.min(projs1))
        
        max_proj[2] = max(max_proj[2], np.max(projs2))
        min_proj[2] = min(min_proj[2], np.min(projs2))
        
    widths = [max_proj[i] - min_proj[i] for i in range(3)]
    max_width = max(widths)
    side = max_width / SQRT3
    
    mid0 = (max_proj[0] + min_proj[0]) / 2.0
    mid1 = (max_proj[1] + min_proj[1]) / 2.0
    
    A = np.array([[n0[0], n0[1]], [n1[0], n1[1]]])
    b = np.array([mid0, mid1])
    
    try:
        center = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        center = np.array([0.0, 0.0])
        
    return side, center

def _penalty(positions, angles):
    penalty = 0.0
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < MIN_DIST:
                penalty += (MIN_DIST - dist) ** 2 * 1000.0
    return penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0

    # Fallback: Hexagonal lattice
    spacing = SQRT3 + 0.05
    positions = []
    angles = []
    hex_coords = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            if abs(q + r) <= 3:
                hex_coords.append((q, r))
                
    for q, r in hex_coords:
        x = spacing * (q + r / 2.0)
        y = spacing * (SQRT3 * r / 2.0)
        positions.append([x, y])
        angles.append(0.0)
        
    positions = positions[:N]
    angles = angles[:N]
    
    centroid = np.mean(np.array(positions), axis=0)
    positions = [p - centroid for p in positions]
    positions = np.array(positions)
    angles = np.array(angles)
    
    side, center = _calculate_required_side_and_center(positions, angles, 0.0)
    best_side = side
    best_positions = positions.copy()
    best_angles = angles.copy()
    best_outer_center = center

    def objective(vars):
        pos = vars[:2*N].reshape(N, 2)
        ang = vars[2*N:]
        pen = _penalty(pos, ang)
        if pen > 1e-6:
            return 1e5 * pen
        side, _ = _calculate_required_side_and_center(pos, ang, 0.0)
        return side

    def run_optimization(init_pos, init_ang, method='L-BFGS-B'):
        nonlocal best_side, best_positions, best_angles, best_outer_center
        if time.time() > deadline:
            return
        try:
            x0 = np.concatenate([init_pos.flatten(), init_ang])
            if method == 'L-BFGS-B':
                res = opt.minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10})
            elif method == 'COBYLA':
                res = opt.minimize(objective, x0, method='COBYLA', options={'maxiter': 500, 'rhobeg': 0.1})
            else:
                return
            if time.time() > deadline:
                return
            pos = res.x[:2*N].reshape(N, 2)
            ang = res.x[2*N:]
            pen = _penalty(pos, ang)
            if pen < 1e-4:
                side, center = _calculate_required_side_and_center(pos, ang, 0.0)
                if side < best_side:
                    best_side = side
                    best_positions = pos
                    best_angles = ang
                    best_outer_center = center
        except Exception:
            pass

    run_optimization(best_positions, best_angles, 'L-BFGS-B')
    
    if time.time() < deadline - 20:
        for k in range(5):
            if time.time() > deadline - 10:
                break
            noise = np.random.uniform(-0.1, 0.1, (N, 2))
            init = best_positions + noise
            run_optimization(init, best_angles, 'L-BFGS-B')
            if time.time() < deadline - 20:
                noise2 = np.random.uniform(-0.5, 0.5, (N, 2))
                init2 = best_positions + noise2
                run_optimization(init2, best_angles, 'COBYLA')

    for k in range(3):
        if time.time() > deadline - 10:
            break
        random_pos = np.random.uniform(-5, 5, (N, 2))
        random_pos -= np.mean(random_pos, axis=0)
        random_ang = np.random.uniform(0, 60, N)
        run_optimization(random_pos, random_ang, 'L-BFGS-B')

    valid = True
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(best_positions[i] - best_positions[j])
            if dist < MIN_DIST - 1e-5:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        best_side, best_outer_center = _calculate_required_side_and_center(best_positions, best_angles, 0.0)

    return [list(p) + [a] for p, a in zip(best_positions, best_angles)], tuple(best_outer_center), best_side, best_outer_angle

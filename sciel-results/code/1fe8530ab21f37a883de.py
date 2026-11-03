import numpy as np
import time

N = 12
START_TIME = time.time()
TIME_LIMIT = 110.0 # Safety margin

def _hex_vertices(center, side, angle):
    cx, cy = center
    theta = np.deg2rad(angle)
    verts = []
    for k in range(6):
        a = theta + k * np.pi / 3
        x = cx + side * np.cos(a)
        y = cy + side * np.sin(a)
        verts.append((x, y))
    return verts

def _max_hex_radius(points, outer_angle):
    theta = np.deg2rad(outer_angle)
    max_proj = 0.0
    for p in points:
        px, py = p
        for k in range(6):
            a = theta + k * np.pi / 3
            nx = np.cos(a)
            ny = np.sin(a)
            proj = px * nx + py * ny
            if proj > max_proj:
                max_proj = proj
    return 2.0 * max_proj / np.sqrt(3.0)

def _check_overlap(p1, p2, tol=1e-9):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dist = np.sqrt(dx*dx + dy*dy)
    return dist < np.sqrt(3.0) - tol

def construct():
    global START_TIME
    if time.time() - START_TIME > TIME_LIMIT:
        # Fallback to simple lattice
        u = np.array([np.sqrt(3), 0.0])
        v = np.array([np.sqrt(3)/2, 1.5])
        positions = []
        for i in range(4):
            for j in range(3):
                pos = i * u + j * v
                positions.append(pos)
        cm = np.mean(positions, axis=0)
        positions = [p - cm for p in positions]
        angles = [0.0] * N
        all_verts = []
        for p, a in zip(positions, angles):
            verts = _hex_vertices(p, 1.0, a)
            all_verts.extend(verts)
        S = _max_hex_radius(all_verts, 0.0)
        return [ [p[0], p[1], a] for p, a in zip(positions, angles) ], (0.0, 0.0), S, 0.0

    # Initial 3x4 lattice
    u = np.array([np.sqrt(3), 0.0])
    v = np.array([np.sqrt(3)/2, 1.5])
    
    positions = []
    angles = [0.0] * N
    for i in range(4):
        for j in range(3):
            pos = i * u + j * v
            positions.append(pos)
            
    cm = np.mean(positions, axis=0)
    positions = [p - cm for p in positions]
    
    best_S = float('inf')
    best_alpha = 0.0
    best_positions = positions[:]
    
    # Coarse scan of outer angle
    for alpha in np.linspace(0, 60, 31):
        all_verts = []
        for p, a in zip(positions, angles):
            verts = _hex_vertices(p, 1.0, a)
            all_verts.extend(verts)
        
        S = _max_hex_radius(all_verts, alpha)
        if S < best_S:
            best_S = S
            best_alpha = alpha
            best_positions = positions[:]
            
    # Fine scan around best alpha
    for alpha in np.linspace(best_alpha - 5, best_alpha + 5, 51):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        all_verts = []
        for p, a in zip(best_positions, angles):
            verts = _hex_vertices(p, 1.0, a)
            all_verts.extend(verts)
        
        S = _max_hex_radius(all_verts, alpha)
        if S < best_S:
            best_S = S
            best_alpha = alpha
            best_positions = best_positions[:]

    # Relaxation
    alpha = best_alpha
    current_positions = best_positions[:]
    step_size = 0.05
    
    for _ in range(200):
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        forces = [np.zeros(2) for _ in range(N)]
        
        # Repulsion
        for i in range(N):
            for j in range(i+1, N):
                if _check_overlap(current_positions[i], current_positions[j]):
                    dx = current_positions[i][0] - current_positions[j][0]
                    dy = current_positions[i][1] - current_positions[j][1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist > 1e-9:
                        f = np.array([dx/dist, dy/dist]) * step_size
                    else:
                        f = np.array([step_size, 0.0])
                    forces[i] += f
                    forces[j] -= f
                    
        # Attraction to center
        for i in range(N):
            forces[i] -= current_positions[i] * 0.01 * step_size * 10 # Scale attraction
            
        new_positions = []
        for i in range(N):
            new_pos = current_positions[i] + forces[i]
            new_positions.append(new_pos)
            
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                if _check_overlap(new_positions[i], new_positions[j]):
                    valid = False
                    break
            if not valid: break
            
        if valid:
            all_verts = []
            for p, a in zip(new_positions, angles):
                verts = _hex_vertices(p, 1.0, a)
                all_verts.extend(verts)
            S_new = _max_hex_radius(all_verts, alpha)
            if S_new < best_S:
                best_S = S_new
                best_positions = new_positions[:]
            current_positions = new_positions[:]
        else:
            step_size *= 0.9 # Dampen if invalid

    return [ [p[0], p[1], a] for p, a in zip(best_positions, angles) ], (0.0, 0.0), best_S, best_alpha

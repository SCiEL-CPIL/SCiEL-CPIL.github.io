import math
import time
import random
import numpy as np

N = 11
DEADLINE = 145.0
START_TIME = time.time()

def _remaining_time():
    return DEADLINE - (time.time() - START_TIME)

def _hex_vertices(cx, cy, side, angle_deg):
    angle_rad = np.radians(angle_deg)
    verts = np.empty((6, 2))
    for k in range(6):
        a = angle_rad + k * np.pi / 3
        verts[k] = [cx + side * np.cos(a), cy + side * np.sin(a)]
    return verts

def _point_in_hex(px, py, cx, cy, side, angle_deg, tol=1e-9):
    dx = px - cx
    dy = py - cy
    angle_rad = np.radians(angle_deg)
    ca = np.cos(-angle_rad)
    sa = np.sin(-angle_rad)
    lx = dx * ca - dy * sa
    ly = dx * sa + dy * ca
    
    inradius = side * np.sqrt(3) / 2
    for k in range(6):
        a = np.radians(30 + k * 60)
        nx = np.cos(a)
        ny = np.sin(a)
        if lx * nx + ly * ny > inradius + tol:
            return False
    return True

def _hexagons_disjoint(verts1, verts2, tol=1e-7):
    def get_normals(verts):
        normals = []
        n = len(verts)
        for i in range(n):
            j = (i + 1) % n
            edge = verts[j] - verts[i]
            nx = -edge[1]
            ny = edge[0]
            norm = np.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                normals.append([nx/norm, ny/norm])
        return normals
    
    normals1 = get_normals(verts1)
    normals2 = get_normals(verts2)
    all_normals = normals1 + normals2
    
    min_sep = float('inf')
    for nx, ny in all_normals:
        proj1 = np.array([v[0]*nx + v[1]*ny for v in verts1])
        proj2 = np.array([v[0]*nx + v[1]*ny for v in verts2])
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2:
            sep = min2 - max1
        elif max2 < min1:
            sep = min1 - max2
        else:
            return False, 0.0
        
        if sep < min_sep:
            min_sep = sep
            
    return True, min_sep

def _min_enclosing_side(inner_positions, outer_center, outer_angle):
    cx, cy = outer_center
    best_side = 0.0
    
    for (ix, iy, iangle) in inner_positions:
        iverts = _hex_vertices(ix, iy, 1.0, iangle)
        for v in iverts:
            dx = v[0] - cx
            dy = v[1] - cy
            angle_rad = np.radians(outer_angle)
            ca = np.cos(-angle_rad)
            sa = np.sin(-angle_rad)
            lx = dx * ca - dy * sa
            ly = dx * sa + dy * ca
            
            r = np.sqrt(lx*lx + ly*ly)
            if r < 1e-12:
                continue
            
            theta = np.arctan2(ly, lx)
            best_cos = 0.0
            for k in range(6):
                a = np.radians(30 + k * 60)
                d = abs(((theta - a) % (2*np.pi) + np.pi) % (2*np.pi))
                d = min(d, 2*np.pi - d)
                c = np.cos(d)
                if c > best_cos:
                    best_cos = c
            
            if best_cos < 1e-12:
                best_side = max(best_side, 1000.0)
            else:
                required_side = 2.0 * r / (np.sqrt(3) * best_cos)
                best_side = max(best_side, required_side)
    
    return best_side

def _check_validity(inner_positions, outer_center, outer_side, outer_angle, tol=1e-6):
    cx, cy = outer_center
    
    for (ix, iy, iangle) in inner_positions:
        iverts = _hex_vertices(ix, iy, 1.0, iangle)
        for v in iverts:
            if not _point_in_hex(v[0], v[1], cx, cy, outer_side, outer_angle, tol=tol):
                return False
    
    all_verts = [_hex_vertices(ix, iy, 1.0, iangle) for (ix, iy, iangle) in inner_positions]
    for i in range(len(inner_positions)):
        for j in range(i+1, len(inner_positions)):
            disjoint, sep = _hexagons_disjoint(all_verts[i], all_verts[j], tol=tol)
            if not disjoint:
                return False
    
    return True

def _compute_energy(positions):
    outer_side = _min_enclosing_side(positions, (0.0, 0.0), 0.0)
    
    overlap_penalty = 0.0
    all_verts = [_hex_vertices(x, y, 1.0, th) for (x, y, th) in positions]
    
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            disjoint, sep = _hexagons_disjoint(all_verts[i], all_verts[j], tol=1e-9)
            if not disjoint:
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = 2.0 * np.sqrt(3)
                if dist < min_dist:
                    overlap_penalty += (min_dist - dist) ** 2
    
    contain_penalty = 0.0
    for (ix, iy, iangle) in positions:
        iverts = _hex_vertices(ix, iy, 1.0, iangle)
        for v in iverts:
            if not _point_in_hex(v[0], v[1], 0.0, 0.0, outer_side, 0.0, tol=1e-6):
                dx = v[0]
                dy = v[1]
                r = np.sqrt(dx*dx + dy*dy)
                theta = np.arctan2(dy, dx)
                best_cos = 0.0
                for k in range(6):
                    a = np.radians(30 + k * 60)
                    d = abs(((theta - a) % (2*np.pi) + np.pi) % (2*np.pi))
                    d = min(d, 2*np.pi - d)
                    c = np.cos(d)
                    if c > best_cos:
                        best_cos = c
                if best_cos < 1e-9:
                    required_r = 100.0
                else:
                    required_r = outer_side * np.sqrt(3) * best_cos / 2.0
                if r > required_r:
                    contain_penalty += (r - required_r) ** 2
    
    com_x = np.mean([p[0] for p in positions])
    com_y = np.mean([p[1] for p in positions])
    com_penalty = 0.01 * (com_x**2 + com_y**2)
    
    total_energy = outer_side + 10.0 * overlap_penalty + 10.0 * contain_penalty + com_penalty
    return total_energy

def _force_directed(positions, n_steps=500, dt=0.01):
    """
    Physics-based relaxation to separate overlapping hexagons and push them inward.
    """
    positions = [list(p) for p in positions]
    n = len(positions)
    
    for step in range(n_steps):
        if _remaining_time() < 5.0:
            break
            
        forces = [[0.0, 0.0] for _ in range(n)]
        verts = [_hex_vertices(x, y, 1.0, th) for (x, y, th) in positions]
        
        # Repulsion between overlapping hexagons
        for i in range(n):
            for j in range(i+1, n):
                disjoint, sep = _hexagons_disjoint(verts[i], verts[j], tol=-0.05)  # Allow slight overlap for force
                if not disjoint:
                    # Compute repulsive force based on centers
                    dx = positions[i][0] - positions[j][0]
                    dy = positions[i][1] - positions[j][1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < 1e-6:
                        # Random nudge if centers coincide
                        dx = random.gauss(0, 0.1)
                        dy = random.gauss(0, 0.1)
                        dist = np.sqrt(dx*dx + dy*dy)
                    
                    # Force magnitude proportional to overlap depth (approx)
                    min_dist = 2.0 * np.sqrt(3) * 0.9  # Slightly less than true min dist
                    overlap = min_dist - dist
                    if overlap > 0:
                        f = overlap * 5.0  # Spring constant
                        fx = f * dx / dist
                        fy = f * dy / dist
                        forces[i][0] += fx
                        forces[i][1] += fy
                        forces[j][0] -= fx
                        forces[j][1] -= fy
        
        # Attraction to center (to keep them compact)
        for i in range(n):
            forces[i][0] -= positions[i][0] * 0.1
            forces[i][1] -= positions[i][1] * 0.1
        
        # Update positions
        for i in range(n):
            positions[i][0] += forces[i][0] * dt
            positions[i][1] += forces[i][1] * dt
            positions[i][2] = (positions[i][2]) % 60.0  # Keep angle in range
        
        # Recompute vertices for next step
        verts = [_hex_vertices(x, y, 1.0, th) for (x, y, th) in positions]
    
    return positions

def _initial_hexagonal_lattice():
    positions = []
    positions.append([0.0, 0.0, 0.0])
    
    # First ring: 6 hexagons
    dist1 = 2.0 * np.sqrt(3)
    for k in range(6):
        a = np.radians(60 * k)
        positions.append([dist1 * np.cos(a), dist1 * np.sin(a), 0.0])
    
    # Second ring: 4 more hexagons to reach 11
    dist2 = dist1 * np.sqrt(3)
    angles2 = [30, 90, 150, 210]
    for a_deg in angles2:
        a = np.radians(a_deg)
        positions.append([dist2 * np.cos(a), dist2 * np.sin(a), 0.0])
    
    return positions

def _initial_random_cluster():
    positions = []
    positions.append([0.0, 0.0, 0.0])
    for i in range(1, N):
        r = random.uniform(1.5, 3.5)
        a = random.uniform(0, 2*np.pi)
        angle = random.uniform(0, 60)
        positions.append([r * np.cos(a), r * np.sin(a), angle])
    return positions

def _simulate_annealing(initial_positions, max_time=100.0):
    start_time = time.time()
    
    positions = [list(p) for p in initial_positions]
    
    current_energy = _compute_energy(positions)
    best_positions = [list(p) for p in positions]
    best_energy = current_energy
    
    T_initial = 5.0
    T_final = 0.01
    cooling_rate = 0.9995
    
    T = T_initial
    step_size = 1.0
    
    while T > T_final and (time.time() - start_time) < max_time:
        if _remaining_time() < 5.0:
            break
        
        new_positions = [list(p) for p in positions]
        
        idx = random.randint(0, N-1)
        
        dx = random.gauss(0, step_size)
        dy = random.gauss(0, step_size)
        dangle = random.gauss(0, step_size * 0.5)
        
        new_positions[idx][0] += dx
        new_positions[idx][1] += dy
        new_positions[idx][2] = (new_positions[idx][2] + dangle) % 60.0
        
        new_energy = _compute_energy(new_positions)
        
        delta_e = new_energy - current_energy
        if delta_e < 0 or random.random() < np.exp(-delta_e / T):
            positions = new_positions
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = [list(p) for p in positions]
        
        T *= cooling_rate
        
        if random.random() < 0.01:
            step_size *= 0.95
    
    best_side = _min_enclosing_side(best_positions, (0.0, 0.0), 0.0)
    
    return best_positions, (0.0, 0.0), best_side, 0.0

def _try_configuration(initial_positions, max_time=30.0):
    try:
        # Step 1: Force directed relaxation to separate overlaps
        relaxed_positions = _force_directed(initial_positions, n_steps=200, dt=0.02)
        
        # Step 2: Simulated annealing to fine-tune
        result = _simulate_annealing(relaxed_positions, max_time=max_time * 0.5)
        inner, center, side, cangle = result
        
        # Step 3: Verify validity
        if _check_validity(inner, center, side, cangle, tol=1e-5):
            return inner, center, side, cangle
        else:
            # Try to fix by expanding side length
            side += 0.05
            if _check_validity(inner, center, side, cangle, tol=1e-5):
                return inner, center, side, cangle
            else:
                return None
    except Exception:
        return None

def construct():
    start_time = time.time()
    
    initializations = [
        _initial_hexagonal_lattice,
        _initial_random_cluster
    ]
    
    best_result = None
    best_side = float('inf')
    
    num_inits = len(initializations)
    time_per_init = 60.0
    
    for i, init_func in enumerate(initializations):
        if _remaining_time() < 15.0:
            break
        
        init_pos = init_func()
        
        result = _try_configuration(init_pos, max_time=time_per_init)
        
        if result is not None:
            inner, center, side, cangle = result
            if side < best_side:
                best_side = side
                best_result = result
    
    if best_result is not None and _remaining_time() > 20.0:
        inner, center, side, cangle = best_result
        refined = _try_configuration(inner, max_time=_remaining_time() - 5.0)
        if refined is not None:
            inner_r, center_r, side_r, cangle_r = refined
            if side_r < best_side:
                best_side = side_r
                best_result = refined
    
    if best_result is None:
        inner = [[3.5 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        center = (0.0, 0.0)
        side = 20.0
        angle = 0.0
        return inner, center, side, angle
    
    return best_result

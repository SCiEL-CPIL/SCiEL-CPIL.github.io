"""Symmetry-constrained hexagon packing with simulated annealing."""
import time
import numpy as np
import random

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 115.0  # Safety margin before 150s kill
SQRT3 = np.sqrt(3)

def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _hex_vertices_batch(positions, angles, side=1.0):
    """Vectorized: return (N*6, 2) array of all vertices for N hexagons."""
    n = len(positions)
    pos = np.asarray(positions, dtype=np.float64)
    ang = np.radians(np.asarray(angles, dtype=np.float64))
    base_angles = np.array([0, 1, 2, 3, 4, 5]) * np.pi / 3
    total_angles = ang[:, None] + base_angles[None, :]
    offsets_x = side * np.cos(total_angles)
    offsets_y = side * np.sin(total_angles)
    verts_x = pos[:, 0:1] + offsets_x
    verts_y = pos[:, 1:2] + offsets_y
    return np.column_stack([verts_x.ravel(), verts_y.ravel()])

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    dx = px - cx
    dy = py - cy
    a = np.radians(angle_deg)
    ca, sa = np.cos(-a), np.sin(-a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    s = side
    if lx > s + 1e-10 or lx < -s - 1e-10:
        return False
    if SQRT3 * lx + ly > SQRT3 * s + 1e-10:
        return False
    if -SQRT3 * lx + ly > SQRT3 * s + 1e-10:
        return False
    if -SQRT3 * lx - ly > SQRT3 * s + 1e-10:
        return False
    if SQRT3 * lx - ly > SQRT3 * s + 1e-10:
        return False
    return True

def _min_side_vectorized(all_verts, cx, cy, outer_angle_deg):
    """
    Compute minimum outer side length S such that all vertices in all_verts are inside
    the outer hexagon centered at (cx,cy) with rotation outer_angle_deg.
    """
    verts = np.asarray(all_verts, dtype=np.float64)
    n_pts = len(verts)
    if n_pts == 0:
        return 0.0
    
    dx = verts[:, 0] - cx
    dy = verts[:, 1] - cy
    
    a = np.radians(outer_angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    
    c1 = lx
    c2 = -lx
    c3 = lx + ly / SQRT3
    c4 = -lx + ly / SQRT3
    c5 = -lx - ly / SQRT3
    c6 = lx - ly / SQRT3
    
    required = np.maximum(
        np.maximum(np.maximum(c1, c2), np.maximum(c3, c4)),
        np.maximum(c5, c6)
    )
    return float(np.max(required))

def _hexagon_overlap(verts1, verts2, tol=1e-9):
    """Check if two convex polygons overlap using SAT (Separating Axis Theorem)."""
    axes = []
    for verts in [verts1, verts2]:
        for i in range(len(verts)):
            v1 = verts[i]
            v2 = verts[(i + 1) % len(verts)]
            edge = v2 - v1
            nx, ny = edge[1], -edge[0]
            norm = np.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                axes.append([nx/norm, ny/norm])
    
    for axis in axes:
        proj1 = np.dot(verts1, axis)
        proj2 = np.dot(verts2, axis)
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 - tol or max2 < min1 - tol:
            return False
    return True

def _check_no_overlap_fast(positions, angles, tol=1e-7):
    """Check that no two inner hexagons overlap. Returns True if valid."""
    n = len(positions)
    verts_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i]))
    for i in range(n):
        for j in range(i + 1, n):
            if _hexagon_overlap(verts_list[i], verts_list[j], tol):
                return False
    return True

def _min_outer_side(inner_positions, inner_angles, outer_center, outer_angle, tol=1e-6):
    """Find minimum outer side length such that all inner hexagons fit inside the outer hexagon."""
    cx, cy = outer_center
    pos = np.asarray(inner_positions, dtype=np.float64)
    ang = np.asarray(inner_angles, dtype=np.float64)
    all_verts = _hex_vertices_batch(pos, ang, side=1.0)
    return _min_side_vectorized(all_verts, cx, cy, outer_angle)

def _check_validity(inner, outer_center, outer_side, outer_angle, tol=1e-6):
    """Check if packing is valid: no overlaps, all inside outer hexagon."""
    cx, cy = outer_center
    verts_list = []
    for x, y, ang in inner:
        verts = _hex_vertices(x, y, 1.0, ang)
        verts_list.append(verts)
    
    for i in range(len(verts_list)):
        for j in range(i + 1, len(verts_list)):
            if _hexagon_overlap(verts_list[i], verts_list[j]):
                return False
    
    outer_verts = _hex_vertices(cx, cy, outer_side, outer_angle)
    for verts in verts_list:
        for v in verts:
            if not _point_in_hexagon(v[0], v[1], cx, cy, outer_side, outer_angle):
                return False
    return True

def _initial_packing_symmetric():
    """
    Create initial hexagonal grid packing leveraging 6-fold symmetry.
    Place 1 at center, 6 in first ring, 4 in second ring with symmetry.
    """
    positions = []
    angles = []
    
    # Center hexagon
    positions.append([0.0, 0.0])
    angles.append(0.0)
    
    # First ring: 6 hexagons at distance 2 from center
    for i in range(6):
        a = i * np.pi / 3
        x = 2.0 * np.cos(a)
        y = 2.0 * np.sin(a)
        positions.append([x, y])
        angles.append(0.0)
    
    # Second ring: place 4 more hexagons with symmetry
    # Positions at distance 4 from center, at angles 0, 60, 120, 180
    for i in range(4):
        a = i * np.pi / 3
        x = 4.0 * np.cos(a)
        y = 4.0 * np.sin(a)
        positions.append([x, y])
        angles.append(0.0)
    
    positions = positions[:N]
    angles = angles[:N]
    return np.array(positions, dtype=np.float64), np.array(angles, dtype=np.float64)

def _simulated_annealing(pos, ang, outer_center, outer_angle, time_limit, initial_temp=1.0, cooling_rate=0.999):
    """
    Simulated annealing to optimize packing.
    Uses symmetry constraints: only move hexagons in a way that preserves 6-fold symmetry where possible.
    """
    n = N
    best_pos = pos.copy()
    best_ang = ang.copy()
    best_side = _min_outer_side(best_pos, best_ang, outer_center, outer_angle)
    
    current_pos = pos.copy()
    current_ang = ang.copy()
    current_side = best_side
    
    temp = initial_temp
    iteration = 0
    
    while time.time() < time_limit and temp > 0.001:
        iteration += 1
        
        # Generate a random perturbation
        # With some probability, perturb a single hexagon
        # With some probability, perturb a symmetric set (to maintain symmetry)
        if random.random() < 0.3 and n >= 7:
            # Perturb a symmetric pair (hexagons opposite to each other)
            # This helps maintain symmetry
            i = random.randint(1, 6)  # Skip center (index 0)
            j = (i + 3) % 6  # Opposite hexagon
            
            dx = random.gauss(0, 0.3)
            dy = random.gauss(0, 0.3)
            dang = random.gauss(0, 0.1)
            
            new_pos = current_pos.copy()
            new_ang = current_ang.copy()
            
            new_pos[i, 0] += dx
            new_pos[i, 1] += dy
            new_pos[j, 0] -= dx
            new_pos[j, 1] -= dy
            new_ang[i] += dang
            new_ang[j] -= dang
        else:
            # Perturb a single hexagon
            i = random.randint(0, n - 1)
            dx = random.gauss(0, 0.3)
            dy = random.gauss(0, 0.3)
            dang = random.gauss(0, 0.1)
            
            new_pos = current_pos.copy()
            new_ang = current_ang.copy()
            
            new_pos[i, 0] += dx
            new_pos[i, 1] += dy
            new_ang[i] += dang
        
        # Check validity
        if not _check_no_overlap_fast(new_pos, new_ang):
            temp *= cooling_rate
            continue
        
        new_side = _min_outer_side(new_pos, new_ang, outer_center, outer_angle)
        
        # Accept or reject
        delta = new_side - current_side
        if delta < 0 or random.random() < np.exp(-delta / temp):
            current_pos = new_pos
            current_ang = new_ang
            current_side = new_side
            
            if current_side < best_side:
                best_pos = current_pos.copy()
                best_ang = current_ang.copy()
                best_side = current_side
        
        temp *= cooling_rate
    
    return best_pos, best_ang, best_side

def _local_search_fine(pos, ang, outer_center, outer_angle, step, min_step, time_limit):
    """Perform fine local search with decreasing step sizes."""
    n = N
    best_side = _min_outer_side(pos, ang, outer_center, outer_angle)
    best_pos = pos.copy()
    best_ang = ang.copy()
    
    while step >= min_step and time.time() < time_limit:
        improved = False
        for i in range(n):
            if time.time() > time_limit:
                break
            # Try moving in x and y directions
            for dx in [-step, step]:
                if time.time() > time_limit:
                    break
                new_pos = best_pos.copy()
                new_pos[i, 0] += dx
                if not _check_no_overlap_fast(new_pos, best_ang):
                    continue
                new_side = _min_outer_side(new_pos, best_ang, outer_center, outer_angle)
                if new_side < best_side - 1e-10:
                    best_pos = new_pos
                    best_side = new_side
                    improved = True
                    break
            if improved:
                continue
            for dy in [-step, step]:
                if time.time() > time_limit:
                    break
                new_pos = best_pos.copy()
                new_pos[i, 1] += dy
                if not _check_no_overlap_fast(new_pos, best_ang):
                    continue
                new_side = _min_outer_side(new_pos, best_ang, outer_center, outer_angle)
                if new_side < best_side - 1e-10:
                    best_pos = new_pos
                    best_side = new_side
                    improved = True
                    break
            if improved:
                continue
            # Try rotating
            for dang in [-step, step]:
                if time.time() > time_limit:
                    break
                new_ang = best_ang.copy()
                new_ang[i] += dang
                if not _check_no_overlap_fast(best_pos, new_ang):
                    continue
                new_side = _min_outer_side(best_pos, new_ang, outer_center, outer_angle)
                if new_side < best_side - 1e-10:
                    best_ang = new_ang
                    best_side = new_side
                    improved = True
                    break
        
        if not improved:
            step *= 0.7
    
    return best_pos, best_ang, best_side

def construct():
    global START_TIME, DEADLINE
    
    # Outer hexagon center and angle (fixed at origin, 0 degrees for simplicity)
    outer_cx, outer_cy = 0.0, 0.0
    outer_angle = 0.0
    
    # Phase 1: Get a good initial packing using symmetry
    pos, ang = _initial_packing_symmetric()
    
    # Phase 2: Simulated annealing for global optimization
    if time.time() < DEADLINE - 50.0:
        try:
            pos_sa, ang_sa, side_sa = _simulated_annealing(
                pos, ang, (outer_cx, outer_cy), outer_angle,
                time_limit=time.time() + 40.0,
                initial_temp=0.5,
                cooling_rate=0.9995
            )
            if side_sa < _min_outer_side(pos, ang, (outer_cx, outer_cy), outer_angle):
                pos, ang = pos_sa, ang_sa
        except Exception:
            pass
    
    # Phase 3: Fine local search
    pos, ang, best_side = _local_search_fine(
        pos, ang, (outer_cx, outer_cy), outer_angle,
        step=0.1, min_step=0.001, time_limit=time.time() + 30.0
    )
    
    # Phase 4: Ultra-fine polish
    if time.time() < DEADLINE - 10.0:
        pos, ang, best_side = _local_search_fine(
            pos, ang, (outer_cx, outer_cy), outer_angle,
            step=0.01, min_step=0.0001, time_limit=time.time() + 15.0
        )
    
    # Final verification
    inner_list = [[float(pos[i][0]), float(pos[i][1]), float(ang[i])] for i in range(N)]
    
    # Recompute the exact side length
    best_side = _min_outer_side(pos, ang, (outer_cx, outer_cy), outer_angle)
    
    if not _check_validity(inner_list, (outer_cx, outer_cy), best_side, outer_angle):
        # Fallback: use initial packing
        pos_init, ang_init = _initial_packing_symmetric()
        inner_list = [[float(pos_init[i][0]), float(pos_init[i][1]), float(ang_init[i])] for i in range(N)]
        best_side = _min_outer_side(pos_init, ang_init, (outer_cx, outer_cy), outer_angle)
        
        if not _check_validity(inner_list, (outer_cx, outer_cy), best_side, outer_angle):
            # Last resort: spread them out
            inner_list = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            return inner_list, (0.0, 0.0), 40.0, 0.0
    
    return inner_list, (float(outer_cx), float(outer_cy)), float(best_side), float(outer_angle)

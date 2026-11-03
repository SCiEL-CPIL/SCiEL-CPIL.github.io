"""Optimized hexagon packing for N=11 unit hexagons in a regular hexagon."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 115.0  # Safety margin before 150s kill

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
    sqrt3 = np.sqrt(3)
    if lx > s + 1e-10 or lx < -s - 1e-10:
        return False
    if sqrt3 * lx + ly > sqrt3 * s + 1e-10:
        return False
    if -sqrt3 * lx + ly > sqrt3 * s + 1e-10:
        return False
    if -sqrt3 * lx - ly > sqrt3 * s + 1e-10:
        return False
    if sqrt3 * lx - ly > sqrt3 * s + 1e-10:
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
    
    sqrt3 = np.sqrt(3)
    
    c1 = lx
    c2 = -lx
    c3 = lx + ly / sqrt3
    c4 = -lx + ly / sqrt3
    c5 = -lx - ly / sqrt3
    c6 = lx - ly / sqrt3
    
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

def _initial_packing():
    """Create initial hexagonal grid packing for N hexagons."""
    positions = []
    angles = []
    sqrt3 = np.sqrt(3)
    
    # Start with the 7-hexagon cluster (center + 6 neighbors)
    positions.append([0.0, 0.0])
    angles.append(0.0)
    for i in range(6):
        a = i * np.pi / 3
        x = 2.0 * np.cos(a)
        y = 2.0 * np.sin(a)
        positions.append([x, y])
        angles.append(0.0)
    
    # Add 4 more hexagons at the next ring positions
    extra_positions = [
        [4.0, 0.0],
        [-4.0, 0.0],
        [2.0, 2.0 * sqrt3],
        [-2.0, 2.0 * sqrt3],
    ]
    for ep in extra_positions:
        positions.append(list(ep))
        angles.append(0.0)
    
    positions = positions[:N]
    angles = angles[:N]
    return np.array(positions, dtype=np.float64), np.array(angles, dtype=np.float64)

def _smooth_objective(params, outer_center, outer_angle, beta=50.0):
    """
    Smooth log-sum-exp objective that approximates the minimum outer side length.
    params: flat array [x_0, y_0, ..., x_{n-1}, y_{n-1}, ang_0, ..., ang_{n-1}]
    """
    n = N
    pos = params[:2*n].reshape(n, 2)
    ang = params[2*n:]
    cx, cy = outer_center
    
    all_verts = _hex_vertices_batch(pos, ang, side=1.0)
    
    dx = all_verts[:, 0] - cx
    dy = all_verts[:, 1] - cy
    a = np.radians(outer_angle)
    ca, sa = np.cos(a), np.sin(a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    
    sqrt3 = np.sqrt(3)
    
    c1 = lx
    c2 = -lx
    c3 = lx + ly / sqrt3
    c4 = -lx + ly / sqrt3
    c5 = -lx - ly / sqrt3
    c6 = lx - ly / sqrt3
    
    all_c = np.vstack([c1, c2, c3, c4, c5, c6])
    
    max_c = np.max(all_c)
    if max_c > 500.0 / beta:
        return float(max_c)
    shifted = all_c - max_c
    log_sum = np.log(np.sum(np.exp(beta * shifted)))
    smooth_max = max_c + np.log(np.sum(np.exp(beta * shifted))) / beta
    
    # Overlap penalty
    overlap_penalty = 0.0
    verts_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(pos[i][0], pos[i][1], 1.0, ang[i]))
    for i in range(n):
        for j in range(i + 1, n):
            dx_c = pos[i][0] - pos[j][0]
            dy_c = pos[i][1] - pos[j][1]
            dist_c = np.sqrt(dx_c*dx_c + dy_c*dy_c)
            if dist_c < 4.0:
                if _hexagon_overlap(verts_list[i], verts_list[j]):
                    overlap_penalty += max(0, 2.0 - dist_c)
    
    return float(smooth_max) + 100.0 * overlap_penalty

def _min_side_from_params(params, outer_center, outer_angle):
    """Compute exact minimum outer side from parameter vector."""
    n = N
    pos = params[:2*n].reshape(n, 2)
    ang = params[2*n:]
    return _min_outer_side(pos, ang, outer_center, outer_angle)

def _local_search(pos, ang, outer_center, outer_angle, step, min_step, time_limit):
    """Perform local search with decreasing step sizes."""
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

def _differential_evolution_packing(outer_center, outer_angle, max_iter=100, pop_size=30, time_limit=None):
    """Use differential evolution to find good initial packing."""
    if time_limit is None:
        time_limit = time.time() + 30.0
    
    # Bounds for positions and angles
    # Assume outer hexagon has side ~4-5, so positions in [-5, 5]
    bounds = [(-5, 5)] * (2*N) + [(-180, 180)] * N
    
    def objective(params):
        if time.time() > time_limit:
            return 1000.0
        pos = params[:2*N].reshape(N, 2)
        ang = params[2*N:]
        
        # Check overlaps first (cheap)
        verts_list = []
        for i in range(N):
            verts_list.append(_hex_vertices(pos[i][0], pos[i][1], 1.0, ang[i]))
        for i in range(N):
            for j in range(i + 1, N):
                dx_c = pos[i][0] - pos[j][0]
                dy_c = pos[i][1] - pos[j][1]
                dist_c = np.sqrt(dx_c*dx_c + dy_c*dy_c)
                if dist_c < 4.0:
                    if _hexagon_overlap(verts_list[i], verts_list[j]):
                        return 100.0 + dist_c  # Heavy penalty for overlap
        
        # Compute outer side
        return _min_outer_side(pos, ang, outer_center, outer_angle)
    
    # Start from initial packing
    pos0, ang0 = _initial_packing()
    x0 = np.concatenate([pos0.flatten(), ang0])
    
    result = minimize(
        objective,
        x0,
        method='Nelder-Mead',
        options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True}
    )
    
    if result.x is not None:
        pos = result.x[:2*N].reshape(N, 2)
        ang = result.x[2*N:]
        if _check_no_overlap_fast(pos, ang):
            return pos, ang
    
    return pos0, ang0

def construct():
    global START_TIME, DEADLINE
    
    # Outer hexagon center and angle (fixed at origin, 0 degrees for simplicity)
    outer_cx, outer_cy = 0.0, 0.0
    outer_angle = 0.0
    
    # Phase 1: Get a good initial packing
    pos, ang = _initial_packing()
    
    # Try to improve with optimization
    if time.time() < DEADLINE - 60.0:
        try:
            pos_opt, ang_opt = _differential_evolution_packing(
                (outer_cx, outer_cy), outer_angle, 
                max_iter=200, time_limit=time.time() + 40.0
            )
            if pos_opt is not None and ang_opt is not None:
                pos, ang = pos_opt, ang_opt
        except Exception:
            pass
    
    # Phase 2: Local search with decreasing step
    pos, ang, best_side = _local_search(
        pos, ang, (outer_cx, outer_cy), outer_angle,
        step=0.5, min_step=0.001, time_limit=time.time() + 40.0
    )
    
    # Phase 3: High-precision polish
    if time.time() < DEADLINE - 10.0:
        pos, ang, best_side = _local_search(
            pos, ang, (outer_cx, outer_cy), outer_angle,
            step=0.01, min_step=0.0001, time_limit=time.time() + 20.0
        )
    
    # Final verification
    inner_list = [[float(pos[i][0]), float(pos[i][1]), float(ang[i])] for i in range(N)]
    
    # Recompute the exact side length
    best_side = _min_outer_side(pos, ang, (outer_cx, outer_cy), outer_angle)
    
    if not _check_validity(inner_list, (outer_cx, outer_cy), best_side, outer_angle):
        # Fallback: use initial packing
        pos_init, ang_init = _initial_packing()
        inner_list = [[float(pos_init[i][0]), float(pos_init[i][1]), float(ang_init[i])] for i in range(N)]
        best_side = _min_outer_side(pos_init, ang_init, (outer_cx, outer_cy), outer_angle)
        
        if not _check_validity(inner_list, (outer_cx, outer_cy), best_side, outer_angle):
            # Last resort: spread them out
            inner_list = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            return inner_list, (0.0, 0.0), 40.0, 0.0
    
    return inner_list, (float(outer_cx), float(outer_cy)), float(best_side), float(outer_angle)

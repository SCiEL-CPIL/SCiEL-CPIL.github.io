"""Rigid-body hexagon packing: treat 11 hexagons as a cluster, optimize cluster pose."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 115.0

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

def _rigid_cluster_positions():
    """
    Return the 11 hexagon positions in the hexagonal lattice as a rigid cluster.
    Center at origin, all angles = 0.
    Positions: center + 6 neighbors + 4 from next ring.
    """
    sqrt3 = np.sqrt(3)
    positions = []
    # Center
    positions.append([0.0, 0.0])
    # 6 neighbors at distance 2 (tangent)
    for i in range(6):
        a = i * np.pi / 3
        x = 2.0 * np.cos(a)
        y = 2.0 * np.sin(a)
        positions.append([x, y])
    # 4 from next ring: at distance 4 along 4 of the 6 axes
    # Choose 4 that form a symmetric pattern
    extra_angles = [0, np.pi/3, 2*np.pi/3, np.pi]
    for a in extra_angles:
        x = 4.0 * np.cos(a)
        y = 4.0 * np.sin(a)
        positions.append([x, y])
    
    return np.array(positions[:N], dtype=np.float64)

def _apply_cluster_transform(positions, tx, ty, theta_deg, scale=1.0):
    """
    Apply rigid transformation to cluster positions.
    tx, ty: translation
    theta_deg: rotation of entire cluster
    scale: uniform scaling (should be 1.0 for unit hexagons, but we optimize it)
    Returns transformed positions and angles (all rotated by theta).
    """
    theta = np.radians(theta_deg)
    ct, st = np.cos(theta), np.sin(theta)
    n = len(positions)
    new_pos = np.zeros((n, 2))
    for i in range(n):
        x, y = positions[i]
        # Scale, rotate, translate
        xs = scale * x
        ys = scale * y
        xr = ct * xs - st * ys
        yr = st * xs + ct * ys
        new_pos[i, 0] = xr + tx
        new_pos[i, 1] = yr + ty
    # All inner hexagons rotate by theta
    new_angles = np.full(n, theta_deg)
    return new_pos, new_angles

def _cluster_objective(params, base_positions, outer_center, outer_angle):
    """
    Objective for rigid cluster optimization.
    params: [tx, ty, theta_deg, scale]
    Returns: outer side length + penalty if overlaps.
    """
    tx, ty, theta_deg, scale = params
    
    # Scale must be positive
    if scale <= 0.1:
        return 1e6
    
    new_pos, new_angles = _apply_cluster_transform(base_positions, tx, ty, theta_deg, scale)
    
    # Check overlaps
    if not _check_no_overlap_fast(new_pos, new_angles):
        return 1e6
    
    # Compute outer side length
    side = _min_outer_side(new_pos, new_angles, outer_center, outer_angle)
    return side

def _optimize_rigid_cluster(base_positions, outer_center, outer_angle, time_limit):
    """
    Optimize the rigid cluster pose using differential evolution.
    Returns best (tx, ty, theta_deg, scale) and corresponding side length.
    """
    bounds = [(-5, 5), (-5, 5), (0, 360), (0.5, 1.5)]
    
    def objective(params):
        if time.time() > time_limit:
            return 1e6
        return _cluster_objective(params, base_positions, outer_center, outer_angle)
    
    try:
        result = differential_evolution(
            objective, bounds,
            maxiter=50, tol=1e-6,
            popsize=15,
            seed=42,
            polish=True,
            init='sohs'
        )
        if result.success or result.fun < 1e5:
            best_params = result.x
            best_side = result.fun
        else:
            # Fallback: try a few manual configurations
            best_params = [0, 0, 0, 1.0]
            best_side = _cluster_objective(best_params, base_positions, outer_center, outer_angle)
    except Exception:
        best_params = [0, 0, 0, 1.0]
        best_side = _cluster_objective(best_params, base_positions, outer_center, outer_angle)
    
    return best_params, best_side

def _fine_tune_positions(pos, ang, outer_center, outer_angle, time_limit, max_iterations=3000):
    """
    Fine-tune individual hexagon positions (non-rigid) to squeeze out extra space.
    Uses local search with small perturbations.
    """
    n = N
    best_side = _min_outer_side(pos, ang, outer_center, outer_angle)
    best_pos = pos.copy()
    best_ang = ang.copy()
    step = 0.05
    min_step = 0.001
    iterations = 0
    
    while step >= min_step and time.time() < time_limit and iterations < max_iterations:
        iterations += 1
        improved = False
        
        for _ in range(3):
            if time.time() > time_limit:
                break
            
            new_pos = best_pos.copy()
            new_ang = best_ang.copy()
            
            i = np.random.randint(n)
            new_pos[i, 0] += np.random.uniform(-step, step)
            new_pos[i, 1] += np.random.uniform(-step, step)
            new_ang[i] += np.random.uniform(-step, step)
            
            if not _check_no_overlap_fast(new_pos, new_ang):
                continue
            
            new_side = _min_outer_side(new_pos, new_ang, outer_center, outer_angle)
            if new_side < best_side - 1e-10:
                best_pos = new_pos
                best_ang = new_ang
                best_side = new_side
                improved = True
        
        if not improved:
            step *= 0.85
    
    return best_pos, best_ang, best_side

def construct():
    global START_TIME, DEADLINE
    
    # Outer hexagon center and angle (fixed at origin, 0 degrees)
    outer_cx, outer_cy = 0.0, 0.0
    outer_angle = 0.0
    
    # Phase 1: Get rigid cluster base positions
    base_positions = _rigid_cluster_positions()
    best_pos = base_positions.copy()
    best_ang = np.zeros(N)
    best_side = _min_outer_side(best_pos, best_ang, (outer_cx, outer_cy), outer_angle)
    
    # Phase 2: Optimize rigid cluster pose
    if time.time() < DEADLINE - 30.0:
        params, side = _optimize_rigid_cluster(
            base_positions, (outer_cx, outer_cy), outer_angle,
            time_limit=time.time() + 25.0
        )
        if side < best_side:
            tx, ty, theta_deg, scale = params
            new_pos, new_ang = _apply_cluster_transform(base_positions, tx, ty, theta_deg, scale)
            if _check_no_overlap_fast(new_pos, new_ang):
                best_pos = new_pos
                best_ang = new_ang
                best_side = side
    
    # Phase 3: Fine-tune individual positions (non-rigid)
    if time.time() < DEADLINE - 10.0:
        pos, ang, side = _fine_tune_positions(
            best_pos, best_ang, (outer_cx, outer_cy), outer_angle,
            time_limit=time.time() + 10.0, max_iterations=2000
        )
        if side < best_side:
            best_pos = pos
            best_ang = ang
            best_side = side
    
    # Phase 4: Final fine-tune with smaller steps
    if time.time() < DEADLINE - 5.0:
        pos, ang, side = _fine_tune_positions(
            best_pos, best_ang, (outer_cx, outer_cy), outer_angle,
            time_limit=time.time() + 5.0, max_iterations=1000
        )
        if side < best_side:
            best_pos = pos
            best_ang = ang
            best_side = side
    
    # Final verification
    inner_list = [[float(best_pos[i][0]), float(best_pos[i][1]), float(best_ang[i])] for i in range(N)]
    
    # Recompute the exact side length
    best_side = _min_outer_side(best_pos, best_ang, (outer_cx, outer_cy), outer_angle)
    
    if not _check_validity(inner_list, (outer_cx, outer_cy), best_side, outer_angle):
        # Fallback: use initial rigid packing
        pos_init = _rigid_cluster_positions()
        ang_init = np.zeros(N)
        inner_list = [[float(pos_init[i][0]), float(pos_init[i][1]), float(ang_init[i])] for i in range(N)]
        best_side = _min_outer_side(pos_init, ang_init, (outer_cx, outer_cy), outer_angle)
        
        if not _check_validity(inner_list, (outer_cx, outer_cy), best_side, outer_angle):
            # Last resort: spread them out
            inner_list = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            return inner_list, (0.0, 0.0), 40.0, 0.0
    
    return inner_list, (float(outer_cx), float(outer_cy)), float(best_side), float(outer_angle)

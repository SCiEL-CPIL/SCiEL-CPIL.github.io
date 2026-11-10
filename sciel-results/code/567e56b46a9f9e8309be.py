# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, s, a_deg):
    """Compute 6 vertices of a regular hexagon."""
    a = np.radians(a_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _hex_separated(v1, v2, tol=1e-6):
    """Check if two hexagons (vertex arrays) are separated using all 12 edge normals."""
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


def _is_valid(cfg):
    """Check if all inner hexagons are pairwise non-overlapping."""
    n = 11
    verts_list = [_hex_vertices(cfg[i,0], cfg[i,1], 1.0, cfg[i,2]) for i in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if not _hex_separated(verts_list[i], verts_list[j]):
                return False
    return True


def _compute_outer_side_fast(inner_data, outer_angle=0.0):
    """Compute the minimum outer hexagon side length needed to contain all inner hexagons."""
    all_verts = np.array([_hex_vertices(r[0], r[1], 1.0, r[2]) for r in inner_data])
    
    max_proj = 0.0
    for i in range(6):
        theta = np.radians(outer_angle) + np.radians(30 + 60 * i)
        n = np.array([np.cos(theta), np.sin(theta)])
        proj = all_verts @ n
        max_proj = max(max_proj, np.max(proj))
    return 2.0 * max_proj / np.sqrt(3)


def _check_containment(inner_data, outer_center, outer_angle, side):
    """Check if all inner hexagons are contained in the outer hexagon."""
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], side, outer_angle)
    for hex_row in inner_data:
        v = _hex_vertices(hex_row[0], hex_row[1], 1.0, hex_row[2])
        for p in v:
            for i in range(6):
                p1 = outer_verts[i]
                p2 = outer_verts[(i+1) % 6]
                cross_val = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                if cross_val < -1e-6:
                    return False
    return True


def _build_hex_cluster_11():
    """
    Build a rigid hexagonal cluster of 11 unit hexagons.
    Structure: center + 6 in first ring + 4 in second ring at specific positions.
    All hexagons have the same orientation (angle=0).
    Returns positions as (11, 2) array.
    """
    sqrt3 = np.sqrt(3)
    dx_e = 3.0
    dx_ne = 1.5
    dy_ne = 2.0 * sqrt3
    
    positions = np.zeros((11, 2))
    positions[0] = [0.0, 0.0]
    positions[1] = [dx_e, 0.0]
    positions[2] = [dx_ne, dy_ne]
    positions[3] = [-dx_ne, dy_ne]
    positions[4] = [-dx_e, 0.0]
    positions[5] = [-dx_ne, -dy_ne]
    positions[6] = [dx_ne, -dy_ne]
    positions[7] = [dx_e, dy_ne]
    positions[8] = [-dx_e, dy_ne]
    positions[9] = [dx_e, -dy_ne]
    positions[10] = [-dx_e, -dy_ne]
    
    return positions


def _build_compact_cluster_11():
    """
    Build a more compact cluster by using a different topology.
    """
    sqrt3 = np.sqrt(3)
    dx_e = 3.0
    dx_ne = 1.5
    dy_ne = 2.0 * sqrt3
    
    positions = np.zeros((11, 2))
    positions[0] = [0.0, 0.0]
    positions[1] = [dx_e, 0.0]
    positions[2] = [2*dx_e, 0.0]
    positions[3] = [-dx_ne, dy_ne]
    positions[4] = [dx_ne, dy_ne]
    positions[5] = [dx_e + dx_ne, dy_ne]
    positions[6] = [2*dx_e + dx_ne, dy_ne]
    positions[7] = [0.0, 2*dy_ne]
    positions[8] = [dx_e, 2*dy_ne]
    positions[9] = [2*dx_e, 2*dy_ne]
    positions[10] = [3*dx_e, 2*dy_ne]
    
    return positions


def _build_staggered_11():
    """
    Build a staggered arrangement.
    """
    sqrt3 = np.sqrt(3)
    dx_e = 3.0
    dx_ne = 1.5
    dy_ne = 2.0 * sqrt3
    
    positions = np.zeros((11, 2))
    positions[0] = [0.0, 0.0]
    positions[1] = [dx_e, 0.0]
    positions[2] = [-dx_e, 0.0]
    positions[3] = [dx_ne, dy_ne]
    positions[4] = [-dx_ne, dy_ne]
    positions[5] = [dx_ne, -dy_ne]
    positions[6] = [-dx_ne, -dy_ne]
    positions[7] = [0.0, 2*dy_ne]
    positions[8] = [0.0, -2*dy_ne]
    positions[9] = [2*dx_e, 0.0]
    positions[10] = [-2*dx_e, 0.0]
    
    return positions


def _objective(params, base_positions, outer_angle):
    """
    Objective function for optimization: minimize outer hexagon side length.
    params: [tx, ty, rotation_deg]
    """
    tx, ty, rot_deg = params
    rot_rad = np.radians(rot_deg)
    cos_r, sin_r = np.cos(rot_rad), np.sin(rot_rad)
    rot_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    
    rotated = base_positions @ rot_matrix.T
    translated = rotated + np.array([tx, ty])
    
    cfg = np.column_stack([translated, np.zeros(11)])
    return _compute_outer_side_fast(cfg, outer_angle)


def _optimize_rigid_cluster(positions_base, outer_angle, time_budget):
    """
    Optimize a rigid cluster by finding the best global rotation and translation.
    """
    center = np.mean(positions_base, axis=0)
    centered = positions_base - center
    
    start_t = time.time()
    best_side = float('inf')
    best_params = np.array([0.0, 0.0, 0.0])
    
    # Coarse grid search over rotation
    for rot_deg in np.arange(0, 60, 1.0):
        if time.time() > start_t + time_budget:
            break
        params = np.array([0.0, 0.0, rot_deg])
        side = _objective(params, centered, outer_angle)
        if side < best_side:
            best_side = side
            best_params = params.copy()
    
    # Fine-tune rotation
    for rot_deg in np.arange(best_params[2] - 1.0, best_params[2] + 1.0, 0.1):
        if time.time() > start_t + time_budget:
            break
        params = np.array([0.0, 0.0, rot_deg])
        side = _objective(params, centered, outer_angle)
        if side < best_side:
            best_side = side
            best_params = params.copy()
    
    # Optimize translation using grid search
    tx_range = np.arange(-1.0, 1.0, 0.1)
    ty_range = np.arange(-1.0, 1.0, 0.1)
    for tx in tx_range:
        if time.time() > start_t + time_budget:
            break
        for ty in ty_range:
            params = np.array([tx, ty, best_params[2]])
            side = _objective(params, centered, outer_angle)
            if side < best_side:
                best_side = side
                best_params = params.copy()
    
    # Fine-tune translation
    for tx in np.arange(best_params[0] - 0.1, best_params[0] + 0.1, 0.02):
        if time.time() > start_t + time_budget:
            break
        for ty in np.arange(best_params[1] - 0.1, best_params[1] + 0.1, 0.02):
            params = np.array([tx, ty, best_params[2]])
            side = _objective(params, centered, outer_angle)
            if side < best_side:
                best_side = side
                best_params = params.copy()
    
    # Use scipy minimize for final refinement
    if time.time() < start_t + time_budget - 5:
        try:
            result = minimize(_objective, best_params, args=(centered, outer_angle),
                            method='Nelder-Mead',
                            options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 500})
            if result.fun < best_side:
                best_side = result.fun
                best_params = result.x
        except:
            pass
    
    # Apply optimal transformation
    tx, ty, rot_deg = best_params
    rot_rad = np.radians(rot_deg)
    cos_r, sin_r = np.cos(rot_rad), np.sin(rot_rad)
    rot_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    final_positions = (centered @ rot_matrix.T) + np.array([tx, ty])
    
    cfg = np.zeros((11, 3))
    cfg[:, :2] = final_positions
    cfg[:, 2] = 0.0
    
    return cfg, best_side


def _local_search(cfg, angle, rng, time_budget, start_step=0.1):
    """Local search to minimize outer side length."""
    best_cfg = cfg.copy()
    best_side = _compute_outer_side_fast(best_cfg, angle)
    current_cfg = cfg.copy()
    current_side = best_side
    step_size = start_step
    temperature = 0.05
    
    start_t = time.time()
    while time.time() - start_t < time_budget:
        new_cfg = current_cfg.copy()
        n_move = rng.integers(1, 4)
        idxs = rng.choice(11, size=n_move, replace=False)
        new_cfg[idxs, :2] += rng.normal(0, step_size, size=(n_move, 2))
        
        if _is_valid(new_cfg):
            s = _compute_outer_side_fast(new_cfg, angle)
            delta = s - current_side
            if delta < 0 or rng.random() < np.exp(-delta / max(temperature, 1e-8)):
                current_cfg = new_cfg.copy()
                current_side = s
                if s < best_side - 1e-10:
                    best_side = s
                    best_cfg = new_cfg.copy()
                    step_size = max(0.002, step_size * 0.97)
                temperature = max(0.003, temperature * 0.997)
            else:
                step_size = min(0.3, step_size * 1.02)
        else:
            step_size = min(0.3, step_size * 1.05)
    
    return best_cfg, best_side


def _refine_single(cfg, angle, rng, time_budget):
    """Fine-grained refinement of individual hexagon positions."""
    best_cfg = cfg.copy()
    best_side = _compute_outer_side_fast(best_cfg, angle)
    step_size = 0.02
    
    start_t = time.time()
    while time.time() - start_t < time_budget:
        idx = rng.integers(0, 11)
        new_cfg = best_cfg.copy()
        new_cfg[idx, :2] += rng.normal(0, step_size, size=2)
        
        if _is_valid(new_cfg):
            s = _compute_outer_side_fast(new_cfg, angle)
            if s < best_side - 1e-10:
                best_side = s
                best_cfg = new_cfg.copy()
                step_size = max(0.001, step_size * 0.93)
            else:
                step_size = min(0.05, step_size * 1.03)
        else:
            step_size = min(0.05, step_size * 1.06)
    
    return best_cfg, best_side


def _optimize_angle(cfg, angle, time_budget):
    """Optimize the outer hexagon angle."""
    best_angle = angle
    best_side = _compute_outer_side_fast(cfg, angle)
    
    start_t = time.time()
    # Coarse search
    for a in np.arange(angle - 15, angle + 15, 0.5):
        if time.time() > start_t + time_budget:
            break
        s = _compute_outer_side_fast(cfg, a)
        if s < best_side:
            best_side = s
            best_angle = a
    
    # Fine search
    for a in np.arange(best_angle - 0.5, best_angle + 0.5, 0.05):
        if time.time() > start_t + time_budget:
            break
        s = _compute_outer_side_fast(cfg, a)
        if s < best_side:
            best_side = s
            best_angle = a
    
    # Very fine
    for a in np.arange(best_angle - 0.05, best_angle + 0.05, 0.005):
        if time.time() > start_t + time_budget:
            break
        s = _compute_outer_side_fast(cfg, a)
        if s < best_side:
            best_side = s
            best_angle = a
    
    return best_cfg, best_side


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,), (x, y, angle_degrees)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 615
    
    best_side = float('inf')
    best_cfg = None
    best_angle = 0.0

    rng = np.random.default_rng(12345)

    # Phase 1: Try multiple initial cluster topologies with rigid optimization
    cluster_builders = [_build_hex_cluster_11, _build_compact_cluster_11, _build_staggered_11]
    
    for builder in cluster_builders:
        if time.time() > deadline - 150:
            break
        cluster_positions = builder()
        
        if time.time() > deadline - 50:
            break
        
        t_budget = min(40, (deadline - time.time()) * 0.12)
        if t_budget > 10:
            cfg_rigid, side_rigid = _optimize_rigid_cluster(cluster_positions, 0.0, t_budget)
            
            # Optimize angle for this configuration
            if time.time() < deadline - 30:
                angle_budget = min(15, (deadline - time.time()) * 0.1)
                if angle_budget > 3:
                    cfg_rigid, side_rigid = _optimize_angle(cfg_rigid, 0.0, angle_budget)
            
            if side_rigid < best_side:
                best_side = side_rigid
                best_cfg = cfg_rigid
                # Extract angle from optimization (it's embedded in the positions)
                best_angle = 0.0

    # Phase 2: Local search from best rigid cluster
    if best_cfg is not None and time.time() < deadline - 100:
        t_budget = min((deadline - time.time()) * 0.25, 100)
        if t_budget > 20:
            best_cfg, best_side = _local_search(best_cfg, best_angle, rng, t_budget)
        
        # Re-optimize angle
        if time.time() < deadline - 50:
            angle_budget = min(20, (deadline - time.time()) * 0.15)
            if angle_budget > 5:
                best_cfg, best_side = _optimize_angle(best_cfg, best_angle, angle_budget)

    # Phase 3: Multiple restarts with perturbations
    if best_cfg is not None and time.time() < deadline - 60:
        seeds = [42, 123, 999, 777, 555, 333, 888]
        for seed in seeds:
            if time.time() > deadline - 40:
                break
            rng2 = np.random.default_rng(seed)
            t_budget = min(30, (deadline - time.time()) * 0.2)
            if t_budget > 8:
                cfg_candidate = best_cfg.copy()
                n_pert = rng2.integers(2, 5)
                idxs = rng2.choice(11, size=n_pert, replace=False)
                cfg_candidate[idxs, :2] += rng2.normal(0, 0.06, size=(n_pert, 2))
                if _is_valid(cfg_candidate):
                    cfg_new, side_new = _local_search(cfg_candidate, best_angle, rng2, t_budget)
                    if side_new < best_side - 1e-8:
                        best_side = side_new
                        best_cfg = cfg_new

    # Phase 4: Fine refinement
    if best_cfg is not None and time.time() < deadline - 30:
        t_budget = min((deadline - time.time()) * 0.4, 80)
        if t_budget > 10:
            best_cfg, best_side = _refine_single(best_cfg, best_angle, rng, t_budget)

    # Final angle optimization
    if best_cfg is not None and time.time() < deadline - 15:
        angle_budget = min(20, (deadline - time.time()) * 0.5)
        if angle_budget > 3:
            best_cfg, best_side = _optimize_angle(best_cfg, best_angle, angle_budget)

    outer_hex_data = np.array([0.0, 0.0, best_angle])
    
    # Verify containment and adjust if needed
    if best_cfg is not None:
        if not _check_containment(best_cfg, np.array([0.0, 0.0]), best_angle, best_side):
            for _ in range(2000):
                if _check_containment(best_cfg, np.array([0.0, 0.0]), best_angle, best_side):
                    break
                best_side *= 1.0001

    if best_cfg is not None:
        if not _check_containment(best_cfg, np.array([0.0, 0.0]), best_angle, best_side):
            required = _compute_outer_side_fast(best_cfg, best_angle) * 1.001
            best_side = max(best_side, required)

    return best_cfg, outer_hex_data, float(best_side)


# EVOLVE-BLOCK-END

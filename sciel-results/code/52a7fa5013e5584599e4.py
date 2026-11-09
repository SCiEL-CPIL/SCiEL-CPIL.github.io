# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, d, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _hex_edge_normals(verts):
    n = len(verts)
    normals = np.zeros((n, 2))
    for i in range(n):
        p1 = verts[i]
        p2 = verts[(i + 1) % n]
        edge = p2 - p1
        nx, ny = edge[1], -edge[0]
        norm = np.linalg.norm([nx, ny])
        if norm > 1e-15:
            normals[i] = [nx / norm, ny / norm]
    return normals

def _separation_check(verts1, verts2, tol=1e-6):
    normals1 = _hex_edge_normals(verts1)
    normals2 = _hex_edge_normals(verts2)
    all_normals = np.vstack([normals1, normals2])
    for normal in all_normals:
        proj1 = verts1 @ normal
        proj2 = verts2 @ normal
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 < min2 - tol or max2 < min1 - tol:
            return True
    return False

def _is_inside_outer(verts_inner, outer_verts, tol=1e-6):
    n = len(outer_verts)
    for i in range(n):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % n]
        edge = p2 - p1
        for v in verts_inner:
            cross = edge[0] * (v[1] - p1[1]) - edge[1] * (v[0] - p1[0])
            if cross < -tol:
                return False
    return True

def _validate(inner_data, outer_data, s):
    n_inner = len(inner_data)
    if n_inner != 12:
        return False
    if not np.all(np.isfinite(inner_data)) or not np.all(np.isfinite(outer_data)):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    outer_verts = _hex_vertices(outer_data[0], outer_data[1], s, outer_data[2])
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False
    inner_verts_list = []
    for i in range(n_inner):
        x, y, ang = inner_data[i]
        verts = _hex_vertices(x, y, 1.0, ang)
        inner_verts_list.append(verts)
        for k in range(6):
            v1 = verts[k]
            v2 = verts[(k+1) % 6]
            if abs(np.linalg.norm(v2 - v1) - 1.0) > 1e-8:
                return False
    for i in range(n_inner):
        for j in range(i+1, n_inner):
            if not _separation_check(inner_verts_list[i], inner_verts_list[j]):
                return False
    for i in range(n_inner):
        if not _is_inside_outer(inner_verts_list[i], outer_verts):
            return False
    return True

def _compute_min_outer(inner_data, outer_center, outer_angle):
    lo = 1.0
    hi = 50.0
    ok = False
    for _ in range(20):
        outer_verts = _hex_vertices(outer_center[0], outer_center[1], hi, outer_angle)
        ok = True
        for i in range(len(inner_data)):
            x, y, ang = inner_data[i]
            verts = _hex_vertices(x, y, 1.0, ang)
            if not _is_inside_outer(verts, outer_verts):
                ok = False
                break
        if ok:
            break
        hi *= 2
    if not ok:
        return hi
    for _ in range(60):
        mid = (lo + hi) / 2
        outer_verts = _hex_vertices(outer_center[0], outer_center[1], mid, outer_angle)
        ok = True
        for i in range(len(inner_data)):
            x, y, ang = inner_data[i]
            verts = _hex_vertices(x, y, 1.0, ang)
            if not _is_inside_outer(verts, outer_verts):
                ok = False
                break
        if ok:
            hi = mid
        else:
            lo = mid
    return hi

def _build_contact_graph_positions(dists, rel_angles, global_angle):
    """
    Build positions from contact graph parameters.
    dists: (11,) array of distances from hex i to its parent
    rel_angles: (11,) array of relative angles from parent to child
    global_angle: rotation of the whole configuration
    
    Contact graph (tree structure):
    0 is root
    1->0, 2->0, 3->0, 4->0, 5->0, 6->0 (ring 1: 6 children of 0)
    7->1, 8->2, 9->3, 10->4, 11->5 (ring 2: 5 children)
    """
    positions = np.zeros((12, 2))
    angles = np.zeros(12)
    
    # Root at origin
    positions[0] = [0, 0]
    angles[0] = global_angle
    
    # Define parent and direction for each non-root node
    # parent[i] = parent index, dist[i] = distance, rel_ang[i] = angle from parent's frame
    parents = [None, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5]
    
    for i in range(1, 12):
        p = parents[i]
        # The direction from parent to child
        # rel_angles[i] is the angle in the global frame (simplified)
        theta = np.radians(rel_angles[i])
        d = dists[i-1]  # dists indexed 0-10 for nodes 1-11
        positions[i] = positions[p] + d * np.array([np.cos(theta), np.sin(theta)])
        # Angle of child hexagon
        angles[i] = global_angle
    
    return positions, angles

def _contact_graph_objective(params, deadline):
    """
    params: [d1..d11, ra1..ra11, global_angle, outer_angle]
    11 distances + 11 relative angles + 1 global + 1 outer = 24 params
    """
    if time.time() > deadline:
        return 1e10
    
    dists = params[0:11]
    rel_angles = params[11:22]
    global_angle = params[22]
    outer_angle = params[23]
    
    # Ensure distances are positive
    if np.any(dists < 1.5):
        return 1e10
    
    positions, inner_angles = _build_contact_graph_positions(dists, rel_angles, global_angle)
    
    # Compute centroid and center configuration there
    centroid = np.mean(positions, axis=0)
    positions_centered = positions - centroid
    
    inner_data = np.column_stack([positions_centered, inner_angles])
    
    # Try to find minimum outer side
    outer_center = np.array([0.0, 0.0])
    s = _compute_min_outer(inner_data, outer_center, outer_angle)
    
    if not np.isfinite(s) or s < np.sqrt(12) - 1e-9:
        return 1e10
    
    outer_data = np.array([0.0, 0.0, outer_angle])
    
    if not _validate(inner_data, outer_data, s):
        return 1e10
    
    return s

def _optimize_contact_graph(deadline):
    """Optimize contact graph parameters."""
    from scipy.optimize import minimize
    
    best_value = -np.inf
    best_params = None
    
    # Initial guesses: hexagonal lattice spacing ~3.46 for unit hexagons
    base_dist = 3.46
    base_ang = 60.0
    
    starts = []
    for angle_offset in [0, 30, 60]:
        for dist_scale in [0.95, 1.0, 1.05, 1.1]:
            d = [base_dist * dist_scale] * 11
            ra = []
            for i in range(6):
                ra.append(angle_offset + i * 60)
            for i in range(5):
                ra.append(angle_offset + (i+1) * 60 + 30)  # offset for ring 2
            ra.append(angle_offset)  # global
            ra.append(angle_offset)  # outer
            starts.append(np.array(d + ra))
    
    bounds = [(1.5, 6.0)] * 11 + [(-180, 180)] * 11 + [(0, 360), (0, 360)]
    
    for start in starts:
        if time.time() > deadline - 60:
            break
        try:
            res = minimize(
                lambda p: _contact_graph_objective(p, deadline),
                start,
                method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 1500, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if res.fun < 1e9:
                val = 1.0 / res.fun
                if val > best_value:
                    best_value = val
                    best_params = res.x
        except Exception:
            pass
    
    return best_value, best_params

def _refine_free_positions(inner_data, outer_data, s, deadline):
    """Final refinement with free positions."""
    from scipy.optimize import minimize
    
    def objective_free(x):
        if time.time() > deadline:
            return 1e10
        inner_x = x[0:36].reshape(12, 3)
        outer_x = x[36:39]
        s_min = _compute_min_outer(inner_x, outer_x[:2], outer_x[2])
        if not np.isfinite(s_min) or s_min < np.sqrt(12) - 1e-9:
            return 1e10
        outer_data_local = np.array([outer_x[0], outer_x[1], outer_x[2]])
        if not _validate(inner_x, outer_data_local, s_min):
            return 1e10
        return s_min
    
    x0 = np.concatenate([inner_data.flatten(), outer_data.flatten()])
    
    res = minimize(
        objective_free,
        x0,
        method='Nelder-Mead',
        options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-9}
    )
    
    if res.fun < 1e9:
        inner_new = res.x[0:36].reshape(12, 3)
        outer_new = res.x[36:39]
        s_new = _compute_min_outer(inner_new, outer_new[:2], outer_new[2])
        if _validate(inner_new, outer_new, s_new):
            return inner_new, outer_new, s_new
    
    return None

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580
    
    best_value = -np.inf
    best_result = None
    
    # Phase 1: Contact graph optimization
    value, params = _optimize_contact_graph(deadline)
    
    if value > best_value and params is not None:
        best_value = value
        dists = params[0:11]
        rel_angles = params[11:22]
        global_angle = params[22]
        outer_angle = params[23]
        
        positions, inner_angles = _build_contact_graph_positions(dists, rel_angles, global_angle)
        centroid = np.mean(positions, axis=0)
        positions_centered = positions - centroid
        inner_data = np.column_stack([positions_centered, inner_angles])
        
        outer_center = np.array([0.0, 0.0])
        s = _compute_min_outer(inner_data, outer_center, outer_angle)
        outer_data = np.array([0.0, 0.0, outer_angle])
        
        if _validate(inner_data, outer_data, s):
            best_result = (inner_data, outer_data, s)
    
    # Phase 2: Free-position refinement if time permits
    elapsed = time.time() - start_time
    if elapsed < 400 and best_result is not None:
        try:
            inner_data, outer_data, s = best_result
            refined = _refine_free_positions(inner_data, outer_data, s, deadline)
            if refined is not None:
                inner_new, outer_new, s_new = refined
                val_new = 1.0 / s_new
                if val_new > best_value:
                    best_value = val_new
                    best_result = (inner_new, outer_new, s_new)
        except Exception:
            pass
    
    # Fallback
    if best_result is None:
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
                [0, -4, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
        best_result = (inner_hex_data, outer_hex_data, outer_hex_side_length)
    
    return best_result[0], best_result[1], best_result[2]

# EVOLVE-BLOCK-END

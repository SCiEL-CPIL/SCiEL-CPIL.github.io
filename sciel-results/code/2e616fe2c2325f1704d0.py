# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

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

def _rigid_cluster_positions(rot_deg):
    """
    Generate 12 unit hexagon centers in a rigid honeycomb cluster.
    The cluster has a fixed internal structure; rot_deg rotates the whole cluster.
    Using a 3-4-3-2 row arrangement for compactness.
    """
    # Honeycomb lattice basis: e1 = (sqrt(3), 0), e2 = (sqrt(3)/2, 3/2)
    # For unit hexagons, the spacing between adjacent centers is sqrt(3)
    sq3 = np.sqrt(3)
    e1 = np.array([sq3, 0.0])
    e2 = np.array([sq3/2, 1.5])
    
    # Define 12 positions in lattice coordinates (i, j) where pos = i*e1 + j*e2
    # Use a compact arrangement:
    # Row 0 (j=0): i = -1, 0, 1  -> 3 hexagons
    # Row 1 (j=1): i = -1, 0, 1, 2 -> 4 hexagons (shifted right)
    # Row 2 (j=2): i = -1, 0, 1  -> 3 hexagons
    # Row 3 (j=3): i = 0, 1  -> 2 hexagons
    lattice_indices = [
        (-1, 0), (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (-1, 2), (0, 2), (1, 2),
        (0, 3), (1, 3),
    ]
    
    # Compute base positions
    base_positions = np.zeros((12, 2))
    for idx, (i, j) in enumerate(lattice_indices):
        base_positions[idx] = i * e1 + j * e2
    
    # Center the cluster at origin
    centroid = np.mean(base_positions, axis=0)
    base_positions -= centroid
    
    # Apply rotation
    theta = np.radians(rot_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    
    positions = base_positions @ R.T
    
    # Inner hexagons all have angle 30 degrees (flat-top orientation relative to cluster)
    inner_angle = 30.0
    return positions, inner_angle

def _objective_rigid(params):
    """
    params: [cluster_rot_deg, outer_rot_deg]
    Returns: s (minimize s)
    """
    cluster_rot, outer_rot = params
    
    positions, inner_angle = _rigid_cluster_positions(cluster_rot)
    inner_data = np.column_stack([positions, np.full(12, inner_angle)])
    
    # Outer hexagon centered at origin
    outer_center = np.array([0.0, 0.0])
    
    s = _compute_min_outer(inner_data, outer_center, outer_rot)
    
    if not np.isfinite(s) or s < np.sqrt(12) - 1e-9:
        return 1e10
    
    # Validate
    outer_data = np.array([0.0, 0.0, outer_rot])
    if not _validate(inner_data, outer_data, s):
        return 1e10
    
    return s

def _optimize_rigid_cluster():
    """Optimize the rigid cluster rotation and outer hexagon orientation."""
    best_s = np.inf
    best_params = None
    
    # Multiple starting points
    starts = [
        [0, 0],
        [0, 30],
        [0, 60],
        [30, 0],
        [30, 30],
        [30, 60],
        [60, 0],
        [60, 30],
        [60, 60],
        [15, 15],
        [45, 15],
        [15, 45],
    ]
    
    for start in starts:
        try:
            res = minimize(
                _objective_rigid,
                start,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res.fun < 1e9 and res.fun < best_s:
                best_s = res.fun
                best_params = res.x
        except Exception:
            pass
    
    return best_s, best_params

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580
    
    best_s = np.inf
    best_result = None
    
    # Phase 1: Rigid cluster optimization
    s_opt, params_opt = _optimize_rigid_cluster()
    elapsed = time.time() - start_time
    
    if s_opt < 1e9 and params_opt is not None:
        cluster_rot, outer_rot = params_opt
        positions, inner_angle = _rigid_cluster_positions(cluster_rot)
        inner_data = np.column_stack([positions, np.full(12, inner_angle)])
        outer_data = np.array([0.0, 0.0, outer_rot])
        
        if _validate(inner_data, outer_data, s_opt):
            best_s = s_opt
            best_result = (inner_data, outer_data, s_opt)
    
    # Phase 2: Fine-tune with more starting points if time permits
    elapsed = time.time() - start_time
    if elapsed < 400:
        additional_starts = [
            [i*5, j*5] for i in range(13) for j in range(13)
            if [i*5, j*5] not in [[0,0],[0,30],[0,60],[30,0],[30,30],[30,60],
                                   [60,0],[60,30],[60,60],[15,15],[45,15],[15,45]]
        ]
        for start in additional_starts:
            if time.time() > deadline - 30:
                break
            try:
                res = minimize(
                    _objective_rigid,
                    start,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res.fun < 1e9 and res.fun < best_s:
                    cluster_rot, outer_rot = res.x
                    positions, inner_angle = _rigid_cluster_positions(cluster_rot)
                    inner_data = np.column_stack([positions, np.full(12, inner_angle)])
                    outer_data = np.array([0.0, 0.0, outer_rot])
                    
                    if _validate(inner_data, outer_data, res.fun):
                        best_s = res.fun
                        best_result = (inner_data, outer_data, best_s)
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

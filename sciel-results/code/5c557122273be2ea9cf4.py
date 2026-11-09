# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog, minimize


def _hex_vertices(cx, cy, d, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with side d and angle angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _compute_min_s_for_rotation(vertices, rot_deg):
    """
    Given all vertices of inner hexagons (N, 2), find minimum side length s
    of a regular hexagon (rotated by rot_deg) that contains all vertices.
    Uses linear programming with variables [ox, oy, s].
    """
    rot = np.radians(rot_deg)
    n_pts = len(vertices)
    n_constraints = 6 * n_pts
    A = np.zeros((n_constraints, 3))
    b = np.zeros(n_constraints)
    
    idx = 0
    for i in range(6):
        angle = rot + np.radians(60 * i + 30)
        ci = np.cos(angle)
        si = np.sin(angle)
        for j in range(n_pts):
            A[idx, 0] = -ci
            A[idx, 1] = -si
            A[idx, 2] = -np.sqrt(3) / 2
            b[idx] = -(ci * vertices[j, 0] + si * vertices[j, 1])
            idx += 1
    
    c_obj = np.array([0, 0, 1])
    bounds = [(None, None), (None, None), (0, None)]
    
    res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    
    if res.status == 0:
        s = res.x[2]
        ox = res.x[0]
        oy = res.x[1]
        return s, (ox, oy)
    else:
        return np.inf, None


def _build_cluster(centers, angle_deg=0.0):
    """
    Build all vertices of the inner hexagons given their centers and a common rotation.
    Returns (72, 2) array of vertices.
    """
    a = np.radians(angle_deg)
    unit_angles = np.radians(60 * np.arange(6))
    unit_verts = np.column_stack([np.cos(unit_angles), np.sin(unit_angles)])
    
    cos_a, sin_a = np.cos(a), np.sin(a)
    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rot_unit_verts = unit_verts @ rot_matrix.T
    
    all_verts = np.zeros((72, 2))
    for j in range(12):
        for k in range(6):
            all_verts[j * 6 + k] = centers[j] + rot_unit_verts[k]
    
    return all_verts


def _validate_packing(inner_hex_data, outer_hex_data, s, eps=1e-6):
    """Check if the packing is valid. Returns True if valid."""
    if not np.all(np.isfinite(inner_hex_data)) or not np.all(np.isfinite(outer_hex_data)):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    
    ox, oy, oang = outer_hex_data
    outer_verts = _hex_vertices(ox, oy, s, oang)
    
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-8:
                return False
    
    for i in range(12):
        cx, cy, aang = inner_hex_data[i]
        verts = _hex_vertices(cx, cy, 1, aang)
        for k in range(6):
            v1 = verts[k]
            v2 = verts[(k + 1) % 6]
            edge_len = np.linalg.norm(v2 - v1)
            if abs(edge_len - 1) > 1e-8:
                return False
    
    for i in range(12):
        for j in range(i + 1, 12):
            vi = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1, inner_hex_data[i, 2])
            vj = _hex_vertices(inner_hex_data[j, 0], inner_hex_data[j, 1], 1, inner_hex_data[j, 2])
            
            normals = []
            for v in [vi, vj]:
                for k in range(6):
                    e = v[(k + 1) % 6] - v[k]
                    n = np.array([-e[1], e[0]])
                    n = n / np.linalg.norm(n)
                    normals.append(n)
            
            separated = False
            for n in normals:
                proj_i = np.dot(vi, n)
                proj_j = np.dot(vj, n)
                min_i, max_i = proj_i.min(), proj_i.max()
                min_j, max_j = proj_j.min(), proj_j.max()
                if max_i + eps < min_j or max_j + eps < min_i:
                    separated = True
                    break
            
            if not separated:
                return False
    
    for i in range(12):
        cx, cy, aang = inner_hex_data[i]
        verts = _hex_vertices(cx, cy, 1, aang)
        for v in verts:
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < -1e-6:
                    inside = False
                    break
            if not inside:
                return False
    
    return True


def _smooth_min_s_for_rotation(vertices, rot_deg):
    """
    Smooth surrogate for the minimum s problem using soft-min.
    Returns (s_approx, center) for use with gradient-based optimization.
    """
    rot = np.radians(rot_deg)
    n_pts = len(vertices)
    
    # For each of 6 directions, compute the required s and center
    # s_i = max_j (n_i . p_j + sqrt(3)/2 * s) ... this is coupled
    # Instead, use the LP result but smooth it
    s, center = _compute_min_s_for_rotation(vertices, rot_deg)
    if s == np.inf:
        return 1000.0, (0, 0)
    return s, center


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    """
    start_time = time.time()
    deadline = start_time + 620.0
    
    sqrt3 = np.sqrt(3)
    
    # Flower pattern: center + 6 neighbors + 5 from next ring
    # Slight expansion to ensure strict separation
    eps_gap = 1e-3
    scale = 1.0 + eps_gap
    
    centers_base = np.array([
        [0, 0],
        [2, 0],
        [1, sqrt3],
        [-1, sqrt3],
        [-2, 0],
        [-1, -sqrt3],
        [1, -sqrt3],
        [3, sqrt3],
        [0, 2 * sqrt3],
        [-3, sqrt3],
        [-3, -sqrt3],
        [0, -2 * sqrt3],
    ]) * scale
    
    best_s = np.inf
    best_inner = None
    best_outer = None
    
    def try_configuration(centers, theta_deg, phi_deg):
        nonlocal best_s, best_inner, best_outer
        cluster_verts = _build_cluster(centers, theta_deg)
        s, center = _compute_min_s_for_rotation(cluster_verts, phi_deg)
        if s < best_s:
            inner_hex_data = np.column_stack([centers, np.full(12, theta_deg)])
            outer_hex_data = np.array([center[0], center[1], phi_deg])
            if _validate_packing(inner_hex_data, outer_hex_data, s):
                best_s = s
                best_inner = inner_hex_data.copy()
                best_outer = outer_hex_data.copy()
    
    # Phase 1: Coarse grid search
    n_theta = 61
    n_phi = 61
    theta_vals = np.linspace(0, 60, n_theta)
    phi_vals = np.linspace(0, 60, n_phi)
    
    for theta_deg in theta_vals:
        if time.time() > deadline - 120:
            break
        for phi_deg in phi_vals:
            if time.time() > deadline - 120:
                break
            try_configuration(centers_base, theta_deg, phi_deg)
    
    # Phase 2: Fine search around best
    if best_inner is not None and time.time() < deadline - 180:
        best_theta = best_inner[0, 2]
        best_phi = best_outer[2]
        
        theta_fine = np.linspace(best_theta - 3, best_theta + 3, 61)
        phi_fine = np.linspace(best_phi - 3, best_phi + 3, 61)
        
        for theta_deg in theta_fine:
            if time.time() > deadline - 120:
                break
            for phi_deg in phi_fine:
                if time.time() > deadline - 120:
                    break
                try_configuration(centers_base, theta_deg, phi_deg)
    
    # Phase 3: Try different lattice configurations
    if time.time() < deadline - 180:
        # Alternative: more rectangular arrangement
        alt_centers = np.array([
            [0, 0],
            [2, 0],
            [4, 0],
            [1, sqrt3],
            [3, sqrt3],
            [-1, sqrt3],
            [-3, sqrt3],
            [-1, -sqrt3],
            [1, -sqrt3],
            [3, -sqrt3],
            [-3, -sqrt3],
            [0, 2 * sqrt3],
        ]) * scale
        
        for theta_deg in np.linspace(0, 60, 31):
            if time.time() > deadline - 90:
                break
            for phi_deg in np.linspace(0, 60, 31):
                if time.time() > deadline - 90:
                    break
                try_configuration(alt_centers, theta_deg, phi_deg)
    
    # Phase 4: Try compressed lattice
    if time.time() < deadline - 120:
        for compress in [0.998, 0.995, 0.99, 0.985, 0.98]:
            if time.time() > deadline - 60:
                break
            centers_comp = centers_base * (scale * compress)
            for theta_deg in np.linspace(0, 60, 31):
                if time.time() > deadline - 60:
                    break
                for phi_deg in np.linspace(0, 60, 31):
                    if time.time() > deadline - 60:
                        break
                    try_configuration(centers_comp, theta_deg, phi_deg)
    
    # Phase 5: Local optimization using Nelder-Mead on (theta, phi)
    if best_inner is not None and time.time() < deadline - 60:
        best_theta = best_inner[0, 2]
        best_phi = best_outer[2]
        
        def objective(params):
            theta, phi = params
            # Wrap angles to [0, 60)
            theta = theta % 60
            phi = phi % 60
            cluster_verts = _build_cluster(centers_base, theta)
            s, center = _compute_min_s_for_rotation(cluster_verts, phi)
            if s == np.inf:
                return 1000.0
            inner_hex_data = np.column_stack([centers_base, np.full(12, theta)])
            outer_hex_data = np.array([center[0], center[1], phi])
            if not _validate_packing(inner_hex_data, outer_hex_data, s):
                return 1000.0
            return s
        
        x0 = np.array([best_theta, best_phi])
        res = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
        
        if res.fun < best_s:
            theta_opt, phi_opt = res.x % 60
            cluster_verts = _build_cluster(centers_base, theta_opt)
            s_opt, center_opt = _compute_min_s_for_rotation(cluster_verts, phi_opt)
            if s_opt < best_s:
                inner_hex_data = np.column_stack([centers_base, np.full(12, theta_opt)])
                outer_hex_data = np.array([center_opt[0], center_opt[1], phi_opt])
                if _validate_packing(inner_hex_data, outer_hex_data, s_opt):
                    best_s = s_opt
                    best_inner = inner_hex_data.copy()
                    best_outer = outer_hex_data.copy()
    
    # Phase 6: Very fine polish around the best
    if best_inner is not None and time.time() < deadline - 30:
        best_theta = best_inner[0, 2]
        best_phi = best_outer[2]
        
        for delta_t in np.linspace(-0.5, 0.5, 21):
            for delta_p in np.linspace(-0.5, 0.5, 21):
                if time.time() > deadline - 10:
                    break
                try_configuration(centers_base, best_theta + delta_t, best_phi + delta_p)
    
    # Fallback
    if best_inner is None:
        inner_hex_data = np.column_stack([centers_base, np.zeros(12)])
        s, center = _compute_min_s_for_rotation(_build_cluster(centers_base, 0), 0)
        outer_hex_data = np.array([center[0], center[1], 0])
        return inner_hex_data, outer_hex_data, s
    
    return best_inner, best_outer, best_s


# EVOLVE-BLOCK-END

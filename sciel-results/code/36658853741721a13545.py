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

    # Precompute the 6 normal directions
    normals = []
    for i in range(6):
        angle = rot + np.radians(60 * i + 30)
        normals.append((np.cos(angle), np.sin(angle)))

    idx = 0
    for ci, si in normals:
        A[idx:idx+n_pts, 0] = -ci
        A[idx:idx+n_pts, 1] = -si
        A[idx:idx+n_pts, 2] = -np.sqrt(3) / 2
        b[idx:idx+n_pts] = -(ci * vertices[:, 0] + si * vertices[:, 1])
        idx += n_pts

    c_obj = np.array([0.0, 0.0, 1.0])
    bounds = [(None, None), (None, None), (0, None)]

    res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')

    if res.status == 0:
        return res.x[2], (res.x[0], res.x[1])
    else:
        return np.inf, None


def _build_cluster(centers, angle_deg=0.0):
    """
    Build all 72 vertices of the 12 hexagons given their centers and a common rotation.
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
        all_verts[j*6:(j+1)*6] = centers[j] + rot_unit_verts

    return all_verts


def _validate_packing(inner_hex_data, outer_hex_data, s, eps=1e-6):
    """Check if the packing is valid. Returns True if valid."""
    if not np.all(np.isfinite(inner_hex_data)) or not np.all(np.isfinite(outer_hex_data)):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False

    ox, oy, oang = outer_hex_data
    outer_verts = _hex_vertices(ox, oy, s, oang)

    # Check outer vertices are distinct
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-8:
                return False

    # Precompute inner hexagon vertices
    inner_verts_list = []
    for i in range(12):
        cx, cy, aang = inner_hex_data[i]
        verts = _hex_vertices(cx, cy, 1, aang)
        # Check unit edges
        for k in range(6):
            v1 = verts[k]
            v2 = verts[(k + 1) % 6]
            if abs(np.linalg.norm(v2 - v1) - 1) > 1e-8:
                return False
        inner_verts_list.append(verts)

    # Check separation between inner hexagons using separating axis test
    for i in range(12):
        for j in range(i + 1, 12):
            vi = inner_verts_list[i]
            vj = inner_verts_list[j]

            # Get all edge normals from both hexagons
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

    # Check all inner vertices are inside outer hexagon
    for i in range(12):
        verts = inner_verts_list[i]
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


def _smooth_min_s_for_rotation(vertices, rot_deg, beta=20.0):
    """
    Smooth surrogate for the minimum enclosing hexagon side length using log-sum-exp.
    Returns (smooth_s, center_estimate) where center_estimate is approximate.
    """
    rot = np.radians(rot_deg)
    n_pts = len(vertices)
    
    # Precompute the 6 normal directions
    normals = np.zeros((6, 2))
    for i in range(6):
        angle = rot + np.radians(60 * i + 30)
        normals[i] = (np.cos(angle), np.sin(angle))
    
    # For each normal direction, compute the max projection
    # The constraint is: n·(v - c) <= s * sqrt(3)/2 for all v, n
    # So s >= max_v (n·v - n·c) / (sqrt(3)/2)
    # To minimize s, we need to choose c optimally
    
    # For a fixed c, s >= max over all (n, v) of (n·v - n·c) / (sqrt(3)/2)
    # = max over all (n, v) of (n·(v - c)) * 2/sqrt(3)
    
    # Smooth approximation: use log-sum-exp to approximate the max
    # But we also need to optimize over c
    
    # Strategy: for given c, compute smooth s, then use gradient-based optimization on c
    # For now, return a rough estimate using centroid
    c_est = np.mean(vertices, axis=0)
    
    # Compute all n·(v - c) values
    # Shape: (6, n_pts)
    diffs = vertices[None, :, :] - c_est[None, None, :]  # (1, n_pts, 2)
    dots = np.einsum('ij,njk->nik', normals, diffs)  # (6, n_pts, 1) -> (6, n_pts)
    dots = dots.squeeze(-1)  # (6, n_pts)
    
    # s >= max(dots) * 2/sqrt(3)
    # Smooth: use log-sum-exp
    max_dots = np.max(dots)
    log_sum_exp = max_dots + np.log(np.sum(np.exp(dots - max_dots)))
    smooth_s = log_sum_exp * 2 / np.sqrt(3)
    
    return smooth_s, c_est


def _exact_min_s_with_center(vertices, rot_deg, center):
    """
    Given vertices and a center, compute the exact minimum s needed.
    s = max over all (n, v) of n·(v - center) * 2/sqrt(3)
    """
    rot = np.radians(rot_deg)
    normals = np.zeros((6, 2))
    for i in range(6):
        angle = rot + np.radians(60 * i + 30)
        normals[i] = (np.cos(angle), np.sin(angle))
    
    # Compute all n·(v - center)
    diffs = vertices - center  # (n_pts, 2)
    dots = np.einsum('ij,nj->ni', normals, diffs)  # (6, n_pts)
    
    s = np.max(dots) * 2 / np.sqrt(3)
    return max(0, s)


def _optimize_center_for_min_s(vertices, rot_deg, x0=None, max_iter=200):
    """
    Optimize the center to minimize the exact minimum s.
    Uses Nelder-Mead on the center coordinates.
    """
    rot = np.radians(rot_deg)
    normals = np.zeros((6, 2))
    for i in range(6):
        angle = rot + np.radians(60 * i + 30)
        normals[i] = (np.cos(angle), np.sin(angle))
    
    if x0 is None:
        x0 = np.mean(vertices, axis=0)
    
    def objective(c):
        diffs = vertices - c
        dots = np.einsum('ij,nj->ni', normals, diffs)
        return np.max(dots) * 2 / np.sqrt(3)
    
    res = minimize(objective, x0, method='Nelder-Mead',
                  options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12})
    
    return res.fun, res.x


def _fast_min_s(vertices, normals, center):
    """Compute exact min s given vertices, precomputed normals, and center."""
    diffs = vertices - center
    dots = np.einsum('ij,nj->ni', normals, diffs)
    return np.max(dots) * 2 / np.sqrt(3)


def _smooth_s_surrogate(vertices, rot_deg, beta=15.0):
    """
    Smooth surrogate for min enclosing hexagon side length.
    Returns (smooth_s, best_center_estimate).
    Uses log-sum-exp to approximate max, optimizes center via few Newton steps.
    """
    rot = np.radians(rot_deg)
    normals = np.zeros((6, 2))
    for i in range(6):
        angle = rot + np.radians(60 * i + 30)
        normals[i] = (np.cos(angle), np.sin(angle))

    n_pts = len(vertices)

    # For a given center c, the exact s = max_{n,v} n·(v-c) * 2/sqrt(3)
    # Smooth: (1/beta) * log(sum exp(beta * n·(v-c))) * 2/sqrt(3)
    # But we also need to optimize c.
    # For fixed beta, the smooth objective in c is convex, so we can use gradient descent.

    c = np.mean(vertices, axis=0)

    for _ in range(30):
        diffs = vertices - c  # (n_pts, 2)
        dots = np.einsum('ij,nj->ni', normals, diffs)  # (6, n_pts)
        max_d = np.max(dots)
        exp_d = np.exp(beta * (dots - max_d))
        # Gradient of smooth max w.r.t. c: -sum(exp_d * n) / sum(exp_d)
        weights = exp_d / np.sum(exp_d)  # (6, n_pts)
        grad = -np.einsum('ni,ij->j', weights, normals)  # gradient of smooth_max w.r.t. c
        # smooth_s = smooth_max * 2/sqrt(3)
        # d(smooth_s)/dc = (2/sqrt(3)) * grad
        # We want to minimize, so c_new = c - lr * grad
        smooth_max = max_d + np.log(np.sum(exp_d)) / beta
        # Simple gradient step
        lr = 0.05
        c = c - lr * grad

    # Compute final smooth value
    diffs = vertices - c
    dots = np.einsum('ij,nj->ni', normals, diffs)
    max_d = np.max(dots)
    exp_d = np.exp(beta * (dots - max_d))
    smooth_max = max_d + np.log(np.sum(exp_d)) / beta
    smooth_s = smooth_max * 2 / np.sqrt(3)

    return smooth_s, c


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    """
    start_time = time.time()
    deadline = start_time + 610.0  # safety margin before 630s kill

    sqrt3 = np.sqrt(3)

    # Define 12 hexagon centers in a hexagonal lattice pattern
    eps = 1e-3
    scale = 1.0 + eps

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

    # Phase 1: Coarse search using smooth surrogate (fast)
    n_theta = 61
    n_phi = 61
    theta_vals = np.linspace(0, 60, n_theta)
    phi_vals = np.linspace(0, 60, n_phi)

    # Precompute cluster vertices for each theta
    cluster_verts_cache = {}
    for theta_deg in theta_vals:
        cluster_verts_cache[theta_deg] = _build_cluster(centers_base, theta_deg)

    for theta_deg in theta_vals:
        if time.time() > deadline - 100:
            break
        cv = cluster_verts_cache[theta_deg]
        for phi_deg in phi_vals:
            if time.time() > deadline - 100:
                break
            s_smooth, c_est = _smooth_s_surrogate(cv, phi_deg, beta=15.0)
            if s_smooth < best_s * 1.01:  # Only refine promising candidates
                s, center = _compute_min_s_for_rotation(cv, phi_deg)
                if s < best_s:
                    inner_hex_data = np.column_stack([centers_base, np.full(12, theta_deg)])
                    outer_hex_data = np.array([center[0], center[1], phi_deg])
                    if _validate_packing(inner_hex_data, outer_hex_data, s):
                        best_s = s
                        best_inner = inner_hex_data.copy()
                        best_outer = outer_hex_data.copy()

    # Phase 2: Fine grid search around best (theta, phi) with exact LP
    if best_inner is not None and time.time() < deadline - 200:
        best_theta = best_inner[0, 2]
        best_phi = best_outer[2]

        theta_fine = np.linspace(best_theta - 3, best_theta + 3, 61)
        phi_fine = np.linspace(best_phi - 3, best_phi + 3, 61)

        for theta_deg in theta_fine:
            if time.time() > deadline - 100:
                break
            cluster_verts = _build_cluster(centers_base, theta_deg)
            for phi_deg in phi_fine:
                if time.time() > deadline - 100:
                    break
                s, center = _compute_min_s_for_rotation(cluster_verts, phi_deg)
                if s < best_s:
                    inner_hex_data = np.column_stack([centers_base, np.full(12, theta_deg)])
                    outer_hex_data = np.array([center[0], center[1], phi_deg])
                    if _validate_packing(inner_hex_data, outer_hex_data, s):
                        best_s = s
                        best_inner = inner_hex_data.copy()
                        best_outer = outer_hex_data.copy()

    # Phase 3: Alternative cluster configurations
    if time.time() < deadline - 250:
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
            if time.time() > deadline - 120:
                break
            cluster_verts = _build_cluster(alt_centers, theta_deg)
            for phi_deg in np.linspace(0, 60, 31):
                if time.time() > deadline - 120:
                    break
                s, center = _compute_min_s_for_rotation(cluster_verts, phi_deg)
                if s < best_s:
                    inner_hex_data = np.column_stack([alt_centers, np.full(12, theta_deg)])
                    outer_hex_data = np.array([center[0], center[1], phi_deg])
                    if _validate_packing(inner_hex_data, outer_hex_data, s):
                        best_s = s
                        best_inner = inner_hex_data.copy()
                        best_outer = outer_hex_data.copy()

    # Phase 4: Compressed lattice
    if time.time() < deadline - 200:
        for compress in [0.998, 0.995, 0.99]:
            if time.time() > deadline - 100:
                break
            centers_comp = centers_base * (scale * compress)
            for theta_deg in np.linspace(0, 60, 31):
                if time.time() > deadline - 100:
                    break
                cluster_verts = _build_cluster(centers_comp, theta_deg)
                for phi_deg in np.linspace(0, 60, 31):
                    if time.time() > deadline - 100:
                        break
                    s, center = _compute_min_s_for_rotation(cluster_verts, phi_deg)
                    if s < best_s:
                        inner_hex_data = np.column_stack([centers_comp, np.full(12, theta_deg)])
                        outer_hex_data = np.array([center[0], center[1], phi_deg])
                        if _validate_packing(inner_hex_data, outer_hex_data, s):
                            best_s = s
                            best_inner = inner_hex_data.copy()
                            best_outer = outer_hex_data.copy()

    # Phase 5: High-precision polish using Nelder-Mead on (theta, phi) with exact LP
    if best_inner is not None and time.time() < deadline - 150:
        best_theta = best_inner[0, 2]
        best_phi = best_outer[2]

        # Ultra-fine grid
        for radius in [1.0, 0.5, 0.25, 0.1]:
            if time.time() > deadline - 80:
                break
            theta_ultra = np.linspace(best_theta - radius, best_theta + radius, 41)
            phi_ultra = np.linspace(best_phi - radius, best_phi + radius, 41)
            for theta_deg in theta_ultra:
                if time.time() > deadline - 80:
                    break
                cluster_verts = _build_cluster(centers_base, theta_deg)
                for phi_deg in phi_ultra:
                    if time.time() > deadline - 80:
                        break
                    s, center = _compute_min_s_for_rotation(cluster_verts, phi_deg)
                    if s < best_s:
                        inner_hex_data = np.column_stack([centers_base, np.full(12, theta_deg)])
                        outer_hex_data = np.array([center[0], center[1], phi_deg])
                        if _validate_packing(inner_hex_data, outer_hex_data, s):
                            best_s = s
                            best_inner = inner_hex_data.copy()
                            best_outer = outer_hex_data.copy()

        # Nelder-Mead on (theta, phi) using LP as objective
        if time.time() < deadline - 60:
            def objective(params):
                theta_d, phi_d = params
                cluster_v = _build_cluster(centers_base, theta_d)
                s_val, _ = _compute_min_s_for_rotation(cluster_v, phi_d)
                return s_val

            x0 = [best_theta, best_phi]
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})

            if res.fun < best_s:
                theta_opt, phi_opt = res.x
                cluster_verts = _build_cluster(centers_base, theta_opt)
                s_opt, center_opt = _compute_min_s_for_rotation(cluster_verts, phi_opt)
                inner_hex_data = np.column_stack([centers_base, np.full(12, theta_opt)])
                outer_hex_data = np.array([center_opt[0], center_opt[1], phi_opt])
                if _validate_packing(inner_hex_data, outer_hex_data, s_opt):
                    best_s = s_opt
                    best_inner = inner_hex_data.copy()
                    best_outer = outer_hex_data.copy()

    # Phase 6: Final high-precision polish with center optimization
    if best_inner is not None and time.time() < deadline - 30:
        best_theta = best_inner[0, 2]
        best_phi = best_outer[2]
        cluster_verts = _build_cluster(centers_base, best_theta)

        # Optimize center more precisely
        s_opt, center_opt = _optimize_center_for_min_s(cluster_verts, best_phi, max_iter=500)

        if s_opt < best_s:
            inner_hex_data = np.column_stack([centers_base, np.full(12, best_theta)])
            outer_hex_data = np.array([center_opt[0], center_opt[1], best_phi])
            if _validate_packing(inner_hex_data, outer_hex_data, s_opt):
                best_s = s_opt
                best_inner = inner_hex_data.copy()
                best_outer = outer_hex_data.copy()

    # If no valid packing was found, use a default
    if best_inner is None:
        centers = centers_base
        inner_hex_data = np.column_stack([centers, np.zeros(12)])
        s, center = _compute_min_s_for_rotation(_build_cluster(centers, 0), 0)
        outer_hex_data = np.array([center[0], center[1], 0])
        return inner_hex_data, outer_hex_data, s

    return best_inner, best_outer, best_s


# EVOLVE-BLOCK-END

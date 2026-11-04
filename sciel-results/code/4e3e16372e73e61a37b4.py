import numpy as np
import time
import scipy.optimize

N = 11
SQRT3 = np.sqrt(3)

def _base_hex_verts():
    return np.array([
        [1.0, 0.0],
        [0.5, SQRT3/2],
        [-0.5, SQRT3/2],
        [-1.0, 0.0],
        [-0.5, -SQRT3/2],
        [0.5, -SQRT3/2]
    ])

def _outer_normals(phi_deg):
    phi_rad = np.deg2rad(phi_deg)
    return np.array([
        [np.cos(phi_rad + k * np.pi / 3), np.sin(phi_rad + k * np.pi / 3)]
        for k in range(6)
    ])

def _compute_side(flat_verts, cluster_angle_deg, phi_deg):
    """Fast analytical side-length: rotate all verts, project onto 6 outer normals."""
    a_rad = np.deg2rad(cluster_angle_deg)
    ca, sa = np.cos(a_rad), np.sin(a_rad)
    Rm = np.array([[ca, -sa], [sa, ca]])
    rot = flat_verts @ Rm.T
    normals = _outer_normals(phi_deg)
    projs = rot @ normals.T
    max_proj = np.max(projs)
    return 2.0 * max_proj / SQRT3

def _fallback_solution():
    """Return a valid rigid-cluster solution as safety net."""
    d = SQRT3
    u = np.array([1.5, d/2])
    v = np.array([0.0, d])
    lattice_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, -1), (2, -1), (-2, 1)
    ]
    cluster_centers = np.array([i * u + j * v for i, j in lattice_coords])
    cluster_centers *= 1.002
    base_verts = _base_hex_verts()
    all_base_verts = np.tile(base_verts, (N, 1, 1))
    cluster_verts = all_base_verts + cluster_centers[:, np.newaxis, :]
    flat_verts = cluster_verts.reshape(-1, 2)
    
    # Quick grid to find a reasonable angle
    best_a, best_p, best_s = 0.0, 0.0, float('inf')
    for a in np.linspace(0, 60, 121, endpoint=False):
        for p in np.linspace(0, 60, 61, endpoint=False):
            s = _compute_side(flat_verts, a, p)
            if s < best_s:
                best_s, best_a, best_p = s, a, p
    
    angle_rad = np.deg2rad(best_a)
    ca, sa = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[ca, -sa], [sa, ca]])
    rotated_centers = cluster_centers @ R.T
    inner = []
    for i in range(N):
        inner.append([float(rotated_centers[i, 0]), float(rotated_centers[i, 1]), float(best_a)])
    return inner, (0.0, 0.0), float(best_s) + 1e-3, float(best_p)

def construct():
    start_time = time.time()
    deadline = 145.0
    
    d = SQRT3
    u = np.array([1.5, d/2])
    v = np.array([0.0, d])
    lattice_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, -1), (2, -1), (-2, 1)
    ]
    cluster_centers = np.array([i * u + j * v for i, j in lattice_coords])
    cluster_centers *= 1.002
    
    base_verts = _base_hex_verts()
    all_base_verts = np.tile(base_verts, (N, 1, 1))
    cluster_verts = all_base_verts + cluster_centers[:, np.newaxis, :]
    flat_verts = cluster_verts.reshape(-1, 2)
    
    best_angle = 0.0
    best_phi = 0.0
    best_side = float('inf')
    
    # Phase 1: Coarse 2D grid over (cluster_angle, outer_phi)
    angles_coarse = np.linspace(0, 60, 121, endpoint=False)
    phis_coarse = np.linspace(0, 60, 61, endpoint=False)
    for a in angles_coarse:
        for p in phis_coarse:
            s = _compute_side(flat_verts, a, p)
            if s < best_side:
                best_side = s
                best_angle = a
                best_phi = p
    
    if time.time() > deadline - 30:
        return _fallback_solution()
    
    # Phase 2: Fine grid around best
    fine_a = np.linspace(best_angle - 2.0, best_angle + 2.0, 201)
    fine_p = np.linspace(best_phi - 2.0, best_phi + 2.0, 101)
    for a in fine_a:
        for p in fine_p:
            s = _compute_side(flat_verts, a, p)
            if s < best_side:
                best_side = s
                best_angle = a
                best_phi = p
    
    if time.time() > deadline - 20:
        return _fallback_solution()
    
    # Phase 3: Multi-start Powell optimization
    def obj(theta):
        return _compute_side(flat_verts, theta[0], theta[1])
    
    # Multiple starting points: best found + diverse random
    starts = [np.array([best_angle, best_phi])]
    rng = np.random.RandomState(42)
    for _ in range(8):
        sa = rng.uniform(0, 60)
        sp = rng.uniform(0, 60)
        starts.append(np.array([sa, sp]))
    
    for start in starts:
        if time.time() > deadline - 10:
            break
        try:
            res = scipy.optimize.minimize(
                obj, start, method='Powell',
                options={'maxiter': 200, 'xtol': 1e-8, 'ftol': 1e-10}
            )
            s = obj(res.x)
            if s < best_side:
                best_side = s
                best_angle = float(res.x[0]) % 60.0
                best_phi = float(res.x[1]) % 60.0
        except Exception:
            pass
    
    if time.time() > deadline - 5:
        return _fallback_solution()
    
    # Phase 4: Nelder-Mead polish
    try:
        res_nm = scipy.optimize.minimize(
            obj, np.array([best_angle, best_phi]),
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        s = obj(res_nm.x)
        if s < best_side:
            best_side = s
            best_angle = float(res_nm.x[0]) % 60.0
            best_phi = float(res_nm.x[1]) % 60.0
    except Exception:
        pass
    
    # Phase 5: Coordinate descent with fine steps
    if time.time() < deadline - 2:
        # Refine cluster angle
        def obj_a(a):
            return _compute_side(flat_verts, a, best_phi)
        try:
            res_a = scipy.optimize.minimize_scalar(
                obj_a, bounds=(best_angle - 0.05, best_angle + 0.05),
                method='bounded', options={'xatol': 1e-12}
            )
            s = obj_a(res_a.x)
            if s < best_side:
                best_side = s
                best_angle = float(res_a.x)
        except Exception:
            pass
        
        # Refine outer phi
        def obj_p(p):
            return _compute_side(flat_verts, best_angle, p)
        try:
            res_p = scipy.optimize.minimize_scalar(
                obj_p, bounds=(best_phi - 0.05, best_phi + 0.05),
                method='bounded', options={'xatol': 1e-12}
            )
            s = obj_p(res_p.x)
            if s < best_side:
                best_side = s
                best_phi = float(res_p.x)
        except Exception:
            pass
    
    # Build inner hexagon positions
    angle_rad = np.deg2rad(best_angle)
    ca, sa = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[ca, -sa], [sa, ca]])
    rotated_centers = cluster_centers @ R.T
    
    inner = []
    for i in range(N):
        inner.append([float(rotated_centers[i, 0]), float(rotated_centers[i, 1]), float(best_angle)])
    
    # Recalculate side length precisely for the final configuration to avoid numerical drift
    # Reconstruct vertices
    base_verts = _base_hex_verts()
    all_base_verts = np.tile(base_verts, (N, 1, 1))
    # Rotate base verts by best_angle
    a_rad = np.deg2rad(best_angle)
    ca_v, sa_v = np.cos(a_rad), np.sin(a_rad)
    R_v = np.array([[ca_v, -sa_v], [sa_v, ca_v]])
    rot_base_verts = all_base_verts @ R_v.T
    # Add rotated centers
    final_verts = rot_base_verts + rotated_centers[:, np.newaxis, :]
    flat_final_verts = final_verts.reshape(-1, 2)
    
    # Compute required side
    normals = _outer_normals(best_phi)
    projs = flat_final_verts @ normals.T
    max_proj = np.max(projs)
    precise_side = 2.0 * max_proj / SQRT3
    
    # Use the larger of the optimized side and the precise side, plus a safety margin
    final_side = max(float(best_side), float(precise_side)) + 1e-3
    
    # Verify containment and adjust if necessary
    # Check all vertices against the final side length
    # Condition: dot(v, n) <= final_side * SQRT3 / 2
    limit = final_side * SQRT3 / 2
    current_max_proj = np.max(projs)
    if current_max_proj > limit:
        # Increase side length to fit
        final_side = 2.0 * current_max_proj / SQRT3 + 1e-3
    
    return inner, (0.0, 0.0), float(final_side), float(best_phi)

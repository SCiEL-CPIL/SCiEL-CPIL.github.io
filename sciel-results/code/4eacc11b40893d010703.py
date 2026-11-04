import numpy as np
import time
import scipy.optimize

N = 11

def hex_vertices(x, y, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    verts = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    verts = verts @ R.T
    verts += np.array([x, y])
    return verts

def construct():
    start_time = time.time()
    deadline = 148.0

    d = np.sqrt(3)

    # 11-hexagonal rigid cluster based on triangular lattice
    u = np.array([1.5, d/2])
    v = np.array([0.0, d])

    lattice_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, -1), (2, -1), (-2, 1)
    ]

    cluster_centers = np.array([i * u + j * v for i, j in lattice_coords])
    cluster_centers *= 1.002

    base_verts = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])

    all_base_verts = np.tile(base_verts, (N, 1, 1))
    cluster_verts = all_base_verts + cluster_centers[:, np.newaxis, :]
    flat_verts = cluster_verts.reshape(-1, 2)

    # Precompute normal directions for phi=0 and phi=30
    normals_0 = np.array([
        [np.cos(k * np.pi / 3), np.sin(k * np.pi / 3)]
        for k in range(6)
    ])
    normals_30 = np.array([
        [np.cos(np.pi/6 + k * np.pi / 3), np.sin(np.pi/6 + k * np.pi / 3)]
        for k in range(6)
    ])

    # Precompute projections of flat_verts onto each normal set
    proj_0 = flat_verts @ normals_0.T  # (66, 6)
    proj_30 = flat_verts @ normals_30.T  # (66, 6)

    def compute_side_fast(angle_deg):
        """Fast vectorized side computation for given cluster angle, trying phi=0 and phi=30."""
        a_rad = np.deg2rad(angle_deg)
        ca, sa = np.cos(a_rad), np.sin(a_rad)
        Rm = np.array([[ca, -sa], [sa, ca]])
        rot = flat_verts @ Rm.T
        p0 = rot @ normals_0.T
        p30 = rot @ normals_30.T
        L0 = 2.0 * np.max(p0) / np.sqrt(3)
        L30 = 2.0 * np.max(p30) / np.sqrt(3)
        return min(L0, L30)

    def compute_side_with_phi(angle_cluster_deg, phi_deg):
        a_rad = np.deg2rad(angle_cluster_deg)
        ca, sa = np.cos(a_rad), np.sin(a_rad)
        Rm = np.array([[ca, -sa], [sa, ca]])
        rot = flat_verts @ Rm.T
        p_rad = np.deg2rad(phi_deg)
        normals = np.array([
            [np.cos(p_rad + k * np.pi / 3), np.sin(p_rad + k * np.pi / 3)]
            for k in range(6)
        ])
        projs = rot @ normals.T
        return 2.0 * np.max(projs) / np.sqrt(3)

    # Phase 1: Vectorized coarse grid over cluster angle (0 to 60)
    angles_coarse = np.linspace(0, 60, 3001, endpoint=False)
    a_rad_arr = np.deg2rad(angles_coarse)
    ca_arr = np.cos(a_rad_arr)
    sa_arr = np.sin(a_rad_arr)

    # Vectorized rotation: flat_verts (66,2) rotated by each angle
    # R_k = [[ca_k, -sa_k], [sa_k, ca_k]]
    # rot_k = flat_verts @ R_k.T = [[x*ca + y*sa, -x*sa + y*ca]]
    x_fv = flat_verts[:, 0:1]  # (66, 1)
    y_fv = flat_verts[:, 1:2]  # (66, 1)

    # For each angle k, rotated coords:
    # rx = x*ca_k + y*sa_k, ry = -x*sa_k + y*ca_k
    # Then projection onto normal j: rx * nx_j + ry * ny_j
    # = (x*ca + y*sa)*nx + (-x*sa + y*ca)*ny
    # = x*(ca*nx - sa*ny) + y*(sa*nx + ca*ny)

    # normals_0: (6,2), normals_30: (6,2)
    # For each angle k and normal j:
    # proj = x * (ca_k * nx_j - sa_k * ny_j) + y * (sa_k * nx_j + ca_k * ny_j)

    # Compute for phi=0
    nx0 = normals_0[:, 0]  # (6,)
    ny0 = normals_0[:, 1]  # (6,)
    # coeff_x_kj = ca_k * nx_j - sa_k * ny_j  -> (3001, 6)
    coeff_x_0 = ca_arr[:, None] * nx0[None, :] - sa_arr[:, None] * ny0[None, :]
    coeff_y_0 = sa_arr[:, None] * nx0[None, :] + ca_arr[:, None] * ny0[None, :]
    # proj_kj = sum_i (x_i * coeff_x_kj + y_i * coeff_y_kj) = x_fv.T @ ... no
    # proj_kj = x_fv.T @ (coeff_x_0.T) ... actually:
    # For fixed k: proj_kj = sum_i x_i * coeff_x_kj + y_i * coeff_y_kj
    # = x_fv[:,0] @ coeff_x_0[k,:] + y_fv[:,0] @ coeff_y_0[k,:]
    # = dot(x_fv_flat, coeff_x_0[k]) + dot(y_fv_flat, coeff_y_0[k])
    x_flat = flat_verts[:, 0]  # (66,)
    y_flat = flat_verts[:, 1]  # (66,)
    # proj_0_kj = x_flat @ coeff_x_0[k,:] + y_flat @ coeff_y_0[k,:]
    # = (x_flat[None,:] @ coeff_x_0.T + y_flat[None,:] @ coeff_y_0.T) -> no
    # Actually: for each k, j: proj = sum_i (x_i * cx_kj + y_i * cy_kj)
    # = (x_i @ cx_kj) + (y_i @ cy_kj) where @ is dot product
    # Vectorized: proj_0 = x_flat[None,None,:] * coeff_x_0[:,:,None] + y_flat[None,None,:] * coeff_y_0[:,:,None]
    # Then max over i (axis=2), then over j (axis=1)
    # That's 3001*6*66 = ~1.2M ops, fine.

    # More memory-efficient: compute max over i first for each (k,j)
    # proj_kj = dot(x_flat, coeff_x_0[k,j]) + dot(y_flat, coeff_y_0[k,j])
    # Use: proj_0 = np.einsum('ij,i->j', ...) no, let's just do:
    # For each k: max_j (max_i (x_i*cx_kj + y_i*cy_kj))
    # = max_j (x_flat @ cx_kj + y_flat @ cy_kj)

    # Efficient: compute all projections as (3001, 6)
    # proj_0[k,j] = x_flat @ coeff_x_0[k,j] + y_flat @ coeff_y_0[k,j]
    # = (x_flat * coeff_x_0).sum(axis=1) + (y_flat * coeff_y_0).sum(axis=1) -- no, x_flat is (66,) not (3001,66)
    # Let me reshape: coeff_x_0 is (3001, 6). x_flat is (66,).
    # I need for each k,j: sum_i x_i * coeff_x_0[k,j]. This is not a standard matmul.
    # Alternative: think of it as: for each k, the projection vector over j is:
    # p_kj = sum_i x_i * (ca_k*nx_j - sa_k*ny_j) + y_i * (sa_k*nx_j + ca_k*ny_j)
    # = ca_k * sum_i (x_i*nx_j + y_i*ny_j) + sa_k * sum_i (-x_i*ny_j + y_i*nx_j)
    # = ca_k * A_j + sa_k * B_j
    # where A_j = sum_i (x_i*nx_j + y_i*ny_j) = sum_i dot(vert_i, normal_j) = sum of projections of all verts onto normal_j
    # and B_j = sum_i (-x_i*ny_j + y_i*nx_j) = sum_i dot(vert_i, perp(normal_j)) where perp rotates 90 deg

    # Wait, that's the SUM not the MAX. We need MAX over i, not sum.
    # So this trick doesn't work directly. Let me just do the direct computation.

    # Direct: for each k, compute rot, then project. 3001 iterations of (66x2)@(2x2) then (66x2)@(2x6)
    # Each iteration: ~66*2*2 + 66*2*6 = 264 + 792 = ~1056 flops. Total: ~3M flops. Very fast.
    # But Python loop overhead... let's vectorize differently.

    # Actually, let me just use a moderate grid and loop, it'll be fast enough.
    sides_coarse = np.empty(len(angles_coarse))
    for idx, a in enumerate(angles_coarse):
        sides_coarse[idx] = compute_side_fast(a)
        if idx % 500 == 499 and time.time() > deadline - 30:
            angles_coarse = angles_coarse[:idx+1]
            sides_coarse = sides_coarse[:idx+1]
            break

    best_idx = np.argmin(sides_coarse)
    best_angle = angles_coarse[best_idx]
    best_side = sides_coarse[best_idx]

    # Phase 2: Fine grid around best cluster angle
    if time.time() < deadline - 20:
        fine_range = 0.5
        angles_fine = np.linspace(best_angle - fine_range, best_angle + fine_range, 2001)
        sides_fine = np.empty(len(angles_fine))
        for idx, a in enumerate(angles_fine):
            sides_fine[idx] = compute_side_fast(a)
        best_idx_fine = np.argmin(sides_fine)
        best_angle = angles_fine[best_idx_fine]
        best_side = sides_fine[best_idx_fine]

    # Phase 3: Optimize outer hexagon angle phi for the best cluster angle
    if time.time() < deadline - 15:
        phi_values = np.linspace(0, 60, 1201, endpoint=False)
        best_phi_side = float('inf')
        best_phi = 0.0
        for phi in phi_values:
            s = compute_side_with_phi(best_angle, phi)
            if s < best_phi_side:
                best_phi_side = s
                best_phi = phi

        if best_phi_side < best_side:
            best_side = best_phi_side
            # Refine phi
            phi_fine = np.linspace(best_phi - 0.25, best_phi + 0.25, 2001)
            for phi in phi_fine:
                s = compute_side_with_phi(best_angle, phi)
                if s < best_side:
                    best_side = s
                    best_phi = phi

    # Phase 4: Joint optimization using smooth surrogate (log-sum-exp) + Nelder-Mead
    if time.time() < deadline - 8:
        def smooth_obj(theta, beta=200.0):
            """Smooth surrogate using log-sum-exp for the max."""
            a_rad = np.deg2rad(theta[0])
            ca, sa = np.cos(a_rad), np.sin(a_rad)
            Rm = np.array([[ca, -sa], [sa, ca]])
            rot = flat_verts @ Rm.T

            p_rad = np.deg2rad(theta[1])
            normals = np.array([
                [np.cos(p_rad + k * np.pi / 3), np.sin(p_rad + k * np.pi / 3)]
                for k in range(6)
            ])
            projs = rot @ normals.T  # (66, 6)

            # For each normal direction j, we need max over i of projs[i,j]
            # Use log-sum-exp: max_j (logsumexp_i(projs[i,j] / beta) * beta)
            # Actually we want max over all (i,j) of projs[i,j]
            # = max_j max_i projs[i,j]
            # Smooth: logsumexp over all 66*6 values
            all_projs = projs.ravel()  # (396,)
            m = np.max(all_projs)
            lse = m + np.log(np.sum(np.exp((all_projs - m) / beta))) * beta
            return 2.0 * lse / np.sqrt(3)

        def exact_obj(theta):
            a_rad = np.deg2rad(theta[0])
            ca, sa = np.cos(a_rad), np.sin(a_rad)
            Rm = np.array([[ca, -sa], [sa, ca]])
            rot = flat_verts @ Rm.T
            p_rad = np.deg2rad(theta[1])
            normals = np.array([
                [np.cos(p_rad + k * np.pi / 3), np.sin(p_rad + k * np.pi / 3)]
                for k in range(6)
            ])
            projs = rot @ normals.T
            return 2.0 * np.max(projs) / np.sqrt(3)

        try:
            # First optimize with smooth surrogate
            res_smooth = scipy.optimize.minimize(
                smooth_obj,
                np.array([best_angle, best_phi]),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            s_smooth = exact_obj(res_smooth.x)
            if s_smooth < best_side:
                best_angle = float(res_smooth.x[0]) % 60.0
                best_phi = float(res_smooth.x[1]) % 60.0
                best_side = float(s_smooth)
        except Exception:
            pass

        # Phase 4b: Nelder-Mead with exact (nonsmooth) objective
        if time.time() < deadline - 5:
            try:
                res_exact = scipy.optimize.minimize(
                    exact_obj,
                    np.array([best_angle, best_phi]),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                s_exact = exact_obj(res_exact.x)
                if s_exact < best_side:
                    best_angle = float(res_exact.x[0]) % 60.0
                    best_phi = float(res_exact.x[1]) % 60.0
                    best_side = float(s_exact)
            except Exception:
                pass

    # Phase 5: High-precision final polish with very tight tolerances
    if time.time() < deadline - 2:
        try:
            res_polish = scipy.optimize.minimize(
                exact_obj,
                np.array([best_angle, best_phi]),
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            s_polish = exact_obj(res_polish.x)
            if s_polish < best_side:
                best_angle = float(res_polish.x[0]) % 60.0
                best_phi = float(res_polish.x[1]) % 60.0
                best_side = float(s_polish)
        except Exception:
            pass

        # Scalar refinement on each angle independently
        if time.time() < deadline - 1:
            def obj_a(a):
                return exact_obj(np.array([a, best_phi]))

            try:
                res_a = scipy.optimize.minimize_scalar(
                    obj_a, bounds=(best_angle - 0.01, best_angle + 0.01),
                    method='bounded', options={'xatol': 1e-12}
                )
                s_a = obj_a(res_a.x)
                if s_a < best_side:
                    best_angle = float(res_a.x)
                    best_side = float(s_a)
            except Exception:
                pass

            def obj_p(p):
                return exact_obj(np.array([best_angle, p]))

            try:
                res_p = scipy.optimize.minimize_scalar(
                    obj_p, bounds=(best_phi - 0.01, best_phi + 0.01),
                    method='bounded', options={'xatol': 1e-12}
                )
                s_p = obj_p(res_p.x)
                if s_p < best_side:
                    best_phi = float(res_p.x)
                    best_side = float(s_p)
            except Exception:
                pass

    # Build the inner hexagon positions
    angle_rad = np.deg2rad(best_angle)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])

    rotated_centers = cluster_centers @ R.T

    inner = []
    for i in range(N):
        x = float(rotated_centers[i, 0])
        y = float(rotated_centers[i, 1])
        ang = float(best_angle)
        inner.append([x, y, ang])

    final_side = float(best_side) + 1e-10

    return inner, (0.0, 0.0), final_side, float(best_phi)

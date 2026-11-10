# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, s, angle_deg):
    angles = np.radians(angle_deg) + np.radians(60) * np.arange(6)
    return np.column_stack([
        cx + s * np.cos(angles),
        cy + s * np.sin(angles)
    ])

def _all_vertices(positions, angles, side=1.0):
    n = len(positions)
    verts = np.zeros((n, 6, 2))
    base_angles = np.radians(angles)[:, None] + np.radians(60) * np.arange(6)[None, :]
    verts[:, :, 0] = positions[:, 0:1] + side * np.cos(base_angles)
    verts[:, :, 1] = positions[:, 1:2] + side * np.sin(base_angles)
    return verts

def _check_separation_fast(positions, angles, tol=1e-6):
    n = len(positions)
    all_verts = _all_vertices(positions, angles)
    hex_normals = np.zeros((n, 6, 2))
    base = np.radians(angles)[:, None] + np.radians(30) + np.radians(60) * np.arange(6)[None, :]
    hex_normals[:, :, 0] = np.cos(base)
    hex_normals[:, :, 1] = np.sin(base)
    for i in range(n):
        for j in range(i + 1, n):
            vi = all_verts[i]
            vj = all_verts[j]
            normals = np.vstack([hex_normals[i], hex_normals[j]])
            proj_i = vi @ normals.T
            proj_j = vj @ normals.T
            max_i = np.max(proj_i, axis=0)
            min_i = np.min(proj_i, axis=0)
            max_j = np.max(proj_j, axis=0)
            min_j = np.min(proj_j, axis=0)
            if np.any(max_i < min_j - tol) or np.any(max_j < min_i - tol):
                continue
            return False
    return True

def _min_enclosing_side(positions, angles):
    all_verts = _all_vertices(positions, angles).reshape(-1, 2)
    normals = np.array([[np.cos(np.radians(30 + 60 * i)), np.sin(np.radians(30 + 60 * i))] for i in range(6)])
    apothem_factor = np.sqrt(3) / 2
    projs = all_verts @ normals.T
    max_proj = np.max(projs)
    return max_proj / apothem_factor

def _min_enclosing_side_smooth(positions, angles, beta=500.0):
    all_verts = _all_vertices(positions, angles).reshape(-1, 2)
    normals = np.array([[np.cos(np.radians(30 + 60 * i)), np.sin(np.radians(30 + 60 * i))] for i in range(6)])
    apothem_factor = np.sqrt(3) / 2
    projs = all_verts @ normals.T  # (N, 6)
    # smooth max per direction via log-sum-exp
    shifted = projs / beta
    m = np.max(shifted, axis=0)
    smax = m + np.log(np.mean(np.exp(shifted - m), axis=0))
    return np.max(smax) * beta / apothem_factor

def _check_containment_fast(positions, angles, side, tol=-1e-6):
    all_verts = _all_vertices(positions, angles).reshape(-1, 2)
    outer_verts = _hex_vertices(0.0, 0.0, side, 0.0)
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        e = p2 - p1
        v = all_verts - p1
        cross = e[0] * v[:, 1] - e[1] * v[:, 0]
        if np.min(cross) < tol:
            return False
    return True

def _build_symmetric_positions(R, V, theta_deg):
    """1 center, 6 ring at radius R (angles 30+60k+theta), 4 valley at radius V (angles 30+120k+theta)."""
    positions = np.zeros((11, 2))
    angles = np.zeros(11)
    positions[0] = [0.0, 0.0]
    th = np.radians(theta_deg)
    for k in range(6):
        a = np.radians(30 + 60 * k) + th
        positions[k + 1] = [R * np.cos(a), R * np.sin(a)]
    for k in range(4):
        a = np.radians(30 + 120 * k) + th
        positions[7 + k] = [V * np.cos(a), V * np.sin(a)]
    return positions, angles

def _separation_violation(positions, angles):
    """Return max penetration depth (positive means overlap). 0 if all separated."""
    n = len(positions)
    all_verts = _all_vertices(positions, angles)
    hex_normals = np.zeros((n, 6, 2))
    base = np.radians(angles)[:, None] + np.radians(30) + np.radians(60) * np.arange(6)[None, :]
    hex_normals[:, :, 0] = np.cos(base)
    hex_normals[:, :, 1] = np.sin(base)
    max_pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            vi = all_verts[i]
            vj = all_verts[j]
            normals = np.vstack([hex_normals[i], hex_normals[j]])
            proj_i = vi @ normals.T
            proj_j = vj @ normals.T
            max_i = np.max(proj_i, axis=0)
            min_i = np.min(proj_i, axis=0)
            max_j = np.max(proj_j, axis=0)
            min_j = np.min(proj_j, axis=0)
            # separation along normal k: max_i[k] - min_j[k] (positive = apart)
            sep1 = max_i - min_j
            sep2 = max_j - min_i
            # penetration = max(0, -min(sep1, sep2)) over the 12 normals
            # Actually we need: on AT LEAST ONE of 12 normals, intervals apart by > tol.
            # If for ALL 12 normals the intervals overlap (or gap <= tol), it's a violation.
            # Gap along normal k = min(sep1[k], sep2[k])? No.
            # Two intervals [min_i, max_i] and [min_j, max_j] are separated iff max_i < min_j OR max_j < min_i.
            # Gap = max(max_i - min_j, max_j - min_i) ... but if both negative, they overlap.
            # Actually gap = max_i - min_j if max_i < min_j, else max_j - min_i if max_j < min_i, else overlap.
            # The "separation" along normal k is: if max_i < min_j: sep = max_i - min_j (negative gap means apart)
            # Let's define: gap_k = min(min_j - max_i, min_i - max_j)  -> if > 0, separated.
            gap = np.minimum(min_j - max_i, min_i - max_j)
            # If any gap_k > 1e-6, separated. Otherwise, penetration = -min(gap)
            if np.max(gap) > 1e-6:
                continue
            pen = -np.min(gap)
            if pen > max_pen:
                max_pen = pen
    return max_pen

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 620.0

    best_positions = None
    best_angles = None
    best_side = float('inf')

    d = np.sqrt(3)

    def objective(params, beta=500.0):
        R, V, theta = params
        if R < 0.5 or V < 1.0 or R > 10.0 or V > 10.0:
            return 1e10
        positions, angles = _build_symmetric_positions(R, V, theta)
        # Separation penalty: use smooth approximation
        pen = _separation_violation(positions, angles)
        if pen > 1e-6:
            # Heavy penalty for overlap
            return 1e6 * pen + _min_enclosing_side_smooth(positions, angles, beta)
        return _min_enclosing_side_smooth(positions, angles, beta)

    from scipy.optimize import minimize

    # Phase 1: Coarse grid over the 3-parameter space
    R_grid = np.arange(1.5, 3.5, 0.25)
    V_grid = np.arange(2.0, 5.5, 0.25)
    theta_grid = np.arange(0, 60, 5)

    for R in R_grid:
        if time.time() > deadline - 120:
            break
        for V in V_grid:
            if time.time() > deadline - 120:
                break
            for theta in theta_grid:
                if time.time() > deadline - 120:
                    break
                positions, angles = _build_symmetric_positions(R, V, theta)
                if not _check_separation_fast(positions, angles):
                    continue
                side = _min_enclosing_side(positions, angles)
                if side < best_side:
                    best_side = side
                    best_positions = positions.copy()
                    best_angles = angles.copy()

    # Phase 2: L-BFGS-B refinement from best grid point and several good neighbors
    if best_positions is not None and time.time() < deadline - 60:
        # Recover R, V, theta from best_positions
        # ring radius
        ring_pts = best_positions[1:7]
        R0 = float(np.mean(np.linalg.norm(ring_pts, axis=1)))
        valley_pts = best_positions[7:11]
        V0 = float(np.mean(np.linalg.norm(valley_pts, axis=1)))
        # theta: angle of first ring point minus 30 deg
        a0 = np.degrees(np.arctan2(ring_pts[0, 1], ring_pts[0, 0])) - 30.0
        theta0 = a0 % 60.0

        init_candidates = [
            [R0, V0, theta0],
            [R0 * 1.01, V0 * 1.01, theta0],
            [R0 * 0.99, V0 * 0.99, theta0],
            [R0, V0 * 1.02, theta0],
            [R0, V0 * 0.98, theta0],
            [R0 * 1.02, V0, theta0 + 2],
            [R0 * 0.98, V0, theta0 - 2],
        ]

        for init in init_candidates:
            if time.time() > deadline - 30:
                break
            try:
                res = minimize(
                    objective,
                    init,
                    method='L-BFGS-B',
                    bounds=[(0.5, 10.0), (1.0, 10.0), (0.0, 60.0)],
                    options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                if res.fun < 1e5:
                    R, V, theta = res.x
                    positions, angles = _build_symmetric_positions(R, V, theta)
                    if _check_separation_fast(positions, angles):
                        side = _min_enclosing_side(positions, angles)
                        if side < best_side:
                            best_side = side
                            best_positions = positions.copy()
                            best_angles = angles.copy()
            except Exception:
                pass

    # Phase 3: Fine coordinate descent on R, V, theta
    if best_positions is not None and time.time() < deadline - 15:
        ring_pts = best_positions[1:7]
        R = float(np.mean(np.linalg.norm(ring_pts, axis=1)))
        valley_pts = best_positions[7:11]
        V = float(np.mean(np.linalg.norm(valley_pts, axis=1)))
        a0 = np.degrees(np.arctan2(ring_pts[0, 1], ring_pts[0, 0])) - 30.0
        theta = a0 % 60.0

        for cd_iter in range(300):
            if time.time() > deadline - 10:
                break
            improved = False
            for step in [0.01, -0.01, 0.005, -0.005, 0.002, -0.002, 0.001, -0.001]:
                for (dR, dV, dth) in [(step, 0, 0), (0, step, 0), (0, 0, step)]:
                    trial_R = R + dR
                    trial_V = V + dV
                    trial_th = (theta + dth) % 60.0
                    positions, angles = _build_symmetric_positions(trial_R, trial_V, trial_th)
                    if not _check_separation_fast(positions, angles):
                        continue
                    side = _min_enclosing_side(positions, angles)
                    if side < best_side - 1e-12:
                        best_side = side
                        best_positions = positions.copy()
                        best_angles = angles.copy()
                        R, V, theta = trial_R, trial_V, trial_th
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break

    # Fallback
    if best_positions is None:
        positions, angles = _build_symmetric_positions(d, 2 * d, 0.0)
        best_positions = positions
        best_angles = angles
        best_side = _min_enclosing_side(best_positions, best_angles)

    # Final verification
    final_side = _min_enclosing_side(best_positions, best_angles)
    if not _check_separation_fast(best_positions, best_angles):
        positions, angles = _build_symmetric_positions(d, 2 * d, 0.0)
        best_positions = positions
        best_angles = angles
        final_side = _min_enclosing_side(best_positions, best_angles)

    if not _check_containment_fast(best_positions, best_angles, final_side):
        final_side *= 1.001

    inner = np.column_stack([best_positions, best_angles])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    side_length = float(final_side)

    return inner, outer_hex_data, side_length

# EVOLVE-BLOCK-END

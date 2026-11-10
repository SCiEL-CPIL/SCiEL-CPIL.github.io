# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

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

def _soft_max(x, beta):
    """Smooth max via log-sum-exp."""
    m = np.max(x)
    if m < -1e15:
        return 0.0
    return m + np.log(np.sum(np.exp(x - m))) / beta

def _soft_min(x, beta):
    """Smooth min via -log-sum-exp of negated."""
    m = np.min(x)
    if m > 1e15:
        return 0.0
    return m - np.log(np.sum(np.exp(m - x))) / beta

def _separation_margins(positions, angles):
    """Return array of all pairwise separation margins (positive = separated)."""
    n = len(positions)
    all_verts = _all_vertices(positions, angles)
    hex_normals = np.zeros((n, 6, 2))
    base = np.radians(angles)[:, None] + np.radians(30) + np.radians(60) * np.arange(6)[None, :]
    hex_normals[:, :, 0] = np.cos(base)
    hex_normals[:, :, 1] = np.sin(base)

    margins = []
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
            # margin on normal k: min(max_i - min_j, max_j - min_i)
            m1 = max_i - min_j
            m2 = max_j - min_i
            margins.append(np.min(m1, m2))
    return np.array(margins)

def _obj_free(x, beta_sep=50.0, beta_enc=30.0):
    """Objective for free 33-DOF optimization.
    x = [x0,y0,a0, x1,y1,a1, ..., x10,y10,a10] (33 values)
    Minimize: soft_max(enclosing_side) + penalty * soft_min(separation_margins)
    """
    n = 11
    positions = np.column_stack([x[0::3], x[1::3]])
    angles = x[2::3]

    # Enclosing side (smooth max of 6 projections)
    all_verts = _all_vertices(positions, angles).reshape(-1, 2)
    normals = np.array([[np.cos(np.radians(30 + 60 * i)), np.sin(np.radians(30 + 60 * i))] for i in range(6)])
    apothem_factor = np.sqrt(3) / 2
    projs = all_verts @ normals.T
    max_projs = np.max(projs, axis=0)  # shape (6,)
    enc_side = _soft_max(max_projs, beta_enc) / apothem_factor

    # Separation: we want all margins > 0. Penalize negative margins.
    margins = _separation_margins(positions, angles)
    # soft_min of margins: if any margin is very negative, this is very negative
    min_margin = _soft_min(margins, beta_sep)
    # Penalty: large positive when min_margin < 0
    penalty = 0.0
    if min_margin < 0:
        penalty = -min_margin * 10.0  # scale penalty

    return enc_side + penalty

def _build_seed_positions():
    """Build a good initial symmetric arrangement: center + 6 ring + 4 outer."""
    positions = np.zeros((11, 2))
    angles = np.zeros(11)
    positions[0] = [0.0, 0.0]
    for k in range(6):
        a = np.radians(30 + 60 * k)
        positions[k + 1] = [2.0 * np.cos(a), 2.0 * np.sin(a)]
    # 4 outer hexagons at valleys 0, 120, 240, 300
    for k, va in enumerate([0, 120, 240, 300]):
        a = np.radians(va)
        positions[7 + k] = [3.464 * np.cos(a), 3.464 * np.sin(a)]
    return positions, angles

def _build_seed_positions_v2():
    """Alternative seed: 4 outer at 0, 60, 120, 180."""
    positions = np.zeros((11, 2))
    angles = np.zeros(11)
    positions[0] = [0.0, 0.0]
    for k in range(6):
        a = np.radians(30 + 60 * k)
        positions[k + 1] = [2.0 * np.cos(a), 2.0 * np.sin(a)]
    for k, va in enumerate([0, 60, 120, 180]):
        a = np.radians(va)
        positions[7 + k] = [3.464 * np.cos(a), 3.464 * np.sin(a)]
    return positions, angles

def _refine_free(positions, angles, deadline, beta_start=80.0, beta_end=200.0):
    """Run L-BFGS-B on the free 33-DOF problem with annealed beta."""
    x0 = np.zeros(33)
    for k in range(11):
        x0[3*k] = positions[k, 0]
        x0[3*k+1] = positions[k, 1]
        x0[3*k+2] = angles[k]

    best_x = x0.copy()
    best_obj = _obj_free(x0, beta_start, beta_enc=30.0)

    # Multi-stage: increase beta for sharper separation
    stages = [
        (30.0, 50.0, 0.3),   # (beta_enc, beta_sep, fraction of time)
        (40.0, 100.0, 0.3),
        (50.0, 200.0, 0.25),
        (60.0, 400.0, 0.15),
    ]

    total_time_budget = max(1.0, (deadline - time.time()) * 0.7)
    elapsed = 0.0

    for beta_enc, beta_sep, frac in stages:
        stage_budget = total_time_budget * frac
        stage_start = time.time()

        def obj(x, be=beta_enc, bs=beta_sep):
            return _obj_free(x, bs, be)

        try:
            res = minimize(obj, best_x, method='L-BFGS-B',
                           options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-10,
                                    'maxfun': 100000})
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

        # Check deadline
        if time.time() > deadline - 10:
            break

    # Extract best
    positions = np.column_stack([best_x[0::3], best_x[1::3]])
    angles = best_x[2::3] % 360.0
    return positions, angles

def _global_rotation_sweep(positions, angles, deadline):
    """Sweep global rotation to minimize enclosing side."""
    def obj_rot(theta):
        c = np.cos(np.radians(theta))
        s = np.sin(np.radians(theta))
        rot_mat = np.array([[c, -s], [s, c]])
        pos_rot = positions @ rot_mat.T
        angles_rot = (angles + theta) % 360
        return _min_enclosing_side(pos_rot, angles_rot)

    best_rot = 0.0
    best_rot_side = obj_rot(0.0)

    # Coarse sweep
    for t in np.linspace(0, 60, 121):
        s_t = obj_rot(t)
        if s_t < best_rot_side:
            best_rot_side = s_t
            best_rot = t

    # Fine sweep around best
    for t in np.linspace(best_rot - 3, best_rot + 3, 121):
        s_t = obj_rot(t)
        if s_t < best_rot_side:
            best_rot_side = s_t
            best_rot = t

    # Nelder-Mead polish
    try:
        res_rot = minimize(obj_rot, [best_rot], method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
        if res_rot.fun < best_rot_side:
            best_rot = res_rot.x[0]
            best_rot_side = res_rot.fun
    except Exception:
        pass

    if best_rot_side < obj_rot(0.0):
        c = np.cos(np.radians(best_rot))
        s = np.sin(np.radians(best_rot))
        rot_mat = np.array([[c, -s], [s, c]])
        positions = positions @ rot_mat.T
        angles = (angles + best_rot) % 360

    return positions, angles

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 600.0  # safety margin

    best_positions = None
    best_angles = None
    best_side = float('inf')

    # Phase 1: Try multiple seeds with free optimization
    seeds = [
        _build_seed_positions(),
        _build_seed_positions_v2(),
    ]

    for idx, (pos_seed, ang_seed) in enumerate(seeds):
        if time.time() > deadline - 30:
            break
        try:
            pos_ref, ang_ref = _refine_free(pos_seed, ang_seed, deadline)
            if pos_ref is not None:
                # Verify separation
                if _check_separation_fast(pos_ref, ang_ref):
                    side = _min_enclosing_side(pos_ref, ang_ref)
                    if side < best_side:
                        best_side = side
                        best_positions = pos_ref.copy()
                        best_angles = ang_ref.copy()
        except Exception:
            pass

    # Fallback if no valid result
    if best_positions is None:
        positions = np.zeros((11, 2))
        angles = np.zeros(11)
        positions[0] = [0.0, 0.0]
        for k in range(6):
            a = np.radians(30 + 60 * k)
            positions[k + 1] = [2.0 * np.cos(a), 2.0 * np.sin(a)]
        for k, va in enumerate([0, 120, 240, 300]):
            a = np.radians(va)
            positions[7 + k] = [3.464 * np.cos(a), 3.464 * np.sin(a)]
        best_positions = positions
        best_angles = angles
        best_side = _min_enclosing_side(best_positions, best_angles)

    # Phase 2: Global rotation optimization
    if time.time() < deadline - 20:
        best_positions, best_angles = _global_rotation_sweep(best_positions, best_angles, deadline)
        best_side = _min_enclosing_side(best_positions, best_angles)

    # Phase 3: Additional local refinement with tighter tolerance
    if time.time() < deadline - 10:
        try:
            pos_ref, ang_ref = _refine_free(best_positions, best_angles, deadline)
            if _check_separation_fast(pos_ref, ang_ref):
                side = _min_enclosing_side(pos_ref, ang_ref)
                if side < best_side:
                    best_side = side
                    best_positions = pos_ref.copy()
                    best_angles = ang_ref.copy()
                    # Re-do rotation
                    if time.time() < deadline - 5:
                        best_positions, best_angles = _global_rotation_sweep(best_positions, best_angles, deadline)
                        best_side = _min_enclosing_side(best_positions, best_angles)
        except Exception:
            pass

    # Final verification
    final_side = _min_enclosing_side(best_positions, best_angles)
    if not _check_separation_fast(best_positions, best_angles):
        # Fallback to known-valid arrangement
        positions = np.zeros((11, 2))
        angles = np.zeros(11)
        positions[0] = [0.0, 0.0]
        for k in range(6):
            a = np.radians(30 + 60 * k)
            positions[k + 1] = [2.0 * np.cos(a), 2.0 * np.sin(a)]
        for k, va in enumerate([0, 120, 240, 300]):
            a = np.radians(va)
            positions[7 + k] = [3.464 * np.cos(a), 3.464 * np.sin(a)]
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

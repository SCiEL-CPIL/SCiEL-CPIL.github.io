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

def _soft_max(x, beta=50.0):
    m = np.max(x)
    return m + np.log(np.mean(np.exp(x - m))) / beta

def _min_enclosing_side_smooth(positions, angles, beta=50.0):
    all_verts = _all_vertices(positions, angles).reshape(-1, 2)
    normals = np.array([[np.cos(np.radians(30 + 60 * i)), np.sin(np.radians(30 + 60 * i))] for i in range(6)])
    apothem_factor = np.sqrt(3) / 2
    projs = all_verts @ normals.T
    max_proj = _soft_max(projs, beta)
    return max_proj / apothem_factor

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

def _build_symmetric_positions(ring_r, valley_r, angle_offset=0.0, ring_angle=0.0, valley_angle=0.0):
    positions = np.zeros((11, 2))
    angles = np.zeros(11)
    positions[0] = [0.0, 0.0]
    angles[0] = angle_offset
    for k in range(6):
        a = np.radians(30 + 60 * k + angle_offset)
        positions[k + 1] = [ring_r * np.cos(a), ring_r * np.sin(a)]
        angles[k + 1] = ring_angle
    valley_angles_deg = [30, 90, 150, 210]
    for k, va in enumerate(valley_angles_deg):
        a = np.radians(va + angle_offset)
        positions[7 + k] = [valley_r * np.cos(a), valley_r * np.sin(a)]
        angles[7 + k] = valley_angle
    return positions, angles

def _refine_positions(positions, angles, deadline):
    """Refine positions using a smooth (soft-max) surrogate for the enclosing side,
    keeping the separation constraint as a hard gate. Then a high-precision polish pass
    with the exact (nonsmooth) objective and tight tolerances."""
    best_positions = positions.copy()
    best_angles = angles.copy()
    best_side = _min_enclosing_side(positions, angles)

    # Smooth surrogate objective: minimize soft-max enclosing side, penalize overlap.
    def obj_smooth(pos_flat):
        pos = pos_flat.reshape(11, 2)
        if not _check_separation_fast(pos, best_angles):
            return 1e10
        return _min_enclosing_side_smooth(pos, best_angles, beta=50.0)

    try:
        res = minimize(
            obj_smooth,
            best_positions.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        if res.fun < best_side and res.fun < 1e9:
            trial_pos = res.x.reshape(11, 2)
            if _check_separation_fast(trial_pos, best_angles):
                side = _min_enclosing_side(trial_pos, best_angles)
                if side < best_side:
                    best_side = side
                    best_positions = trial_pos
    except Exception:
        pass

    # High-precision polish pass with the exact nonsmooth objective.
    if time.time() < deadline - 30:
        def obj_exact(pos_flat):
            pos = pos_flat.reshape(11, 2)
            if not _check_separation_fast(pos, best_angles):
                return 1e10
            return _min_enclosing_side(pos, best_angles)
        try:
            res2 = minimize(
                obj_exact,
                best_positions.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if res2.fun < best_side and res2.fun < 1e9:
                trial_pos = res2.x.reshape(11, 2)
                if _check_separation_fast(trial_pos, best_angles):
                    side = _min_enclosing_side(trial_pos, best_angles)
                    if side < best_side:
                        best_side = side
                        best_positions = trial_pos
        except Exception:
            pass

    return best_positions, best_angles, best_side

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 620.0

    best_positions = None
    best_angles = None
    best_side = float('inf')

    # Phase 1: Solve specific topologies (graph-embedding with exact tangency).
    topologies = ["symmetric_6_4"]
    for topo in topologies:
        if time.time() > deadline - 120:
            break
        res = _solve_topology(topo)
        if res is not None:
            positions, angles = res
            side = _min_enclosing_side(positions, angles)
            if side < best_side:
                best_side = side
                best_positions = positions.copy()
                best_angles = angles.copy()

    # Phase 2: Dense symmetric grid search to seed a good starting point.
    if time.time() < deadline - 180:
        ring_r_range = np.linspace(1.732, 2.6, 35)
        valley_r_range = np.linspace(2.5, 4.6, 30)
        angle_offset_range = np.linspace(0, 30, 13)
        for ar in ring_r_range:
            if time.time() > deadline - 180:
                break
            for vr in valley_r_range:
                if time.time() > deadline - 180:
                    break
                for ao in angle_offset_range:
                    if time.time() > deadline - 180:
                        break
                    positions, angles = _build_symmetric_positions(ar, vr, ao, 0.0, 0.0)
                    if not _check_separation_fast(positions, angles):
                        continue
                    side = _min_enclosing_side(positions, angles)
                    if side < best_side:
                        best_side = side
                        best_positions = positions.copy()
                        best_angles = angles.copy()

    # Fallback if nothing valid found.
    if best_positions is None:
        positions = np.zeros((11, 2))
        angles = np.zeros(11)
        positions[0] = [0.0, 0.0]
        for k in range(6):
            a = np.radians(30 + 60 * k)
            positions[k + 1] = [2 * np.cos(a), 2 * np.sin(a)]
        for k in range(4):
            a = np.radians(0 + 60 * k)
            positions[7 + k] = [3.5 * np.cos(a), 3.5 * np.sin(a)]
        best_positions = positions
        best_angles = angles
        best_side = _min_enclosing_side(best_positions, best_angles)

    # Phase 3: Smooth-surface local refinement + high-precision polish.
    if time.time() < deadline - 60:
        best_positions, best_angles, best_side = _refine_positions(
            best_positions, best_angles, deadline
        )

    # Phase 4: Best global rotation (hexagon symmetry: 60 deg period).
    if time.time() < deadline - 30:
        def obj_rot(theta):
            c = np.cos(np.radians(theta))
            s = np.sin(np.radians(theta))
            rot_mat = np.array([[c, -s], [s, c]])
            pos_rot = best_positions @ rot_mat.T
            angles_rot = (best_angles + theta) % 360
            return _min_enclosing_side(pos_rot, angles_rot)

        best_rot = 0.0
        best_rot_side = best_side
        for t in np.linspace(0, 60, 121):
            s_t = obj_rot(t)
            if s_t < best_rot_side:
                best_rot_side = s_t
                best_rot = t
        try:
            res_rot = minimize(obj_rot, [best_rot], method='Nelder-Mead',
                               options={'maxiter': 1000, 'xatol': 1e-10})
            if res_rot.fun < best_rot_side:
                best_rot = res_rot.x[0]
                best_rot_side = res_rot.fun
        except Exception:
            pass
        if best_rot_side < best_side:
            best_side = best_rot_side
            c = np.cos(np.radians(best_rot))
            s = np.sin(np.radians(best_rot))
            rot_mat = np.array([[c, -s], [s, c]])
            best_positions = best_positions @ rot_mat.T
            best_angles = (best_angles + best_rot) % 360

    # Final verification and containment.
    final_side = _min_enclosing_side(best_positions, best_angles)
    if not _check_separation_fast(best_positions, best_angles):
        positions = np.zeros((11, 2))
        angles = np.zeros(11)
        positions[0] = [0.0, 0.0]
        for k in range(6):
            a = np.radians(30 + 60 * k)
            positions[k + 1] = [2 * np.cos(a), 2 * np.sin(a)]
        for k in range(4):
            a = np.radians(0 + 60 * k)
            positions[7 + k] = [3.5 * np.cos(a), 3.5 * np.sin(a)]
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

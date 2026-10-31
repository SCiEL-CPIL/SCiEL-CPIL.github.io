"""Packing unit hexagons inside a regular hexagon – vectorised + SA."""
import math
import time
import numpy as np

N = 11
SQRT3 = math.sqrt(3)
DEADLINE = 135.0

# Pre-compute rotation matrices for inner hex vertices
_UNIT_HEX_VERTS = np.array(
    [[math.cos(i * math.pi / 3.0), math.sin(i * math.pi / 3.0)] for i in range(6)],
    dtype=np.float64,
)
_UNIT_HEX_NORMALS = np.array(
    [[math.cos(math.radians(30 + i * 60)), math.sin(math.radians(30 + i * 60))] for i in range(6)],
    dtype=np.float64,
)

# Edge axes for the unit hexagon (perpendicular to edges), used for fast overlap
# For a flat-top unit hexagon the 6 edge normals are exactly the 6 face normals.
_AXES_6 = _UNIT_HEX_NORMALS  # (6,2)


def _rot_mat(angle_deg):
    a = math.radians(angle_deg)
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def _hex_verts_vec(inner):
    """inner: (N,3) ndarray -> (N,6,2) vertices."""
    xy = inner[:, :2]
    ang = inner[:, 2]
    R = np.array([_rot_mat(a) for a in ang])  # (N,2,2)
    # verts = xy + R @ verts_unit  ->  broadcast
    v = np.einsum('nij,kj->niki', R, _UNIT_HEX_VERTS)  # (N,1,6,2)
    return xy[:, np.newaxis, np.newaxis, :] + v  # (N,6,2)


def _compute_required_side(inner, outer_center, outer_angle_deg):
    """
    inner: (N,3) -> (N,6,2) world vertices (absolute).
    outer_center: (2,), outer_angle_deg: float.
    Return side.
    """
    ox, oy = outer_center
    R_o = _rot_mat(outer_angle_deg)
    # rotate inner vertices by -outer angle (inverse rotation)
    # p_rel = v - o ;  p_local = R_o^T @ p_rel  (same as R_o(-angle))
    verts = _hex_verts_vec(inner)
    rel = verts - np.array([ox, oy])
    # R_o^T = _rot_mat(-outer_angle_deg)
    R_ot = _rot_mat(-outer_angle_deg)
    local = np.einsum('ij,nkj->nki', R_ot, rel)  # (N,6,2)
    # projection on each of 6 normals
    # (N,6) x (6,2) dot -> (N,6,1)
    proj = np.einsum('nkp,kd->npd', local, _UNIT_HEX_NORMALS)  # (N,6,1)
    maxp = proj.max()
    return 2.0 * maxp / SQRT3


def _sat_overlap_fast(inner):
    """
    Return True if any pair overlaps.
    inner: (N,3) ndarray.
    Uses edge axes from both hexagons.
    """
    n = len(inner)
    verts = _hex_verts_vec(inner)  # (N,6,2)
    for i in range(n):
        vi = verts[i]
        for j in range(i + 1, n):
            vj = verts[j]
            # collect axes: edge normals of vi and vj
            ok = True
            # axes from vi (6) and vj (6)
            for base in (vi, vj):
                for k in range(6):
                    p1 = base[k]
                    p2 = base[(k + 1) % 6]
                    e = p2 - p1
                    nrm = np.array([-e[1], e[0]])
                    ln = np.linalg.norm(nrm)
                    if ln < 1e-12:
                        continue
                    axis = nrm / ln
                    # project both polygons
                    d1 = vi @ axis
                    d2 = vj @ axis
                    if d1.max() < d2.min() - 1e-9 or d2.max() < d1.min() - 1e-9:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return True
    return False


def _check_inside(inner, outer_center, outer_side, outer_angle_deg, eps=1e-7):
    ox, oy = outer_center
    R_ot = _rot_mat(-outer_angle_deg)
    apothem = outer_side * SQRT3 / 2.0
    verts = _hex_verts_vec(inner)
    rel = verts - np.array([ox, oy])
    local = np.einsum('ij,nkj->nki', R_ot, rel)
    proj = np.einsum('nkp,kd->npd', local, _UNIT_HEX_NORMALS).max(axis=(0, 1))
    return proj <= apothem + eps


def _axial_to_cartesian(qr_list, scale=1.0):
    pts = []
    for q, r in qr_list:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + 0.5 * q)
        pts.append((x, y))
    return pts


# A set of good initial axial coords (7 + 4)
_AXIAL_DEFAULT = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
                  (2, 0), (1, 1), (-1, 2), (-2, 1)]


def _energy(inner, outer_center, outer_angle_deg):
    side = _compute_required_side(inner, outer_center, outer_angle_deg)
    # overlap penalty via center distances (approx for fast SA)
    xy = inner[:, :2]
    penalty = 0.0
    min_d = SQRT3 * (1.0)  # for aligned; with rotation the min can be lower, use safe
    min_d_safe = SQRT3 * 0.98
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(xy[i] - xy[j])
            if d < min_d_safe:
                penalty += (min_d_safe - d) ** 2
    return side + 50.0 * penalty


def construct():
    start_time = time.time()

    # Initial inner positions: axial lattice, all angles 0
    pts = _axial_to_cartesian(_AXIAL_DEFAULT, scale=1.0)
    inner0 = np.zeros((N, 3), dtype=np.float64)
    for k, (x, y) in enumerate(pts):
        inner0[k, 0] = x
        inner0[k, 1] = y
        inner0[k, 2] = 0.0

    # Initial outer: centroid and 0 angle
    cx0 = float(np.mean(inner0[:, 0]))
    cy0 = float(np.mean(inner0[:, 1]))
    o_angle0 = 0.0
    o_c0 = np.array([cx0, cy0], dtype=np.float64)

    best_state = (inner0.copy(), o_c0.copy(), o_angle0)
    best_score = float('inf')

    def _eval_state(inner, oc, oa):
        s = _compute_required_side(inner, oc, oa)
        return s

    best_score = _eval_state(*best_state)

    # Simulated Annealing
    # State: positions (N,2), angles (N,), outer center (2,), outer angle (1)
    rng = np.random.default_rng(42)

    x_pos = inner0[:, :2].copy()  # (N,2)
    x_ang = inner0[:, 2].copy()   # (N,)
    oc = o_c0.copy()
    oa = o_angle0

    def _pack_state(x_pos, x_ang, oc, oa):
        st = np.zeros((N, 3), dtype=np.float64)
        st[:, 0] = x_pos[:, 0]
        st[:, 1] = x_pos[:, 1]
        st[:, 2] = x_ang
        return st, oc, oa

    T0 = 0.5
    Tf = 1e-4
    cooling = 0.985
    T = T0
    max_iter = 200000
    accept = 0

    for it in range(max_iter):
        if time.time() - start_time > DEADLINE - 5.0:
            break
        if it % 200 == 0 and T <= Tf:
            break

        # random move: pick a coordinate to perturb
        move_type = rng.integers(0, 6)
        step_pos = 0.05 + 0.2 * T
        step_ang = 0.3 + 1.0 * T
        step_oe = 0.05 + 0.2 * T

        old_score = _energy(
            np.column_stack([x_pos, x_ang]), oc, oa
        )

        if move_type == 0:  # position of a random inner
            i = rng.integers(0, N)
            delta = rng.normal(0, step_pos, size=2)
            x_pos[i] += delta
        elif move_type == 1:  # angle of a random inner
            i = rng.integers(0, N)
            delta = rng.normal(0, step_ang)
            x_ang[i] = (x_ang[i] + delta + 180.0) % 360.0 - 180.0
        elif move_type == 2:  # outer center x
            delta = rng.normal(0, step_oe)
            oc[0] += delta
        elif move_type == 3:  # outer center y
            delta = rng.normal(0, step_oe)
            oc[1] += delta
        elif move_type == 4:  # outer angle
            delta = rng.normal(0, step_oe)
            oa += delta
            oa = (oa + 30.0) % 60.0  # hexagonal symmetry
        else:  # nudge: small simultaneous move
            i = rng.integers(0, N)
            x_pos[i] += rng.normal(0, step_pos * 0.5, size=2)
            x_ang[i] = (x_ang[i] + rng.normal(0, step_ang * 0.5) + 180.0) % 360.0 - 180.0

        new_score = _energy(
            np.column_stack([x_pos, x_ang]), oc, oa
        )

        if new_score < old_score or rng.random() < math.exp(-(new_score - old_score) / max(T, 1e-9)):
            T *= cooling

        # track best by side length (not penalised) if valid enough
        # We'll just keep the best by side (overlap will be corrected later)
        if new_score < best_score:
            best_score = new_score
            best_state = (
                np.column_stack([x_pos, x_ang]).copy(),
                oc.copy(),
                oa,
            )
        accept += 1

    # After SA, extract best state
    inner_best, oc_best, oa_best = best_state

    # Nelder-Mead polish using scipy
    try:
        from scipy.optimize import minimize
    except Exception:
        minimize = None

    if minimize is not None:
        def _obj(params):
            # params: N*2+ N angles + 3 outer
            xp = params[:N * 2].reshape(N, 2)
            xa = params[N * 2:N * 3]
            o = params[N * 3:]
            st = np.column_stack([xp, xa])
            s = _compute_required_side(st, o[:2], o[2])
            # hard-ish overlap penalty
            pen = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.linalg.norm(xp[i] - xp[j])
                    if d < SQRT3 * 0.97:
                        pen += (SQRT3 * 0.97 - d) ** 2
            return s + 1e4 * pen

        init_params = np.concatenate([inner_best[:, :2].ravel(), inner_best[:, 2].ravel(),
                                       [oc_best[0], oc_best[1], oa_best]])
        for trial in range(3):
            if time.time() - start_time > DEADLINE - 10:
                break
            p0 = init_params.copy()
            if trial > 0:
                p0 += rng.normal(0, 0.02, size=len(p0))
            try:
                res = minimize(
                    _obj, p0, method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9, 'maxfev': 600}
                )
                xp = res.x[:N * 2].reshape(N, 2)
                xa = res.x[N * 2:N * 3]
                o = res.x[N * 3:]
                st = np.column_stack([xp, xa])
                s = _compute_required_side(st, o[:2], o[2])
                if s < best_score:
                    # Validate
                    if not _sat_overlap_fast(st) and _check_inside(st, o[:2], s, o[2]):
                        best_score = s
                        best_state = (st.copy(), o[:2].copy(), float(o[2]))
            except Exception:
                pass

    # Final validation and fallback
    final_inner, final_oc, final_oa = best_state
    final_side = _compute_required_side(final_inner, final_oc, final_oa)

    if _sat_overlap_fast(final_inner) or not _check_inside(final_inner, final_oc, final_side, final_oa):
        # Fallback: slightly expanded lattice to guarantee validity
        scale_f = 1.02
        pts_f = _axial_to_cartesian(_AXIAL_DEFAULT, scale_f)
        fb = np.zeros((N, 3), dtype=np.float64)
        for k, (x, y) in enumerate(pts_f):
            fb[k, 0] = x
            fb[k, 1] = y
        fb_oc = np.array([0.0, 0.0])
        fb_oa = 0.0
        fb_side = _compute_required_side(fb, fb_oc, fb_oa)
        final_inner = fb
        final_oc = fb_oc
        final_oa = fb_oa
        final_side = fb_side

    # Format return values as required: list of rows, center tuple, side, angle
    inner_list = [[float(r[0]), float(r[1]), float(r[2])] for r in final_inner]
    center = (float(final_oc[0]), float(final_oc[1]))

    return inner_list, center, float(final_side), float(final_oa)


# Alias kept for compatibility if evaluator expects the exact name
if __name__ == '__main__':
    # Quick self-test
    res = construct()
    print("side =", res[2])

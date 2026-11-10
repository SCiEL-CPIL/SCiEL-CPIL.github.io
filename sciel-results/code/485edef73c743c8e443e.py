# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# ---------- geometry helpers (match the task contract exactly) ----------
def _hex_vertices(cx, cy, s, angle_deg):
    angles = np.radians(angle_deg) + np.radians(60) * np.arange(6)
    return np.column_stack([
        cx + s * np.cos(angles),
        cy + s * np.sin(angles)
    ])

def _all_vertices(positions, angles, side=1.0):
    n = len(positions)
    base_angles = np.radians(angles)[:, None] + np.radians(60) * np.arange(6)[None, :]
    verts = np.zeros((n, 6, 2))
    verts[:, :, 0] = positions[:, 0:1] + side * np.cos(base_angles)
    verts[:, :, 1] = positions[:, 1:2] + side * np.sin(base_angles)
    return verts

def _separated(positions, angles, tol=1e-6):
    n = len(positions)
    all_verts = _all_vertices(positions, angles)
    base = np.radians(angles)[:, None] + np.radians(30) + np.radians(60) * np.arange(6)[None, :]
    hex_normals = np.stack([np.cos(base), np.sin(base)], axis=2)
    for i in range(n):
        vi = all_verts[i]
        ni = hex_normals[i]
        for j in range(i + 1, n):
            vj = all_verts[j]
            nj = hex_normals[j]
            normals = np.vstack([ni, nj])
            proj_i = vi @ normals.T
            proj_j = vj @ normals.T
            if np.any(np.max(proj_i, axis=0) < np.min(proj_j, axis=0) - tol) \
               or np.any(np.max(proj_j, axis=0) < np.min(proj_i, axis=0) - tol):
                continue
            return False
    return True

def _enclosing_side(positions, angles, outer_angle=0.0):
    all_verts = _all_vertices(positions, angles).reshape(-1, 2)
    normals = np.array([[np.cos(np.radians(outer_angle + 30 + 60 * i)),
                         np.sin(np.radians(outer_angle + 30 + 60 * i))] for i in range(6)])
    projs = all_verts @ normals.T
    return np.max(projs) / (np.sqrt(3) / 2)

def _soft_max(x, beta=100.0):
    m = np.max(x)
    return m + np.log(np.mean(np.exp(x - m))) / beta

def _enclosing_side_soft(positions, angles, outer_angle=0.0, beta=100.0):
    all_verts = _all_vertices(positions, angles).reshape(-1, 2)
    normals = np.array([[np.cos(np.radians(outer_angle + 30 + 60 * i)),
                         np.sin(np.radians(outer_angle + 30 + 60 * i))] for i in range(6)])
    projs = all_verts @ normals.T
    return _soft_max(projs, beta) / (np.sqrt(3) / 2)

def _contained(positions, angles, side, outer_angle=0.0, tol=-1e-6):
    all_verts = _all_vertices(positions, angles).reshape(-1, 2)
    outer = _hex_vertices(0.0, 0.0, side, outer_angle)
    for i in range(6):
        p1 = outer[i]
        p2 = outer[(i + 1) % 6]
        e = p2 - p1
        v = all_verts - p1
        cross = e[0] * v[:, 1] - e[1] * v[:, 0]
        if np.min(cross) < tol:
            return False
    return True

# ---------- candidate generators ----------
def _sym_candidate(ring_r, valley_r, angle_offset=0.0):
    positions = np.zeros((11, 2))
    angles = np.zeros(11)
    positions[0] = [0.0, 0.0]
    for k in range(6):
        a = np.radians(30 + 60 * k + angle_offset)
        positions[k + 1] = [ring_r * np.cos(a), ring_r * np.sin(a)]
    for k in range(4):
        a = np.radians(0 + 60 * k + angle_offset)
        positions[7 + k] = [valley_r * np.cos(a), valley_r * np.sin(a)]
    return positions, angles

def _random_candidate(rng):
    positions = np.zeros((11, 2))
    angles = np.zeros(11)
    positions[0] = [0.0, 0.0]
    angles[0] = rng.uniform(0, 60)
    for k in range(6):
        a = np.radians(30 + 60 * k + rng.uniform(-4, 4))
        r = rng.uniform(1.8, 2.6)
        positions[k + 1] = [r * np.cos(a), r * np.sin(a)]
        angles[k + 1] = rng.uniform(0, 60)
    for k in range(4):
        a = np.radians(0 + 60 * k + rng.uniform(-8, 8))
        r = rng.uniform(2.8, 4.4)
        positions[7 + k] = [r * np.cos(a), r * np.sin(a)]
        angles[7 + k] = rng.uniform(0, 60)
    return positions, angles

def _perturb(positions, angles, rng, scale):
    p = positions + rng.normal(0, scale, positions.shape)
    a = (angles + rng.normal(0, scale * 6, angles.shape)) % 360
    return p, a

# ---------- local refinement ----------
def _refine(positions, angles, deadline, rng, outer_angle=0.0):
    best_pos = positions.copy()
    best_ang = angles.copy()
    best_side = _enclosing_side(positions, angles, outer_angle)

    def _run(obj, x0, maxiter):
        nonlocal best_pos, best_ang, best_side
        try:
            res = minimize(
                obj, x0, method='Nelder-Mead',
                options={'maxiter': maxiter, 'xatol': 1e-12, 'fatol': 1e-13,
                         'adaptive': True, 'maxfev': maxiter * 40}
            )
            pos = res.x[:22].reshape(11, 2)
            ang = res.x[22:] % 360
            if _separated(pos, ang):
                s = _enclosing_side(pos, ang, outer_angle)
                if s < best_side:
                    best_side = s
                    best_pos = pos.copy()
                    best_ang = ang.copy()
        except Exception:
            pass

    # smooth-surface pass
    def obj_smooth(x):
        pos = x[:22].reshape(11, 2)
        ang = x[22:]
        if not _separated(pos, ang):
            return 1e6 + 0.001 * np.sum((pos) ** 2)
        return _enclosing_side_soft(pos, ang, outer_angle, beta=100.0)

    x0 = np.concatenate([positions.flatten(), angles])
    _run(obj_smooth, x0, 2000)

    # exact (nonsmooth) polish pass
    if time.time() < deadline - 30:
        def obj_exact(x):
            pos = x[:22].reshape(11, 2)
            ang = x[22:]
            if not _separated(pos, ang):
                return 1e6 + 0.001 * np.sum((pos) ** 2)
            return _enclosing_side(pos, ang, outer_angle)

        x1 = np.concatenate([best_pos.flatten(), best_ang])
        _run(obj_exact, x1, 3000)

    return best_pos, best_ang, best_side

# ---------- joint rotation + outer angle optimization ----------
def _best_rotation_joint(positions, angles):
    """Optimize cluster rotation theta and outer hex angle phi jointly."""
    best_side = _enclosing_side(positions, angles, 0.0)
    best_theta = 0.0
    best_phi = 0.0

    def _eval(theta, phi):
        c = np.cos(np.radians(theta))
        s = np.sin(np.radians(theta))
        rot = np.array([[c, -s], [s, c]])
        p_rot = positions @ rot.T
        a_rot = (angles + theta) % 360
        return _enclosing_side(p_rot, a_rot, phi)

    # Coarse grid
    for theta in np.linspace(0, 60, 31):
        for phi in np.linspace(0, 60, 31):
            s = _eval(theta, phi)
            if s < best_side:
                best_side = s
                best_theta = theta
                best_phi = phi

    # Fine grid around best
    for theta in np.linspace(max(0, best_theta - 5), min(60, best_theta + 5), 41):
        for phi in np.linspace(max(0, best_phi - 5), min(60, best_phi + 5), 41):
            s = _eval(theta, phi)
            if s < best_side:
                best_side = s
                best_theta = theta
                best_phi = phi

    # Nelder-Mead polish
    try:
        res = minimize(
            lambda v: _eval(v[0], v[1]),
            [best_theta, best_phi],
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12,
                     'adaptive': True, 'maxfev': 60000}
        )
        if res.fun < best_side:
            best_side = res.fun
            best_theta = res.x[0] % 60
            best_phi = res.x[1] % 60
    except Exception:
        pass

    c = np.cos(np.radians(best_theta))
    s = np.sin(np.radians(best_theta))
    rot = np.array([[c, -s], [s, c]])
    p_rot = positions @ rot.T
    a_rot = (angles + best_theta) % 360
    return p_rot, a_rot, best_phi, best_side

# ---------- main entry point ----------
def hexagon_packing_11():
    start = time.time()
    deadline = start + 580.0  # safety margin
    rng = np.random.default_rng(42)

    best_pos = None
    best_ang = None
    best_side = float('inf')
    best_phi = 0.0

    def consider(pos, ang):
        nonlocal best_pos, best_ang, best_side
        if not _separated(pos, ang):
            return False
        s = _enclosing_side(pos, ang, 0.0)
        if s < best_side:
            best_side = s
            best_pos = pos.copy()
            best_ang = ang.copy()
            return True
        return False

    # --- Phase A: coarse deterministic grid ---
    for ring_r in np.linspace(1.75, 2.6, 15):
        if time.time() > deadline - 180:
            break
        for valley_r in np.linspace(2.9, 4.5, 15):
            if time.time() > deadline - 180:
                break
            for ao in (0.0, 10.0, 20.0, 30.0):
                pos, ang = _sym_candidate(ring_r, valley_r, ao)
                consider(pos, ang)

    # --- Phase B: randomized multi-start + local refinement ---
    n_starts = 0
    while time.time() < deadline - 200:
        if n_starts % 3 == 0:
            pos, ang = _sym_candidate(rng.uniform(1.8, 2.6), rng.uniform(2.9, 4.4), rng.uniform(0, 30))
        else:
            pos, ang = _random_candidate(rng)
        n_starts += 1
        consider(pos, ang)
        pos2, ang2, s2 = _refine(pos, ang, deadline, rng, outer_angle=0.0)
        if s2 < best_side:
            best_side = s2
            best_pos = pos2.copy()
            best_ang = ang2.copy()

    # --- fallback ---
    if best_pos is None:
        pos, ang = _sym_candidate(2.0, 3.5, 0.0)
        best_pos, best_ang, best_side = pos, ang, _enclosing_side(pos, ang, 0.0)

    # --- Phase C: polish the best, then joint rotation + outer angle ---
    if time.time() < deadline - 80:
        pos2, ang2, s2 = _refine(best_pos, best_ang, deadline, rng, outer_angle=0.0)
        if s2 < best_side:
            best_pos, best_ang, best_side = pos2, ang2, s2

    if time.time() < deadline - 40:
        pos_r, ang_r, phi_r, s_r = _best_rotation_joint(best_pos, best_ang)
        if s_r < best_side:
            best_pos, best_ang, best_side, best_phi = pos_r, ang_r, phi_r, s_r
        # one more polish after rotation
        if time.time() < deadline - 25:
            pos2, ang2, s2 = _refine(best_pos, best_ang, deadline, rng, outer_angle=best_phi)
            if s2 < best_side:
                best_pos, best_ang, best_side = pos2, ang2, s2

    # --- Final verification ---
    final_side = _enclosing_side(best_pos, best_ang, best_phi)
    if not _separated(best_pos, best_ang):
        pos, ang = _sym_candidate(2.0, 3.5, 0.0)
        best_pos, best_ang, best_phi = pos, ang, 0.0
        final_side = _enclosing_side(pos, ang, 0.0)

    if not _contained(best_pos, best_ang, final_side, best_phi):
        final_side *= 1.001

    inner = np.column_stack([best_pos, best_ang])
    outer = np.array([0.0, 0.0, best_phi])
    return inner, outer, float(final_side)

# EVOLVE-BLOCK-END

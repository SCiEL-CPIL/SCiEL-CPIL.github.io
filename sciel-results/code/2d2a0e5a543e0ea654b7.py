# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, side, angle_deg):
    """Compute 6 vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])


# Outer hexagon (centered at origin, angle 0) outward unit edge normals
_NORMAL_ANGLES = np.radians([30, 90, 150, 210, 270, 330])
_NORMALS = np.column_stack([np.cos(_NORMAL_ANGLES), np.sin(_NORMAL_ANGLES)])  # (6,2)
_APOTHEM = np.sqrt(3.0) / 2.0


def _objective(params):
    """Minimum outer side length s (centered at origin, angle 0) containing all inner hexagons."""
    pos = params[0::3]
    ys = params[1::3]
    ang = params[2::3]
    a = np.radians(ang)
    # vertices[i, j, k] = coordinate k of vertex j of hexagon i
    angles = a[:, None] + np.arange(6)[None, :] * np.pi / 3.0
    verts = pos[:, None, None] + np.stack([np.cos(angles), np.sin(angles)], axis=2)
    # max projection of any vertex onto any outward normal
    max_proj = np.max(np.einsum('ijk,dk->ijd', verts, _NORMALS))
    return max_proj / _APOTHEM


def _check_validity(inner_hex_data, outer_hex_data, s, tol=1e-6):
    """Check if the packing is valid (mirrors the task contract)."""
    if not np.all(np.isfinite(inner_hex_data)) or not np.all(np.isfinite(outer_hex_data)):
        return False
    if s < np.sqrt(12.0) - 1e-9:
        return False

    outer_verts = _hex_vertices(outer_hex_data[0], outer_hex_data[1], s, outer_hex_data[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False

    # unit edge lengths
    for i in range(12):
        verts = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
        for k in range(6):
            if abs(np.linalg.norm(verts[(k + 1) % 6] - verts[k]) - 1.0) > 1e-8:
                return False

    # separation between inner hexagons (SAT, normals of both)
    allverts = np.array([
        _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
        for i in range(12)
    ])
    for i in range(12):
        for j in range(i + 1, 12):
            vi, vj = allverts[i], allverts[j]
            separated = False
            for vv in (vi, vj):
                for k in range(6):
                    edge = vv[(k + 1) % 6] - vv[k]
                    nrm = np.linalg.norm(edge)
                    if nrm < 1e-12:
                        continue
                    normal = np.array([-edge[1], edge[0]]) / nrm
                    if np.max(vi @ normal) + tol < np.min(vj @ normal) or \
                       np.max(vj @ normal) + tol < np.min(vi @ normal):
                        separated = True
                        break
                if separated:
                    break
            if not separated:
                return False

    # containment: every inner vertex inside outer hexagon
    for i in range(12):
        verts = allverts[i]
        for v in verts:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < -tol:
                    return False
    return True


def hexagon_packing_12():
    """
    Packs 12 unit regular hexagons in a regular hexagon, maximizing 1/outer_side_length.
    Returns (inner (12,3), outer (3,), s).
    """
    start_time = time.time()
    deadline = 600.0

    def _remaining():
        return deadline - (time.time() - start_time)

    # ---- starting configurations ----
    sqrt3 = np.sqrt(3.0)
    seed_positions = np.array([
        [0, 0], [-2.5, 0], [2.5, 0],
        [-1.25, 2.17], [1.25, 2.17], [-1.25, -2.17], [1.25, -2.17],
        [-3.75, 2.17], [3.75, 2.17], [-3.75, -2.17], [3.75, -2.17],
        [0, -4]
    ])
    hex_positions = np.array([
        [0, 0],
        [3, 0], [1.5, sqrt3], [-1.5, sqrt3], [-3, 0], [-1.5, -sqrt3], [1.5, -sqrt3],
        [4.5, sqrt3], [3, 2 * sqrt3], [0, 2 * sqrt3], [-3, 2 * sqrt3], [-4.5, sqrt3]
    ])

    configs = []
    configs.append((seed_positions.copy(), np.zeros(12)))
    configs.append((hex_positions.copy(), np.zeros(12)))
    for rot in (15.0, 30.0, 45.0):
        configs.append((hex_positions.copy(), np.full(12, rot)))
    for trial in range(12):
        rng = np.random.RandomState(trial * 100 + 42)
        pos = hex_positions + rng.randn(12, 2) * 0.1
        ang = rng.uniform(-30, 30, 12)
        configs.append((pos, ang))
    for trial in range(24):
        rng = np.random.RandomState(trial * 1000 + 7)
        pos = np.zeros((12, 2))
        for i in range(12):
            r = np.sqrt(rng.uniform(0, 1)) * 4.0
            th = rng.uniform(0, 2 * np.pi)
            pos[i] = [r * np.cos(th), r * np.sin(th)]
        ang = rng.uniform(-30, 30, 12)
        configs.append((pos, ang))

    def _pack(pos, ang):
        p = np.zeros(36)
        p[0::3] = pos[:, 0]
        p[1::3] = pos[:, 1]
        p[2::3] = ang
        return p

    best_s = float('inf')
    best_params = None

    for pos, ang in configs:
        if _remaining() < 20.0:
            break
        p0 = _pack(pos, ang)
        s0 = _objective(p0)
        if s0 < best_s:
            best_s = s0
            best_params = p0.copy()

        # Nelder-Mead (derivative-free, robust for the non-smooth max)
        try:
            res = minimize(_objective, p0, method='Nelder-Mead',
                           options={'maxiter': 1500, 'xatol': 1e-8,
                                    'fatol': 1e-10, 'adaptive': True})
            if res.fun < best_s:
                best_s = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

        # L-BFGS-B (fast local refinement)
        if _remaining() < 20.0:
            break
        try:
            res2 = minimize(_objective, p0, method='L-BFGS-B',
                            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
            if res2.fun < best_s:
                best_s = res2.fun
                best_params = res2.x.copy()
        except Exception:
            pass

    if best_params is None:
        best_params = _pack(seed_positions, np.zeros(12))

    # ---- final output ----
    inner = np.zeros((12, 3))
    inner[:, 0] = best_params[0::3]
    inner[:, 1] = best_params[1::3]
    inner[:, 2] = best_params[2::3]
    outer = np.array([0.0, 0.0, 0.0])
    s = _objective(best_params)

    if not _check_validity(inner, outer, s):
        # safe fallback: the known-good seed arrangement
        inner = np.array([
            [0, 0, 0], [-2.5, 0, 0], [2.5, 0, 0],
            [-1.25, 2.17, 0], [1.25, 2.17, 0], [-1.25, -2.17, 0], [1.25, -2.17, 0],
            [-3.75, 2.17, 0], [3.75, 2.17, 0], [-3.75, -2.17, 0], [3.75, -2.17, 0],
            [0, -4, 0]
        ])
        outer = np.array([0.0, 0.0, 0.0])
        s = _objective(_pack(inner[:, 0], inner[:, 1]))

    return inner, outer, s


# EVOLVE-BLOCK-END

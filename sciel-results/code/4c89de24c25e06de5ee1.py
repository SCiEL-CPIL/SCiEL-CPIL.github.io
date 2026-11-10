# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = 620.0  # safety margin before 630s kill


def _hex_verts(cx, cy, s, a_deg):
    a = np.radians(a_deg)
    ang = a + np.radians(60.0 * np.arange(6))
    return np.stack([cx + s * np.cos(ang), cy + s * np.sin(ang)], axis=1)


def _outer_verts(cx, cy, s, a_deg):
    return _hex_verts(cx, cy, s, a_deg)


def _normals_of(verts):
    n = verts.shape[0]
    out = []
    for i in range(n):
        p1 = verts[i]
        p2 = verts[(i + 1) % n]
        e = p2 - p1
        ln = np.linalg.norm(e)
        if ln < 1e-15:
            out.append(np.zeros(2))
            continue
        # inward normal for CCW polygon: rotate edge by -90
        nrm = np.array([e[1], -e[0]]) / ln
        out.append(nrm)
    return out


def _check_pair_separated(v1, n1, v2, n2):
    """Return True if two convex polygons are separated (non-intersecting).
    A pair is separated if on at least one of the combined edge normals the
    projected intervals are apart by more than 1e-6."""
    eps = 1e-6
    normals = list(n1) + list(n2)
    for nrm in normals:
        d1 = v1 @ nrm
        d2 = v2 @ nrm
        min1, max1 = d1.min(), d1.max()
        min2, max2 = d2.min(), d2.max()
        # intervals apart means max of one < min of other
        if max1 < min2 - eps or max2 < min1 - eps:
            return True
    return False


def _validate(inner, outer, side):
    """Return (valid, max_violation)."""
    # finite
    if not (np.all(np.isfinite(inner)) and np.all(np.isfinite(outer))):
        return False, np.inf
    if not np.isfinite(side) or side < np.sqrt(11.0):
        return False, np.inf
    ocx, ocy, oang = outer[0], outer[1], outer[2]
    outer_v = _outer_verts(ocx, ocy, side, oang)
    # outer edges
    on = len(outer_v)
    oedges = []
    for i in range(on):
        p1 = outer_v[i]
        p2 = outer_v[(i + 1) % on]
        oedges.append((p1, p2))
    onorms = _normals_of(outer_v)

    # build inner vertices (shifted by outer center)
    inner_verts = []
    inner_norms = []
    for k in range(inner.shape[0]):
        cx, cy, adeg = inner[k]
        v = _hex_verts(cx, cy, 1.0, adeg)
        vs = v - np.array([ocx, ocy])
        # edge lengths within 1e-8 of 1
        for i in range(6):
            e = vs[(i + 1) % 6] - vs[i]
            if abs(np.linalg.norm(e) - 1.0) > 1e-8:
                return False, np.inf
        inner_verts.append(vs)
        inner_norms.append(_normals_of(vs))

    # containment: every inner vertex inside outer
    max_viol = 0.0
    for vs in inner_verts:
        for p in vs:
            for (p1, p2) in oedges:
                cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                if cross < max_viol:
                    max_viol = cross
                if cross < -1e-6:
                    return False, max_viol
    # pairwise separation
    for i in range(inner.shape[0]):
        for j in range(i + 1, inner.shape[0]):
            if not _check_pair_separated(inner_verts[i], inner_norms[i],
                                         inner_verts[j], inner_norms[j]):
                return False, max_viol
    return True, max_viol


def _required_side(inner, outer_center, oang):
    """Smallest side length (centered at outer_center, orientation oang)
    containing all inner hexagons. We sample many candidate side lengths by
    computing the max over inner vertices of the support function along each
    of the 6 outer edge normals; side must satisfy s*cos(30)*? Actually for a
    regular hexagon with flat orientation, the inradius is s*cos(30).
    The half-width along each of the 6 edge normals equals s*cos(30)."""
    ocx, ocy = outer_center
    # edge normals of outer hexagon (6 directions, 60 deg apart)
    a = np.radians(oang)
    nrm_angs = a + np.radians(60.0 * np.arange(6)) + np.radians(90.0)
    normals = np.stack([np.cos(nrm_angs), np.sin(nrm_angs)], axis=1)
    # inradius = s * cos(30)
    inr = np.cos(np.radians(30.0))
    max_proj = 0.0
    for k in range(inner.shape[0]):
        cx, cy, adeg = inner[k]
        v = _hex_verts(cx, cy, 1.0, adeg) - np.array([ocx, ocy])
        proj = v @ normals.T  # (6,)
        max_proj = max(max_proj, proj.max())
    side = max_proj / inr
    return side


def _objective(x, oang):
    # x: 11*3 = 33 values: (cx, cy, adeg) per hexagon
    inner = x.reshape(11, 3)
    # containment: outer hexagon centered at origin (we optimize around it)
    # side needed
    side = _required_side(inner, np.array([0.0, 0.0]), oang)
    # non-overlap penalty
    verts = [_hex_verts(inner[k][0], inner[k][1], 1.0, inner[k][2]) for k in range(11)]
    norms = [_normals_of(v) for v in verts]
    pen = 0.0
    for i in range(11):
        for j in range(i + 1, 11):
            # compute min gap over combined normals; if any normal separates, ok
            sep = -np.inf
            for nrm in norms[i] + norms[j]:
                d1 = verts[i] @ nrm
                d2 = verts[j] @ nrm
                gap = min(d1.max(), d2.max()) - max(d1.min(), d2.min())
                # gap > 0 means separated along this normal
                sep = max(sep, gap)
            if sep < 1e-6:
                pen += (1e-6 - sep) ** 2
    return side + 100.0 * pen


def hexagon_packing_11():
    deadline = _START + _DEADLINE
    # initial arrangement: hexagonal cluster of 11 (center + 6 ring + 4 more)
    # Use axial-ish layout
    s = 1.0
    # ring positions for a hexagon lattice: distance between centers 2.0
    d = 2.0
    init = np.array([
        [0.0, 0.0, 0.0],
        [d, 0.0, 0.0],
        [d / 2, d * np.sqrt(3) / 2, 0.0],
        [-d / 2, d * np.sqrt(3) / 2, 0.0],
        [-d, 0.0, 0.0],
        [-d / 2, -d * np.sqrt(3) / 2, 0.0],
        [d / 2, -d * np.sqrt(3) / 2, 0.0],
        [1.5 * d, d * np.sqrt(3) / 2, 0.0],
        [1.5 * d, -d * np.sqrt(3) / 2, 0.0],
        [-1.5 * d, d * np.sqrt(3) / 2, 0.0],
        [-1.5 * d, -d * np.sqrt(3) / 2, 0.0],
    ])

    best_inner = init.copy()
    best_side = _required_side(best_inner, np.array([0.0, 0.0]), 0.0)

    # try several outer orientations and local optimizations
    for oang in [0.0, 30.0]:
        x0 = init.copy()
        # run a few restarts
        for restart in range(3):
            if time.time() > deadline - 30:
                break
            if restart > 0:
                x0 = init + np.random.randn(33) * 0.5
            res = minimize(_objective, x0, args=(oang,), method='Nelder-Mead',
                           options={'maxiter': 4000, 'xatol': 1e-6, 'fatol': 1e-8})
            inner = res.x.reshape(11, 3)
            side = _required_side(inner, np.array([0.0, 0.0]), oang)
            # validate
            outer = np.array([0.0, 0.0, oang])
            valid, _ = _validate(inner, outer, side)
            if valid and side < best_side:
                best_inner = inner.copy()
                best_side = side
            # also try to shrink: use side as outer
            if side < best_side:
                best_inner = inner.copy()
                best_side = side

    # final: build outer centered at origin with best_side
    outer = np.array([0.0, 0.0, 0.0])
    # recompute side precisely with final orientation 0
    side = _required_side(best_inner, np.array([0.0, 0.0]), 0.0)
    # validate once more; if invalid, fall back to a safe large side
    valid, _ = _validate(best_inner, outer, side)
    if not valid:
        # expand until valid
        side2 = side
        for _ in range(50):
            valid, _ = _validate(best_inner, outer, side2)
            if valid:
                break
            side2 *= 1.05
        side = side2
    if not valid:
        # ultimate fallback: static grid with large side
        inner = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
        ], dtype=float)
        outer = np.array([0.0, 0.0, 0.0])
        side = 8.0

    return best_inner, outer, float(side)


# EVOLVE-BLOCK-END

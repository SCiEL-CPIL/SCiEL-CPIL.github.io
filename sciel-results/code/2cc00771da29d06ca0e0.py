# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 600.0
_MARGIN = 10.0
_EPS = 1e-9


def _hex_verts(cx, cy, d, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0) * np.arange(6)
    return np.column_stack([cx + d * np.cos(th), cy + d * np.sin(th)])


def _project_minmax(pts, axis):
    p = pts @ axis
    return float(p.min()), float(p.max())


def _separation(v1, v2):
    best = 0.0
    for k in range(6):
        p1 = v1[k]
        p2 = v1[(k + 1) % 6]
        e = p2 - p1
        L = np.hypot(e[0], e[1])
        if L < 1e-12:
            continue
        ax = np.array([-e[1], e[0]])
        ax = ax / L
        a1, a2 = _project_minmax(v1, ax)
        b1, b2 = _project_minmax(v2, ax)
        s = min(a2, b2) - max(a1, b1)
        if s > best:
            best = s
    return best


def _separation_all(vlist):
    n = len(vlist)
    best = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            s = _separation(vlist[i], vlist[j])
            if s > best:
                best = s
    return best


def _inner_outside(vlist, outer):
    m = 0.0
    for v in vlist:
        for i in range(6):
            p1 = outer[i]
            p2 = outer[(i + 1) % 6]
            e = p2 - p1
            cr = e[0] * (v[1] - p1[1]) - e[1] * (v[0] - p1[0])
            if cr < m:
                m = cr
    return m


def _evaluate(inner_xy, s, ocx, ocy):
    # unit edges
    for i in range(12):
        v = _hex_verts(inner_xy[i, 0], inner_xy[i, 1], 1.0, 0.0)
        for k in range(6):
            d = np.hypot(v[(k + 1) % 6, 0] - v[k, 0], v[(k + 1) % 6, 1] - v[k, 1])
            if abs(d - 1.0) > 1e-8:
                return float('inf')
    vlist = [_hex_verts(inner_xy[i, 0], inner_xy[i, 1], 1.0, 0.0) for i in range(12)]
    sep = _separation_all(vlist)
    if sep <= 1e-6:
        return float('inf')
    outer = _hex_verts(ocx, ocy, s, 0.0)
    out = _inner_outside(vlist, outer)
    if out < -1e-6:
        return float('inf')
    return 1.0 / s


def _required_s(inner_xy):
    best = 0.0
    for i in range(12):
        v = _hex_verts(inner_xy[i, 0], inner_xy[i, 1], 1.0, 0.0)
        for k in range(6):
            cr = np.cross(v[k], v[(k + 1) % 6])
            if cr < best:
                best = cr
    return best / np.sqrt(3.0)


def hexagon_packing_12():
    # 12 unit hexagons: center, 6 ring (r=2.5), 6 outer (r=4.0)
    inner_xy = np.zeros((12, 2))
    inner_xy[0] = [0.0, 0.0]
    for i in range(6):
        a = np.deg2rad(60.0 * i)
        inner_xy[1 + i] = [2.5 * np.cos(a), 2.5 * np.sin(a)]
    for i in range(6):
        a = np.deg2rad(60.0 * i)
        inner_xy[7 + i] = [4.0 * np.cos(a), 4.0 * np.sin(a)]

    s = _required_s(inner_xy) + 1e-6
    ocx = ocy = 0.0
    inner_data = np.column_stack([inner_xy, np.zeros(12)])

    base = _evaluate(inner_xy, s, ocx, ocy)
    best = (inner_data.copy(), np.array([ocx, ocy, 0.0]), s, base)

    # Bounded local polish on ring radii (preserves 6-fold symmetry, no overlap risk)
    r1, r2 = 2.5, 4.0
    for _ in range(200):
        if time.time() > _DEADLINE - _MARGIN:
            break
        improved = False
        for (i, j, r) in [(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)]:
            for step in (0.005, -0.005):
                r1t = r1 + (step if i == 0 else 0.0)
                r2t = r2 + (step if j == 0 else 0.0)
                if r1t < 2.0 or r2t < 3.0 or r2t > 5.0 or r1t > 3.0:
                    continue
                xy = inner_xy.copy()
                for k in range(6):
                    a = np.deg2rad(60.0 * k)
                    xy[1 + k] = [r1t * np.cos(a), r1t * np.sin(a)]
                    xy[7 + k] = [r2t * np.cos(a), r2t * np.sin(a)]
                st = _required_s(xy) + 1e-6
                val = _evaluate(xy, st, 0.0, 0.0)
                if val < base - 1e-12:
                    base = val
                    r1, r2 = r1t, r2t
                    improved = True
        if not improved:
            break

    if base < best[3]:
        for k in range(6):
            a = np.deg2rad(60.0 * k)
            inner_xy[1 + k] = [r1 * np.cos(a), r1 * np.sin(a)]
            inner_xy[7 + k] = [r2 * np.cos(a), r2 * np.sin(a)]
        s = _required_s(inner_xy) + 1e-6
        inner_data = np.column_stack([inner_xy, np.zeros(12)])
        best = (inner_data, np.array([ocx, ocy, 0.0]), s, base)

    if time.time() > _DEADLINE - _MARGIN:
        return best[0], best[1], best[2]

    return best[0], best[1], best[2]


# EVOLVE-BLOCK-END

"""Packing 11 unit regular hexagons inside a regular hexagon (minimize outer side)."""
import math
import time

N = 11
_T0 = time.time()
_TIME_LIMIT = 135.0          # safety margin before the 150 s kill
_EPS = 1e-9


def _hex_center(i, j):
    x = 1.5 * i
    y = math.sqrt(3.0) / 2.0 * (i + 2.0 * j)
    return (x, y)


def _hex_verts(cx, cy, r, ang):
    c, s = math.cos(ang), math.sin(ang)
    out = []
    for k in range(6):
        a = math.radians(60.0 * k)
        px, py = r * math.cos(a), r * math.sin(a)
        out.append((cx + c * px - s * py, cy + s * px + c * py))
    return out


def _seg_seg(p1, p2, p3, p4):
    """Min distance between two line segments."""
    d1 = (p2[0] - p1[0], p2[1] - p1[1])
    d2 = (p4[0] - p3[0], p4[1] - p3[1])
    r = (p1[0] - p3[0], p1[1] - p3[1])
    a = d1[0] * d1[0] + d1[1] * d1[1]
    e = d2[0] * d2[0] + d2[1] * d2[1]
    f = d2[0] * r[0] + d2[1] * r[1]
    if a < _EPS:
        if e < _EPS:
            dx = p1[0] - p3[0]; dy = p1[1] - p3[1]
            return math.hypot(dx, dy)
        s = 0.0; t = f / e
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        dx = p1[0] - (p3[0] + t * d2[0]); dy = p1[1] - (p3[1] + t * d2[1])
        return math.hypot(dx, dy)
    c = d1[0] * r[0] + d1[1] * r[1]
    if e < _EPS:
        s = c / a; t = 0.0
        if s < 0.0:
            s = 0.0
        elif s > 1.0:
            s = 1.0
        dx = p1[0] + s * d1[0] - p3[0]; dy = p1[1] + s * d1[1] - p3[1]
        return math.hypot(dx, dy)
    b = d1[0] * d2[0] + d1[1] * d2[1]
    den = a * e - b * b
    if den < _EPS:
        s = 0.0; t = f / e
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        dx = p1[0] - (p3[0] + t * d2[0]); dy = p1[1] - (p3[1] + t * d2[1])
        return math.hypot(dx, dy)
    s = (b * f - c * e) / den
    if s < 0.0:
        s = 0.0; t = f / e
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        dx = p1[0] - (p3[0] + t * d2[0]); dy = p1[1] - (p3[1] + t * d2[1])
        return math.hypot(dx, dy)
    elif s > 1.0:
        s = 1.0; t = (f + b) / e
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        dx = p1[0] + d1[0] - (p3[0] + t * d2[0]); dy = p1[1] + d1[1] - (p3[1] + t * d2[1])
        return math.hypot(dx, dy)
    else:
        t = (b * c - a * f) / den
        dx = p1[0] + s * d1[0] - (p3[0] + t * d2[0])
        dy = p1[1] + s * d1[1] - (p3[1] + t * d2[1])
        return math.hypot(dx, dy)


def _hex_gap(v1, v2):
    """Minimum distance between two unit hexagons (0 if they overlap)."""
    best = 1e18
    n1 = len(v1); n2 = len(v2)
    for a in range(n1):
        for b in range(n2):
            d = _seg_seg(v1[a], v1[(a + 1) % n1], v2[b], v2[(b + 1) % n2])
            if d < best:
                best = d
    return best


def _gap_table():
    """Exact minimum gap between two lattice unit hexagons for index offsets."""
    g = {}
    for i in range(-3, 4):
        for j in range(-3, 4):
            if i == 0 and j == 0:
                continue
            v1 = _hex_verts(*_hex_center(0, 0), 1.0, 0.0)
            v2 = _hex_verts(*_hex_center(i, j), 1.0, 0.0)
            g[(i, j)] = _hex_gap(v1, v2)
    return g


def _build_sets(g):
    """All 11-subsets of the lattice with a valid (non-overlapping) inner packing."""
    keys = list(g.keys())
    m = len(keys)
    sets = []
    for mask in range(1 << m):
        if bin(mask).count("1") != N:
            continue
        idx = [k for k in range(m) if (mask >> k) & 1]
        ok = True
        for a in range(N):
            for b in range(a + 1, N):
                di = keys[idx[a]][0] - keys[idx[b]][0]
                dj = keys[idx[a]][1] - keys[idx[b]][1]
                if g[(di, dj)] <= _EPS:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            sets.append(idx)
    return sets


def _outer_req(idx, g):
    """Smallest outer side S so every hexagon fits (ignoring rotation)."""
    req = 1.0
    for a in range(N):
        for b in range(a + 1, N):
            di = keys_idx[idx[a]][0] - keys_idx[idx[b]][0]
            dj = keys_idx[idx[a]][1] - keys_idx[idx[b]][1]
            gap = g[(di, dj)]
            d = math.hypot(1.5 * di, math.sqrt(3.0) / 2.0 * (di + 2.0 * dj))
            r = (d - gap) / 2.0
            if r > req:
                req = r
    return req + 1.0


# Precompute lattice data and valid inner packings.
_G = _gap_table()
_KEYS = list(_G.keys())
keys_idx = _KEYS
_VALID = _build_sets(_G)

# Precompute per-set center data and required outer radius (r = S - 1).
_SETS = []
for idx in _VALID:
    centers = [_hex_center(_KEYS[idx[a]][0], _KEYS[idx[a]][1]) for a in range(N)]
    req = _outer_req(idx, _G)
    _SETS.append((req, centers))

# Sort by required outer side (smallest first).
_SETS.sort(key=lambda t: t[0])


def construct():
    # Best feasible configuration: outer centered at origin, flat-top, angle 0.
    best_S = None
    best_inner = None
    for req, centers in _SETS:
        S = req
        inner = [[centers[a][0], centers[a][1], 0.0] for a in range(N)]
        if best_S is None or S < best_S - 1e-12:
            best_S = S
            best_inner = inner
        if time.time() - _T0 > _TIME_LIMIT:
            break

    if best_inner is None:
        # Fallback: a guaranteed-valid (though large) linear arrangement.
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_inner, (0.0, 0.0), best_S, 0.0

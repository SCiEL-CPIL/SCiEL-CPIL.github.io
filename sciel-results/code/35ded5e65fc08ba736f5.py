# EVOLVE-BLOCK-START
import time

import numpy as np

# ---------------- geometry helpers ----------------
# A hexagon (cx, cy, s, a_deg) has vertices v_i = (cx + s*cos(a + 60i), cy + s*sin(a + 60i)).
# For s = 1, the 6 unit outward edge normals are at angles 30 + 60k (pointing away from center),
# and the apothem (distance from center to each edge) is 1.
_NORMALS = np.array([[np.cos(np.radians(30 + 60 * k)), np.sin(np.radians(30 + 60 * k))] for k in range(6)])
_ANGS = np.array([np.radians(60 * i) for i in range(6)])
_COS = np.cos(_ANGS)
_SIN = np.sin(_ANGS)
_UNIT_VERTS = np.stack([_COS, _SIN], axis=1)  # (6,2) unit hexagon vertices (angle 0)
_N3 = np.array([np.sqrt(3.0) / 2.0, 0.0])  # unit vector along the x-axis (a hex edge direction)


def _verts(cx, cy, ang_deg):
    a = np.radians(ang_deg)
    return np.stack([cx + _COS * np.cos(a) - _SIN * np.sin(a),
                     cy + _SIN * np.cos(a) + _COS * np.sin(a)], axis=1)


def _hex_separated(a, b, eps=1e-6):
    """True if two unit hexagons are separated by > eps on at least one of the 12 edge normals."""
    ra = _verts(a[0], a[1], a[2])
    rb = _verts(b[0], b[1], b[2])
    for k in range(6):
        n = _NORMALS[k]
        pa = ra @ n
        pb = rb @ n
        if (pa.min() - pb.max() > eps) or (pb.min() - pa.max() > eps):
            return True
    return False


def _pairwise_ok(p, eps=1e-6):
    n = len(p)
    for i in range(n):
        for j in range(i + 1, n):
            if not _hex_separated(p[i], p[j], eps):
                return False
    return True


def _enclosing_side(p):
    """Exact minimal side length of a regular hexagon (orientation 0, centered at origin) enclosing all unit hexagons."""
    side = 0.0
    for (cx, cy, ang) in p:
        a = np.radians(ang)
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        verts = cx + _UNIT_VERTS @ rot.T + cy
        proj = verts @ _NORMALS.T  # (6,6)
        side = max(side, (proj.max(axis=0) + 1.0).max())
    return side


def _build_valid(p):
    """Return (inner, outer, side) if p is a valid packing, else None."""
    if not np.all(np.isfinite(p)):
        return None
    if not _pairwise_ok(p):
        return None
    side = _enclosing_side(p)
    if side < np.sqrt(11.0) - 1e-9:
        return None
    inner = np.asarray(p, dtype=float)
    outer = np.array([0.0, 0.0, 0.0])
    return inner, outer, float(side)


# ---------------- initial configuration ----------------
def _initial_positions():
    d = np.sqrt(3.0)
    base = [
        (0.0, 0.0, 0.0),
        (1.5, 0.0, 0.0), (-1.5, 0.0, 0.0),
        (0.75, d / 2.0, 0.0), (-0.75, d / 2.0, 0.0),
        (0.75, -d / 2.0, 0.0), (-0.75, -d / 2.0, 0.0),
        (3.0, d / 2.0, 0.0), (3.0, -d / 2.0, 0.0),
        (-3.0, d / 2.0, 0.0), (-3.0, -d / 2.0, 0.0),
    ]
    return [list(v) for v in base]


# ---------------- local optimizer ----------------
def _optimize(p, deadline):
    """Local optimization: minimize enclosing side while keeping hexagons separated."""
    p = [list(v) for v in p]
    step = 0.05
    best_side = _enclosing_side(p)
    best = [list(v) for v in p]
    rng = np.random.default_rng(20240607)
    while time.time() < deadline:
        improved = False
        for it in range(400):
            if time.time() >= deadline:
                break
            i = rng.integers(0, len(p))
            axis = rng.integers(0, 3)
            sgn = 1.0 if rng.random() < 0.5 else -1.0
            cand = [list(v) for v in p]
            cand[i][axis] += sgn * step
            if not np.all(np.isfinite(cand)):
                continue
            if not _pairwise_ok(cand):
                continue
            side = _enclosing_side(cand)
            if side < best_side - 1e-12:
                best_side = side
                best = [list(v) for v in cand]
                p = cand
                improved = True
        if not improved:
            if step < 1e-4:
                break
            step *= 0.5
    return best, best_side


# ---------------- main ----------------
def hexagon_packing_11():
    deadline = time.time() + 590.0  # safety margin before the 630s kill
    p = _initial_positions()
    res = _build_valid(p)
    best = res if res is not None else (None, None, float('inf'))
    best_side = best[2]

    # Local optimization from the symmetric seed.
    opt, opt_side = _optimize(p, min(time.time() + 300.0, deadline - 60.0))
    if opt_side < best_side:
        r = _build_valid(opt)
        if r is not None:
            best = r
            best_side = r[2]

    # Random restarts with local optimization, if budget allows.
    rng = np.random.default_rng(7)
    while time.time() < deadline - 90.0:
        base = _initial_positions()
        for i in range(len(base)):
            base[i][0] += rng.uniform(-0.12, 0.12)
            base[i][1] += rng.uniform(-0.12, 0.12)
            base[i][2] += rng.uniform(-12.0, 12.0)
        if time.time() >= deadline - 90.0:
            break
        opt, opt_side = _optimize(base, min(time.time() + 60.0, deadline - 90.0))
        if opt_side < best_side:
            r = _build_valid(opt)
            if r is not None:
                best = r
                best_side = r[2]

    # Fallback to the seed (guaranteed valid) if nothing better was found.
    if best[0] is None:
        d = np.sqrt(3.0)
        fallback = [
            [0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [-1.5, 0.0, 0.0],
            [0.75, d / 2.0, 0.0], [-0.75, d / 2.0, 0.0],
            [0.75, -d / 2.0, 0.0], [-0.75, -d / 2.0, 0.0],
            [3.0, d / 2.0, 0.0], [3.0, -d / 2.0, 0.0],
            [-3.0, d / 2.0, 0.0], [-3.0, -d / 2.0, 0.0],
        ]
        best = _build_valid(fallback)
        best_side = best[2]

    return best[0], best[1], best[2]


# EVOLVE-BLOCK-END

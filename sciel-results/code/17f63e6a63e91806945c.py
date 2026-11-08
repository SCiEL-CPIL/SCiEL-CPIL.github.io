"""Packing 12 unit regular hexagons into the smallest enclosing regular hexagon.

Key insight: any subset of distinct cells from a regular hexagon tiling is
guaranteed to be disjoint (edge-to-edge).  So we can combinatorially search
compact 12-cell blobs of the tiling, and for each blob solve an exact linear
program for the smallest enclosing regular hexagon, sweeping the outer
hexagon's rotation.  This is far tighter than the trivial seed layout.
"""
import time
import math
import random
import numpy as np
from scipy.optimize import linprog

N = 12
SQ3 = math.sqrt(3.0)
START = time.time()
DEADLINE = START + 130.0          # safety margin before the 150 s wall-clock kill
MARGIN = 1e-4                     # tiny safety margin on the outer side length

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _hex_vertices(px, py, ang_deg):
    """Vertices of a unit regular hexagon (flat-top, vertex at angle 0)."""
    a = math.radians(ang_deg)
    out = []
    for k in range(6):
        th = a + k * math.pi / 3.0
        out.append((px + math.cos(th), py + math.sin(th)))
    return out

def _axial_to_pixel(q, r):
    """Flat-top hex tiling, side length 1."""
    x = 1.5 * q
    y = SQ3 * (r + 0.5 * q)
    return x, y

def _rot_xy(x, y, ang_deg):
    a = math.radians(ang_deg)
    c, s = math.cos(a), math.sin(a)
    return (c * x - s * y, s * x + c * y)

# ---------------------------------------------------------------------------
# Minimal enclosing regular hexagon (fixed orientation 0) via linear program.
# Returns (side, cx, cy) in the frame where the points are given.
# ---------------------------------------------------------------------------
def _enclosing_hex(points):
    """points: list of (x, y).  Outer hexagon is flat-top (0 deg)."""
    if not points:
        return (0.0, 0.0, 0.0)
    A = []
    b = []
    for (x, y) in points:
        # 1: cx - x <= s
        A.append([1.0, 0.0, -1.0]); b.append(x)
        # 2: -cx + x <= s
        A.append([-1.0, 0.0, -1.0]); b.append(-x)
        # 3: cy - y <= (sqrt3/2) s
        A.append([0.0, 1.0, -0.5 * SQ3]); b.append(y)
        # 4: -cy + y <= (sqrt3/2) s
        A.append([0.0, -1.0, -0.5 * SQ3]); b.append(-y)
        # 5: sqrt3(cx-x) + (cy-y) <= sqrt3 s
        A.append([SQ3, 1.0, -SQ3]); b.append(SQ3 * x + y)
        # 6: -sqrt3(cx-x) + (cy-y) <= sqrt3 s
        A.append([-SQ3, 1.0, -SQ3]); b.append(-SQ3 * x + y)
        # 7: sqrt3(cx-x) - (cy-y) <= sqrt3 s
        A.append([SQ3, -1.0, -SQ3]); b.append(SQ3 * x - y)
        # 8: -sqrt3(cx-x) - (cy-y) <= sqrt3 s
        A.append([-SQ3, -1.0, -SQ3]); b.append(-SQ3 * x - y)

    res = linprog(c=[0.0, 0.0, 1.0],
                  A_ub=np.array(A), b_ub=np.array(b),
                  bounds=[(None, None), (None, None), (0.0, None)],
                  method='highs')
    if res.status != 0:
        return (None, None, None)
    cx, cy, s = res.x
    return (float(s), float(cx), float(cy))

def _best_for_blob(centers, orients):
    """Sweep outer rotation; return (side, phi_deg, cx_orig, cy_orig)."""
    verts = []
    for (px, py), ao in zip(centers, orients):
        verts.extend(_hex_vertices(px, py, ao))

    best = (None, None, None, None)
    # coarse sweep 0..60 deg (hexagon has 60 deg symmetry)
    for phi in range(0, 60, 2):
        rp = [_rot_xy(x, y, -phi) for (x, y) in verts]
        s, cx, cy = _enclosing_hex(rp)
        if s is None:
            continue
        if best[0] is None or s < best[0]:
            ox, oy = _rot_xy(cx, cy, phi)
            best = (s, float(phi), ox, oy)
    # refine around the coarse best
    if best[0] is not None:
        base = best[1]
        for dphi in np.arange(-3.0, 3.0 + 1e-9, 0.5):
            phi = base + dphi
            if phi < 0:
                phi += 60.0
            rp = [_rot_xy(x, y, -phi) for (x, y) in verts]
            s, cx, cy = _enclosing_hex(rp)
            if s is None:
                continue
            if s < best[0]:
                ox, oy = _rot_xy(cx, cy, phi)
                best = (s, float(phi), ox, oy)
    return best

# ---------------------------------------------------------------------------
# Candidate blob generation (all are subsets of the tiling -> disjoint).
# ---------------------------------------------------------------------------
_NEIGH = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]

def _make_pool():
    pool = {}
    for q in range(-7, 8):
        for r in range(-7, 8):
            pool[(q, r)] = _axial_to_pixel(q, r)
    return pool

def _rectangle_blob(q0, q1, r0, r1, pool):
    cells = []
    for q in range(q0, q1 + 1):
        for r in range(r0, r1 + 1):
            if (q, r) in pool:
                cells.append((q, r))
    return cells

def _random_connected_blob(pool, rng):
    cells = set()
    start = rng.choice(list(pool.keys()))
    cells.add(start)
    frontier = [start]
    while len(cells) < N:
        if not frontier:
            # restart
            return None
        cur = frontier.pop()
        nbrs = [n for n in _NEIGH
                if (cur[0] + n[0], cur[1] + n[1]) in pool
                and (cur[0] + n[0], cur[1] + n[1]) not in cells]
        if not nbrs:
            continue
        pick = rng.choice(nbrs)
        ncell = (cur[0] + pick[0], cur[1] + pick[1])
        cells.add(ncell)
        frontier.append(ncell)
        # keep frontier modest
        if len(frontier) > 40:
            frontier = frontier[-40:]
    return list(cells)

# ---------------------------------------------------------------------------
def construct():
    rng = random.Random(12345)
    pool = _make_pool()

    candidates = []

    # Systematic parallelogram rectangles of area 12
    for (qa, qb, ra, rb) in [(0, 2, 0, 3), (0, 3, 0, 2),
                             (-1, 1, -1, 3), (-2, 1, -2, 2),
                             (-1, 2, -2, 1), (0, 1, -2, 4)]:
        c = _rectangle_blob(qa, qb, ra, rb, pool)
        if len(c) == N:
            candidates.append(c)

    # Compact "hexagon radius 1" (7) plus a contiguous arc of ring-2 cells
    hex1 = [(0, 0)] + [(q, r) for (q, r) in
                       [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]]
    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -1),
             (-2, 0), (-2, 1), (-2, 2), (-1, 2), (0, 2), (1, 1)]
    ring2 = [c for c in ring2 if c in pool]
    if len(hex1) == 7 and len(ring2) >= 5:
        for i in range(len(ring2)):
            arc = ring2[i:i + 5]
            if len(arc) < 5:
                arc += ring2[:5 - len(arc)]
            blob = hex1 + arc[:5]
            if len(blob) == N:
                candidates.append(blob)

    # Random connected blobs (time limited)
    tried = 0
    while time.time() < DEADLINE - 30.0 and tried < 4000:
        blob = _random_connected_blob(pool, rng)
        if blob is None:
            tried += 1
            continue
        key = frozenset(blob)
        if key not in {frozenset(c) for c in candidates}:
            candidates.append(blob)
        tried += 1
        if time.time() > DEADLINE - 30.0:
            break

    best = None  # (side, centers, orients, phi, cx, cy)
    for cells in candidates:
        if time.time() > DEADLINE - 20.0:
            break
        centers = [pool[c] for c in cells]
        orients = [0.0] * N
        res = _best_for_blob(centers, orients)
        if res[0] is None:
            continue
        if best is None or res[0] < best[0]:
            best = (res[0], centers, orients, res[1], res[2], res[3])

    if best is None:
        # safe fallback: a big hexagon that certainly contains everything
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    side, centers, orients, phi, cx, cy = best
    inner = [[float(px), float(py), float(ao)] for (px, py), ao in zip(centers, orients)]
    return inner, (float(cx), float(cy)), float(side) + MARGIN, float(phi)

# EVOLVE-BLOCK-END

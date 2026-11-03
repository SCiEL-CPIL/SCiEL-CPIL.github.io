"""Packing 12 unit hexagons in a regular hexagon: analytic hex-lattice block + local search."""
import math
import time
import random

N = 12
DEADLINE = 150.0
_START = time.time()


def _time_left():
    return DEADLINE - (time.time() - _START)


def hex_verts(cx, cy, ang_deg, s=1.0):
    a = math.radians(ang_deg)
    ca, sa = math.cos(a), math.sin(a)
    out = []
    for k in range(6):
        t = math.radians(60.0 * k)
        px, py = s * math.cos(t), s * math.sin(t)
        rx = ca * px - sa * py
        ry = sa * px + ca * py
        out.append((cx + rx, cy + ry))
    return out


def _separating_axis(A, B):
    for P in (A, B):
        n = len(P)
        for i in range(n):
            p1 = P[i]
            p2 = P[(i + 1) % n]
            ax = p2[1] - p1[1]
            ay = -(p2[0] - p1[0])
            minA = maxA = None
            minB = maxB = None
            for (x, y) in A:
                d = x * ax + y * ay
                if minA is None or d < minA:
                    minA = d
                if maxA is None or d > maxA:
                    maxA = d
            for (x, y) in B:
                d = x * ax + y * ay
                if minB is None or d < minB:
                    minB = d
                if maxB is None or d > maxB:
                    maxB = d
            if maxA <= minB or maxB <= minA:
                return False
    return True


def _polygons_overlap(A, B):
    return _separating_axis(A, B)


def _in_hex(cx, cy, ox, oy, S, rot_deg, eps=1e-9):
    dx = cx - ox
    dy = cy - oy
    a = math.radians(-rot_deg)
    ca, sa = math.cos(a), math.sin(a)
    x = ca * dx - sa * dy
    y = sa * dx + ca * dy
    r = math.sqrt(3.0) * S
    hw = r
    hh = 0.5 * r
    if abs(x) > hw + eps or abs(y) > hh + eps:
        return False
    if abs(x) / hw + abs(y) / hh > 1.0 + eps:
        return False
    return True


def _valid(inner, center, side, rot_deg, eps=1e-9):
    ox, oy = center
    verts = [hex_verts(x, y, ang, 1.0) for (x, y, ang) in inner]
    for v in verts:
        for (vx, vy) in v:
            if not _in_hex(vx, vy, ox, oy, side, rot_deg, eps):
                return False
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            if _polygons_overlap(verts[i], verts[j], eps):
                return False
    return True


def _hex_lattice_block():
    cells = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) > 2:
                continue
            if (q, r) in ((2, 0), (0, 2), (-2, -2)):
                continue
            cells.append((q, r))
    pts = []
    for (q, r) in cells:
        x = math.sqrt(3.0) * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
    return pts


# EVOLVE-BLOCK-START
def construct():
    if _time_left() < 5.0:
        pts = _hex_lattice_block()
        inner = [[x, y, 0.0] for (x, y) in pts]
        return inner, (0.0, 0.0), 4.0, 0.0

    rng = random.Random(12345)
    pts = _hex_lattice_block()
    base_inner = [[x, y, 0.0] for (x, y) in pts]
    assert _valid(base_inner, (0.0, 0.0), 4.0, 0.0), "base block must be valid"

    best_inner = [list(v) for v in base_inner]
    best_center = (0.0, 0.0)
    best_rot = 0.0
    best_side = 4.0

    # Find a valid lower side for the base block
    for s in [4.0, 3.9, 3.8, 3.7, 3.6, 3.5, 3.4, 3.3, 3.2, 3.1]:
        if _valid(base_inner, (0.0, 0.0), s, 0.0):
            best_side = s
            break

    cur_inner = [list(v) for v in base_inner]
    cur_center = (0.0, 0.0)
    cur_rot = 0.0
    cur_side = best_side

    step = 0.06
    for _ in range(30000):
        if _time_left() < 10.0:
            break
        idx = rng.randrange(N)
        dx = rng.uniform(-step, step)
        dy = rng.uniform(-step, step)
        da = rng.uniform(-20.0, 20.0)
        old = list(cur_inner[idx])
        cur_inner[idx][0] += dx
        cur_inner[idx][1] += dy
        cur_inner[idx][2] += da
        rot_try = cur_rot + (rng.uniform(-4.0, 4.0) if rng.random() < 0.05 else 0.0)
        if _valid(cur_inner, cur_center, cur_side, rot_try, 1e-9):
            cur_rot = rot_try
            new_side = cur_side - 0.02
            if _valid(cur_inner, cur_center, new_side, cur_rot, 1e-9):
                cur_side = new_side
                if cur_side < best_side:
                    best_side = cur_side
                    best_inner = [list(v) for v in cur_inner]
                    best_center = cur_center
                    best_rot = cur_rot
            if _time_left() < 10.0:
                break
        else:
            cur_inner[idx] = old

    if not _valid(best_inner, best_center, best_side, best_rot, 1e-9):
        best_inner = [list(v) for v in base_inner]
        best_center = (0.0, 0.0)
        best_rot = 0.0
        best_side = 4.0
    return best_inner, best_center, best_side, best_rot
# EVOLVE-BLOCK-END

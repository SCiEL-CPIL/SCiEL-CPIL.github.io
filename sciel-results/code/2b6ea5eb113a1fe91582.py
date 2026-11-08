"""Packing unit hexagons inside a minimal regular hexagon.

Design (from scratch):
  * Place the n inner unit hexagons on a hexagonal (close-packed) lattice,
    which is the densest 2D arrangement of equal bodies.
  * Orient the whole cluster so that its three principal axes align with the
    six flat-face normals of the outer container.  For a regular hexagon the
    width along a flat normal is 2*cos(theta), so the enclosing side length is
    L = 2*cos(theta)*(1 + (n-1)*sqrt(3)), which is minimised at theta = 60deg
    (pointy orientation) giving  L = 0.5*(1 + (n-1)*sqrt(3)).
  * Verify feasibility (containment + disjointness) and fall back to a safe
    line packing if anything is off.  A wall-clock deadline guard returns the
    best valid result found.
"""
import math
import time

N = 11

_T0 = time.time()
_DEADLINE = 150.0          # evaluator wall-clock limit
_MARGIN = 8.0              # safety margin (seconds)


def _hexagon_vertices(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon (pointy-top at angle 0)."""
    a = math.radians(angle_deg)
    pts = []
    for k in range(6):
        t = a + math.radians(60.0 * k)
        pts.append((cx + side * math.cos(t), cy + side * math.sin(t)))
    return pts


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _segments_intersect(p1, p2, p3, p4):
    d1 = _cross(p3, p4, p1)
    d2 = _cross(p3, p4, p2)
    d3 = _cross(p1, p2, p3)
    d4 = _cross(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """True if (px,py) is inside (or on) the hexagon.  Convex -> all cross same sign."""
    pts = _hexagon_vertices(cx, cy, side, angle_deg)
    pos = neg = False
    for k in range(6):
        c = _cross(pts[k], pts[(k + 1) % 6], (px, py))
        if c > 1e-9:
            pos = True
        elif c < -1e-9:
            neg = True
        if pos and neg:
            return False
    return True


def _poly_overlap(poly_a, poly_b):
    """True if two convex polygons share interior area."""
    for pa in poly_a:
        if _point_in_hexagon(pa[0], pa[1], *poly_b[:2], poly_b[2], poly_b[3]):
            return True
    for pb in poly_b:
        if _point_in_hexagon(pb[0], pb[1], *poly_a[:2], poly_a[2], poly_a[3]):
            return True
    for i in range(6):
        for j in range(6):
            if _segments_intersect(poly_a[i], poly_a[(i + 1) % 6],
                                   poly_b[j], poly_b[(j + 1) % 6]):
                return True
    return False


def _pack_line():
    """Safe fallback: unit hexagons in a single row, large container."""
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0


def _pack_hexagonal():
    """Hexagonal close-packed lattice, pointy orientation (theta=60deg)."""
    R = math.sqrt(3.0)            # lattice spacing between adjacent centers
    rows = []
    for row in range(-N, N + 1):
        count = N - abs(row)
        if count <= 0:
            continue
        x0 = -(count - 1) * 0.5
        for col in range(count):
            x = x0 + col * 1.0
            y = row * R
            rows.append((x, y))
    # keep the N closest to the origin (lattice is symmetric -> exactly N)
    rows.sort(key=lambda p: (p[0] * p[0] + p[1] * p[1], p[1], p[0]))
    rows = rows[:N]

    theta = 60.0                   # pointy orientation -> minimal enclosing hex
    L = 0.5 * (1.0 + (N - 1) * math.sqrt(3.0))

    inner = [[float(x), float(y), float(theta)] for (x, y) in rows]
    return inner, (0.0, 0.0), float(L), 0.0


def _feasible(inner, center, side, angle):
    """Containment + pairwise disjointness check."""
    cx, cy = center
    outer = _hexagon_vertices(cx, cy, side, angle)
    polys = [_hexagon_vertices(x, y, 1.0, a) for (x, y, a) in inner]
    for poly in polys:
        for v in poly:
            if not _point_in_hexagon(v[0], v[1], cx, cy, side, angle):
                return False
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if _poly_overlap(polys[i], polys[j]):
                return False
    return True


def construct():
    best = None
    if time.time() - _T0 > _DEADLINE - _MARGIN:
        return _pack_line()

    candidates = [_pack_hexagonal(), _pack_line()]
    for cand in candidates:
        if time.time() - _T0 > _DEADLINE - _MARGIN:
            break
        if _feasible(*cand):
            best = cand
            break
    if best is None:
        # hexagonal failed feasibility (should not happen); use safe line
        best = _pack_line()
    return best

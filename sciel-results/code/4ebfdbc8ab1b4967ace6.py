"""Fresh design: fixed hexagonal-cluster topology + global-rotation optimization.

A regular hexagon of side s is the Minkowski sum of 3 segments (length 2s) at
0/60/120 deg.  The minimum enclosing hexagon of a point set P, with orientation
alpha, has side s(alpha) = max_i width_i(alpha)/2 where width_i is the width of
P along direction i (0/60/120 + alpha).  So the outer side is the max of three
directional widths -- a piecewise-linear function of the global angle and
linear in the centering translation.

We fix a compact hexagonal-cluster topology for the 11 unit hexagons (lattice
centers) and optimize only the global rotation + center.  This is a 1-D
(rotation) / 2-D (center) problem with a well-defined periodic structure,
rather than 33 free per-body coordinates.
"""
import math
import time

N = 11
_T0 = time.time()
_DEADLINE = 150.0
_MARGIN = 5.0

# ---------------------------------------------------------------------------
# Geometry of a unit regular hexagon (flat-top, vertices at angles 0,60,...)
# ---------------------------------------------------------------------------
_HEX_VERT = [(math.cos(math.radians(a)), math.sin(math.radians(a)))
             for a in range(0, 360, 60)]
_HEX_VERT = [(x, y) for x, y in _HEX_VERT]
_HEX_VERT_ARR = [complex(x, y) for x, y in _HEX_VERT]


def _hex_rotated(theta):
    """Vertices of a unit hexagon rotated by theta (radians)."""
    c, s = math.cos(theta), math.sin(theta)
    return [(x * c - y * s, x * s + y * c) for x, y in _HEX_VERT]


def _hex_rotated_c(theta):
    c = complex(math.cos(theta), math.sin(theta))
    return [z * c for z in _HEX_VERT_ARR]


# ---------------------------------------------------------------------------
# Cluster topologies: list of lattice center offsets (in unit-hex spacing).
# A flat-top unit hexagon tiles with center offsets (3/2, +/-sqrt(3)/2) and
# (0, sqrt(3)) in its own frame.  We generate compact clusters and let the
# optimizer choose the best global rotation.
# ---------------------------------------------------------------------------
def _make_cluster():
    """Return a list of (x, y) center offsets for a compact 11-hex cluster."""
    a = 1.5          # horizontal spacing (3/2)
    b = math.sqrt(3) / 2.0  # vertical half-spacing
    # axial coordinates (q, r) -> cartesian in the hex lattice frame
    def ax(q, r):
        return (a * q + a / 2.0 * r, b * r)
    # A compact 11-cell hexagon of radius 1 (center + 6 + 6) has 19 cells; we
    # need 11.  Use a radius-1 hexagon (19) and trim, or a "plus" shape.
    # Radius-1 hexagon cells (axial, |q|+|r|+|q+r|<=1):
    cells = []
    for q in range(-1, 2):
        for r in range(-1, 2):
            if abs(q) + abs(r) + abs(q + r) <= 1:
                cells.append((q, r))
    # That's 7 cells.  Extend to 11 with a second ring partially.
    # Add the 6 ring-2 cells along the axes to reach 11.
    extra = [(2, 0), (0, 2), (-2, 0), (0, -2), (2, -2), (-2, 2)]
    for e in extra:
        cells.append(e)
    pts = [ax(q, r) for (q, r) in cells]
    return pts


def _make_cluster2():
    """A 2-row / 3-row brick-like cluster (11 cells)."""
    a = 1.5
    b = math.sqrt(3) / 2.0
    pts = []
    # row 0: 5 cells
    for i in range(5):
        pts.append((a * i, 0.0))
    # row 1: 3 cells offset
    for i in range(3):
        pts.append((a * (i + 0.5), b))
    # row 2: 3 cells offset
    for i in range(3):
        pts.append((a * (i + 0.5), -b))
    return pts


def _make_cluster3():
    """A 4-3-4 ladder cluster (11 cells)."""
    a = 1.5
    b = math.sqrt(3) / 2.0
    pts = []
    for i in range(4):
        pts.append((a * i, 0.0))
    for i in range(3):
        pts.append((a * (i + 0.5), b))
    for i in range(4):
        pts.append((a * i, 2 * b))
    return pts


# ---------------------------------------------------------------------------
# Objective: given cluster centers (in cluster frame), a global rotation theta,
# and a centering offset, compute the minimum enclosing hexagon side.
# The enclosing hexagon is axis-aligned in the "world" frame (alpha=0), so the
# inner cluster is rotated by theta.
# ---------------------------------------------------------------------------
def _enclosing_side(centers, theta):
    """Minimum enclosing hexagon side for points `centers` rotated by theta,
    enclosing hexagon axis-aligned (orientation 0).  Returns (side, cx, cy)."""
    c, s = math.cos(theta), math.sin(theta)
    # rotated points
    px = [x * c - y * s for x, y in centers]
    py = [x * s + y * c for x, y in centers]
    # three directions: 0, 60, 120 deg
    dirs = [(math.cos(math.radians(d)), math.sin(math.radians(d)))
            for d in (0, 60, 120)]
    widths = []
    for (dx, dy) in dirs:
        proj = [x * dx + y * dy for x, y in zip(px, py)]
        widths.append(max(proj) - min(proj))
    side = max(widths) / 2.0
    # center the cluster at origin
    cx = sum(px) / len(px)
    cy = sum(py) / len(py)
    return side, cx, cy


def _build_solution(cluster, theta):
    """Return (inner, center, side, angle) for a cluster rotated by theta,
    centered at origin, enclosing hexagon axis-aligned."""
    c, s = math.cos(theta), math.sin(theta)
    inner = []
    for (x, y) in cluster:
        rx = x * c - y * s
        ry = x * s + y * c
        inner.append([rx, ry, math.degrees(theta)])
    # center
    cx = sum(p[0] for p in inner) / len(inner)
    cy = sum(p[1] for p in inner) / len(inner)
    inner = [[p[0] - cx, p[1] - cy, p[2]] for p in inner]
    # enclosing side (recompute after centering)
    side, _, _ = _enclosing_side([(p[0], p[1]) for p in inner], 0.0)
    return inner, (0.0, 0.0), side, 0.0


# ---------------------------------------------------------------------------
# Optimize the global rotation theta for a given cluster.
# The side(theta) is periodic (period 60 deg) and piecewise smooth.  We do a
# dense sweep + golden-section refinement on the best interval.
# ---------------------------------------------------------------------------
def _optimize_rotation(cluster):
    best_theta = 0.0
    best_side = float('inf')
    # dense sweep
    n_sweep = 720
    for k in range(n_sweep):
        th = math.radians(60.0 * k / n_sweep)
        side, _, _ = _enclosing_side(cluster, th)
        if side < best_side:
            best_side = side
            best_theta = th
    # golden-section refine around best_theta
    g = (math.sqrt(5.0) - 1.0) / 2.0
    lo = best_theta - math.radians(60.0 / n_sweep)
    hi = best_theta + math.radians(60.0 / n_sweep)
    a = lo + (1 - g) * (hi - lo)
    b = lo + g * (hi - lo)
    fa = _enclosing_side(cluster, a)[0]
    fb = _enclosing_side(cluster, b)[0]
    for _ in range(60):
        if fa < fb:
            hi = b
            b = a
            fb = fa
            a = lo + (1 - g) * (hi - lo)
            fa = _enclosing_side(cluster, a)[0]
        else:
            lo = a
            a = b
            fa = fb
            b = lo + g * (hi - lo)
            fb = _enclosing_side(cluster, b)[0]
    th = (lo + hi) / 2.0
    side, _, _ = _enclosing_side(cluster, th)
    return th, side


# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------
def construct():
    clusters = [_make_cluster(), _make_cluster2(), _make_cluster3()]
    best = None
    for cluster in clusters:
        th, side = _optimize_rotation(cluster)
        if best is None or side < best[1]:
            best = (cluster, th, side)
        if time.time() > _DEADLINE - _MARGIN:
            break
    if best is None:
        # fallback: seed-like row
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    cluster, th, side = best
    return _build_solution(cluster, th)

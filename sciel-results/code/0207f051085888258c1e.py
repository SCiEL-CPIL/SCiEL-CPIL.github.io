"""Pack disjoint unit regular hexagons inside a minimal regular hexagon.

Approach family (distinct from the seed's straight-line placement):
  * Place the N unit hexagons on an *axial* hex grid.  Because neighbouring
    cells are separated by the full inradius (sqrt(3)) while two unit hexagons
    can approach to at most 2*inradius = 2*sqrt(3) apart, any axial placement is
    guaranteed to be disjoint for ANY rotation.  This removes the expensive
    pair-wise collision check.
  * Compact clusters (a hexagonal ring, a 1-2-3-2-1 pyramid, and a 3-4-3
    "diamond") are generated in axial coordinates.
  * For each candidate (and for each of a handful of orientations) the true
    minimum enclosing regular hexagon is found by a small LP: minimise the side
    length s subject to every vertex of every inner hexagon lying inside the
    outer hexagon, which is 6 linear inequalities per vertex.  The LP gives the
    exact optimal outer side length for that fixed inner configuration.
  * The inner hexagons are also rotated by 30 deg so that an edge (not a flat
    face) points toward the container walls, which trims the required side.
"""

import math
import numpy as np
from scipy.optimize import linprog

N = 11

_EPS = 1e-9


# ---- regular-hexagon vertex generator --------------------------------------
def _hex_vertex_offsets(side):
    """Vertex offsets (relative to centre) of a regular hexagon of the given
    side length with vertices at angles 0, 60, ..., 300 degrees.
    (For side == 1 the circumscribed-radius equals the side length.)"""
    ang = np.deg2rad(np.arange(0, 360, 60))
    return np.column_stack((side * np.cos(ang), side * np.sin(ang)))


# ---- axial -> Cartesian ----------------------------------------------------
def _axial_to_xy(a, b, spacing):
    x = spacing * (a + 0.5 * b)
    y = spacing * (math.sqrt(3.0) / 2.0) * b
    return x, y


# ---- axial cluster shapes (all exactly N cells) ----------------------------
def _ring_positions(r):
    """rings 0..r inclusive: a filled hexagon of axial radius r
    (1 + 3r(r+1) cells)."""
    pts = [(0, 0)]
    for ring in range(1, r + 1):
        # walk the hex perimeter of this ring
        corners = [
            (ring, -ring), (2 * ring, -ring), (2 * ring, 0),
            (ring, ring), (0, ring), (-ring, 0),
        ]
        # edges between successive corners
        pts += [(0, 0)]  # placeholder removed below
    pts = [(0, 0)]
    for ring in range(1, r + 1):
        corners = [
            (ring, -ring), (2 * ring, -ring), (2 * ring, 0),
            (ring, ring), (0, ring), (-ring, 0),
        ]
        for i in range(6):
            x0, y0 = corners[i]
            x1, y1 = corners[(i + 1) % 6]
            dx = (x1 - x0) // 1
            dy = (y1 - y0) // 1
            steps = abs(x1 - x0) if abs(x1 - x0) >= abs(y1 - y0) else abs(y1 - y0)
            for s in range(steps):
                pts.append((x0 + dx * s, y0 + dy * s))
    return pts


def _pyramid_positions(nrows):
    """A centred pyramid of odd rows; row i (0-based) has (nrows - 1 - 2*i)
    cells and spans axially a = -w..w at fixed b = i.  Total = nrows^2."""
    pts = []
    for b in range(nrows):
        w = (nrows - 1 - 2 * b) // 2
        for a in range(-w, w + 1):
            pts.append((a, b))
    return pts


def _diamond_positions(m):
    """A centred diamond of axial radius m: |a|+|b| <= m (hexagon in the (a,b)
    plane) -> 3m(m+1)+1 cells."""
    pts = []
    for b in range(-m, m + 1):
        for a in range(-m, m + 1):
            if abs(a) + abs(b) <= m:
                pts.append((a, b))
    return pts


def _cluster_candidates():
    shapes = []
    # ring r=2 -> 19 cells, trim the 8 farthest from the centre
    ring = _ring_positions(2)
    shapes.append(_trim_to_n(ring, N))
    # pyramid 5x5 -> 25 cells, trim the 14 farthest
    pyr = _pyramid_positions(5)
    shapes.append(_trim_to_n(pyr, N))
    # diamond m=2 -> 19 cells, trim the 8 farthest
    dia = _diamond_positions(2)
    shapes.append(_trim_to_n(dia, N))
    return [s for s in shapes if s is not None and len(s) == N]


def _trim_to_n(pts, n):
    if len(pts) < n:
        return None
    if len(pts) == n:
        return pts
    # keep the n closest to the origin (ties broken deterministically)
    order = sorted(range(len(pts)), key=lambda i: (pts[i][0] ** 2 + pts[i][1] ** 2, i))
    keep = set(order[:n])
    return [pts[i] for i in sorted(keep)]


# ---- minimum enclosing regular hexagon via LP -----------------------------
def _min_hexagon_side(vertices, outer_rot_deg):
    """Return the smallest regular-hexagon side length whose centre (free)
    encloses the given vertices.  vertices: (M,2) array."""
    M = vertices.shape[0]
    if M == 0:
        return 0.0
    orad = math.radians(outer_rot_deg)
    # 6 outward normals of the container (pointing to the midpoints of sides)
    nrm = [orad + math.radians(60 * k) for k in range(6)]
    nn = np.array([math.cos(a) for a in nrm])
    ns = np.array([math.sin(a) for a in nrm])

    A = np.empty((6 * M, 5))
    b = np.empty(6 * M)
    for k in range(6):
        r0 = k * M
        A[r0:r0 + M, 0] = nn[k]
        A[r0:r0 + M, 1] = ns[k]
        A[r0:r0 + M, 2] = vertices[:, 0] * nn[k]
        A[r0:r0 + M, 3] = vertices[:, 1] * ns[k]
        A[r0:r0 + M, 4] = -1.0
        b[r0:r0 + M] = 0.0

    res = linprog(np.array([0.0, 0.0, 0.0, 0.0, 1.0]), A_ub=A, b_ub=b,
                  method='highs', options={'presolve': True})
    if res is None or not res.success:
        return math.inf
    return float(res.fun)


# ---- main -------------------------------------------------------------------
def construct():
    hexoff = _hex_vertex_offsets(1.0)          # unit-hexagon vertices
    spacing = math.sqrt(3.0)                   # axial cell spacing == inradius

    best = None
    for shape in _cluster_candidates():
        base_xy = np.array([_axial_to_xy(a, b, spacing) for (a, b) in shape])
        for inner_rot in (0.0, 30.0):
            th = math.radians(inner_rot)
            c, s = math.cos(th), math.sin(th)
            R = np.array([[c, -s], [s, c]])
            verts = base_xy + (hexoff @ R.T)   # all inner-hexagon vertices
            for outer_rot in (0.0, 15.0, 30.0, 45.0):
                side = _min_hexagon_side(verts, outer_rot)
                if not math.isfinite(side):
                    continue
                if best is None or side < best[0] - _EPS:
                    cx = float(base_xy[:, 0].mean())
                    cy = float(base_xy[:, 1].mean())
                    best = (side, cx, cy, verts, base_xy, inner_rot, outer_rot)

    # assemble the result
    side, cx, cy, verts, base_xy, inner_rot, outer_rot = best
    inner = [[float(base_xy[i, 0]), float(base_xy[i, 1]), inner_rot]
             for i in range(base_xy.shape[0])]
    return inner, (cx, cy), float(side), outer_rot

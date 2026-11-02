"""
Hexagon packing: 11 unit hexagons inside a minimal regular hexagon.

Strategy:
  * The 7 hexagons at the centre of a hexagonal ring (a "flower") are fixed
    in a flat-top axial lattice.
  * The remaining 4 hexagons are placed on the boundary of the ring-2 shell.
    For each valid choice of 4 boundary slots we analytically compute the
    minimum enclosing regular hexagon (over centre and rotation) by
    projecting all 66 vertices onto six normal directions.
  * A small local refinement pass nudges the 4 outer hexagons to shave
    off a fraction of a unit of outer side-length.
"""

import math
import time
import itertools
from math import cos, sin, radians, pi

N = 11
S3 = math.sqrt(3)

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _hex_verts(cx, cy, side, ang_rad):
    """Return 6 vertex coordinates of a regular hexagon (flat-top at 0 rad)."""
    out = []
    for k in range(6):
        a = ang_rad + k * pi / 3.0
        out.append((cx + side * cos(a), cy + side * sin(a)))
    return out


def _axial_xy(q, r):
    """Flat-top unit-hexagon axial -> Cartesian (centre position)."""
    x = 1.5 * q
    y = S3 * (r + 0.5 * q)
    return (x, y)


# ---------------------------------------------------------------------------
# Enclosing-hexagon solver
#
# For a set of points and a hexagon orientation theta, the minimum side
# length of a regular hexagon (apothem form) that contains all points,
# with centre free to move, is:
#
#   apothem(theta) = max_k  ( max_i proj_k(p_i) - min_i proj_k(p_i) ) / 2
#
# where proj_k uses the normal at angle  theta + 30° + 60°*k.
#
# The minimum over theta is found by scanning theta in fine steps and
# then polishing with a local golden-section search around the best theta.
# ---------------------------------------------------------------------------

def _apothem_for_theta(points, theta):
    """Return (apothem, centre_x, centre_y) for the minimal enclosing
    hexagon with the given normal-set orientation theta."""
    maxes = [0.0] * 6
    mins  = [0.0] * 6
    for k in range(6):
        a = theta + pi / 6.0 + k * pi / 3.0
        nx, ny = cos(a), sin(a)
        for (px, py) in points:
            v = px * nx + py * ny
            if v > maxes[k]:
                maxes[k] = v
            if v < mins[k]:
                mins[k] = v
    # The hexagon centre is the average of the two support lines for each
    # normal direction; we take the midpoint of the bounding slab.
    cx_acc = 0.0
    cy_acc = 0.0
    ap = 0.0
    for k in range(6):
        a = theta + pi / 6.0 + k * pi / 3.0
        nx, ny = cos(a), sin(a)
        centre_coord = (maxes[k] + mins[k]) / 2.0
        cx_acc += centre_coord * nx
        cy_acc += centre_coord * ny
        half = (maxes[k] - mins[k]) / 2.0
        if half > ap:
            ap = half
    # The true centre is the intersection of the 6 slab mid-planes.
    # For a regular hexagon the 6 mid-planes are consistent; we solve
    # using the first two.
    a0 = theta + pi / 6.0
    a1 = theta + pi / 6.0 + pi / 3.0
    n0x, n0y = cos(a0), sin(a0)
    n1x, n1y = cos(a1), sin(a1)
    c0 = (maxes[0] + mins[0]) / 2.0
    c1 = (maxes[1] + mins[1]) / 2.0
    det = n0x * n1y - n0y * n1x
    if abs(det) < 1e-14:
        cx = cx_acc / 6.0
        cy = cy_acc / 6.0
    else:
        cx = (c0 * n1y - c1 * n0y) / det
        cy = (n0x * c1 - n1x * c0) / det
    return ap, cx, cy


def _min_enclosing_hex(points, theta_init=0.0, n_scan=720):
    """Find the minimum-side regular hexagon enclosing `points`.
    Returns (side, centre_x, centre_y, theta_rad)."""
    # Coarse scan over theta in [0, pi/3)
    best_ap = float('inf')
    best_theta = theta_init
    step = (pi / 3.0) / n_scan
    for i in range(n_scan + 1):
        th = i * step
        ap, _, _ = _apothem_for_theta(points, th)
        if ap < best_ap:
            best_ap = ap
            best_theta = th

    # Local polish: golden-section search in a window around best_theta
    lo = max(0.0, best_theta - 2 * step)
    hi = min(pi / 3.0, best_theta + 2 * step)
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    a = lo + (1 - gr) * (hi - lo)
    b = lo + gr * (hi - lo)
    fa = _apothem_for_theta(points, a)[0]
    fb = _apothem_for_theta(points, b)[0]
    for _ in range(60):
        if fa < fb:
            hi = b
            b = a
            fb = fa
            a = lo + (1 - gr) * (hi - lo)
            fa = _apothem_for_theta(points, a)[0]
        else:
            lo = a
            a = b
            fa = fb
            b = lo + gr * (hi - lo)
            fb = _apothem_for_theta(points, b)[0]
    best_theta = (lo + hi) / 2.0
    best_ap, cx, cy = _apothem_for_theta(points, best_theta)
    side = 2.0 * best_ap / S3
    return side, cx, cy, best_theta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def construct():
    t0 = time.time()
    deadline = t0 + 110.0          # leave 40 s margin

    # --- fixed 7 hexagons: centre + ring-1 --------------------------------
    fixed_axial = [(0, 0),
                   (1, 0), (1, -1), (0, -1),
                   (-1, 0), (-1, 1), (0, 1)]
    fixed_xy = [_axial_xy(q, r) for q, r in fixed_axial]

    # --- ring-2 boundary slots --------------------------------------------
    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2),
             (-1, -2), (-2, -2), (-2, -1), (-2, 0), (-2, 1),
             (-2, 2), (-1, 2), (0, 2), (1, 2)]
    ring2_xy = [_axial_xy(q, r) for q, r in ring2]

    # We need to choose 4 of the 14 ring-2 slots.
    # To keep the search tractable we restrict to combinations where
    # the 4 chosen slots are "spread out" (no two adjacent on the shell).
    # In practice we try all C(14,4)=1001 combos but prune early.

    best_side = float('inf')
    best_inner = None       # list of (x, y)
    best_ocx = best_ocy = 0.0
    best_otheta = 0.0

    # Pre-compute the 42 vertices of the 7 fixed hexagons (side=1, flat-top)
    fixed_verts = []
    for (fx, fy) in fixed_xy:
        fixed_verts.extend(_hex_verts(fx, fy, 1.0, 0.0))

    n_r2 = len(ring2_xy)
    for combo in itertools.combinations(range(n_r2), 4):
        if time.time() > deadline:
            break

        # Build the 24 vertices of the 4 chosen ring-2 hexagons
        r2_verts = []
        for idx in combo:
            (rx, ry) = ring2_xy[idx]
            r2_verts.extend(_hex_verts(rx, ry, 1.0, 0.0))

        all_points = fixed_verts + r2_verts

        # Minimum enclosing hexagon
        side, ocx, ocy, otheta = _min_enclosing_hex(all_points)
        if side < best_side:
            best_side = side
            best_inner = fixed_xy + [ring2_xy[i] for i in combo]
            best_ocx, best_ocy, best_otheta = ocx, ocy, otheta

    # --- local polish: nudge the 4 outer hexagons -------------------------
    if best_inner is not None and time.time() < deadline - 20.0:
        # Try small radial perturbations of the 4 outer hexagons
        # (they sit on ring-2; we allow a small inward shift along the
        #  direction from origin to the hexagon centre).
        polish_iters = 30
        for it in range(polish_iters):
            if time.time() > deadline:
                break
            improved = False
            # Collect the 4 outer hexagon indices
            outer_indices = list(range(7, 11))
            # Random-ish deterministic jitter
            import random
            rng = random.Random(42 + it)
            for oi in outer_indices:
                (hx, hy) = best_inner[oi]
                dist = math.hypot(hx, hy)
                if dist < 1e-9:
                    continue
                ux, uy = hx / dist, hy / dist
                # Try a small inward shift
                for delta in [0.02, 0.04, 0.06]:
                    nx = hx - delta * ux
                    ny = hy - delta * uy
                    trial = list(best_inner)
                    trial[oi] = (nx, ny)
                    # Re-build point set
                    pts = []
                    for (tx, ty) in trial:
                        pts.extend(_hex_verts(tx, ty, 1.0, 0.0))
                    # Quick check: does the current enclosing hex still work?
                    # If not, recompute.
                    ap, _, _ = _apothem_for_theta(pts, best_otheta)
                    side_trial = 2.0 * ap / S3
                    if side_trial < best_side - 1e-10:
                        best_side = side_trial
                        best_inner = trial
                        # Recompute centre for the new theta
                        _, best_ocx, best_ocy, _ = _min_enclosing_hex(pts)
                        improved = True
                        break
            if not improved:
                break

    # --- safety: verify no overlaps and containment -----------------------
    # Disjointness check (pairwise, using vertex-in-hex + edge-intersect)
    inner_hexes = [(x, y, 1.0, 0.0) for (x, y) in best_inner]
    if not _all_disjoint_fast(inner_hexes):
        # Fallback: scale positions outward slightly
        cxm = sum(p[0] for p in best_inner) / len(best_inner)
        cym = sum(p[1] for p in best_inner) / len(best_inner)
        best_inner = [((p[0] - cxm) * 1.01 + cxm,
                       (p[1] - cym) * 1.01 + cym) for p in best_inner]
        inner_hexes = [(x, y, 1.0, 0.0) for (x, y) in best_inner]
        pts = []
        for (tx, ty) in best_inner:
            pts.extend(_hex_verts(tx, ty, 1.0, 0.0))
        best_side, best_ocx, best_ocy, best_otheta = _min_enclosing_hex(pts)

    # Containment check
    outer_side = best_side
    for (hx, hy) in best_inner:
        verts = _hex_verts(hx, hy, 1.0, 0.0)
        for (vx, vy) in verts:
            if not _pt_in_hex(vx, vy, best_ocx, best_ocy, outer_side, best_otheta):
                outer_side *= 1.001
                break

    # --- format output ----------------------------------------------------
    inner = [[p[0], p[1], 0.0] for p in best_inner]
    return inner, (best_ocx, best_ocy), outer_side, math.degrees(best_otheta)


# ---------------------------------------------------------------------------
# Fast collision / containment helpers
# ---------------------------------------------------------------------------

def _pt_in_hex(px, py, cx, cy, side, ang_rad):
    """Is (px,py) inside the regular hexagon (flat-top at 0)?"""
    dx = px - cx
    dy = py - cy
    a = -ang_rad
    lx = dx * cos(a) - dy * sin(a)
    ly = dx * sin(a) + dy * cos(a)
    ap = side * S3 / 2.0
    for k in range(6):
        a_n = pi / 6.0 + k * pi / 3.0
        nx, ny = cos(a_n), sin(a_n)
        if lx * nx + ly * ny > ap + 1e-9:
            return False
    return True


def _seg_cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _segs_intersect(p1, p2, p3, p4):
    d1 = _seg_cross(p3, p4, p1)
    d2 = _seg_cross(p3, p4, p2)
    d3 = _seg_cross(p1, p2, p3)
    d4 = _seg_cross(p1, p2, p4)
    return ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))


def _hex_disjoint(h1, h2):
    (cx1, cy1, s1, a1) = h1
    (cx2, cy2, s2, a2) = h2
    dx = cx2 - cx1
    dy = cy2 - cy1
    if dx * dx + dy * dy > (s1 + s2) ** 2 + 1e-12:
        return True
    v1 = _hex_verts(cx1, cy1, s1, a1)
    v2 = _hex_verts(cx2, cy2, s2, a2)
    for (vx, vy) in v1:
        if _pt_in_hex(vx, vy, cx2, cy2, s2, a2):
            return False
    for (vx, vy) in v2:
        if _pt_in_hex(vx, vy, cx1, cy1, s1, a1):
            return False
    for i in range(6):
        for j in range(6):
            if _segs_intersect(v1[i], v1[(i + 1) % 6],
                               v2[j], v2[(j + 1) % 6]):
                return False
    return True


def _all_disjoint_fast(hexes):
    n = len(hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if not _hex_disjoint(hexes[i], hexes[j]):
                return False
    return True

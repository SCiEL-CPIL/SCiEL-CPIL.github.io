"""Packing 12 unit hexagons in a minimal outer hexagon.

Structural design: start from a hexagonal lattice (densest natural packing),
optimize the outer hexagon rotation/center to enclose the lattice, then run a
penalty-based local refinement of the inner hexagon positions using the
smooth support-function penetration measure (Minkowski difference).
"""
import math
import time
import numpy as np

N = 12
DEADLINE = 135.0  # safety margin before 150s wall-clock kill


def _hex_vertices(cx, cy, side, ang_deg):
    a = math.radians(ang_deg)
    c, s = math.cos(a), math.sin(a)
    pts = []
    for k in range(6):
        th = math.pi / 3.0 * k
        x = side * math.cos(th)
        y = side * math.sin(th)
        pts.append((cx + c * x - s * y, cy + s * x + c * y))
    return pts


def _penetration(v, w):
    """Penetration depth of convex hexagon w into convex hexagon v (>=0 means overlap).

    v, w: (6,2) arrays of vertices in CCW order.
    """
    nv = len(v)
    nw = len(w)
    best = -np.inf
    for i in range(nv):
        a = v[i]
        b = v[(i + 1) % nv]
        e = b - a
        n = np.array([-e[1], e[0]])
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n = n / nn
        # outward normal for CCW polygon
        d = np.dot(w, n) - np.dot(a, n)
        # if all w are on the inside (d <= 0), then max d is the deepest
        maxd = d.max()
        if maxd > best:
            best = maxd
    # best is max over v's edges of (max distance of w beyond edge), negative means separated
    # penetration = -best if best < 0
    return -best


def _inside_violation(p, ocx, ocy, side, ang_deg):
    """Max distance by which point p is outside the outer hexagon (0 if inside)."""
    # Transform p into outer hexagon's local frame
    a = math.radians(ang_deg)
    c, s = math.cos(a), math.sin(a)
    dx = p[0] - ocx
    dy = p[1] - ocy
    lx = c * dx + s * dy
    ly = -s * dx + c * dy
    # Regular hexagon centered at origin, flat orientation: vertices at angles 0,60,...
    # Support: a point (lx,ly) is inside iff for all 6 edge normals, dot <= side*sqrt(3)/2 ... 
    # For a regular hexagon of side s, the distance from center to each edge (apothem) = s*sqrt(3)/2.
    # Edge normals point at angles 30,90,150,210,270,330 degrees.
    apothem = side * math.sqrt(3) / 2.0
    max_v = 0.0
    for k in range(6):
        th = math.pi / 3.0 * k + math.pi / 6.0
        nx = math.cos(th)
        ny = math.sin(th)
        d = lx * nx + ly * ny - apothem
        if d > max_v:
            max_v = d
    return max_v


def _score_of(inner, ocx, ocy, side, ang_deg):
    """Total violation: pairwise overlaps + outside violations. Lower is better; 0 = feasible."""
    verts = [_hex_vertices(x, y, 1.0, a) for (x, y, a) in inner]
    viol = 0.0
    n = len(inner)
    for i in range(n):
        for j in range(i + 1, n):
            p = _penetration(verts[i], verts[j])
            if p > 0:
                viol += p
    for (x, y, a) in inner:
        o = _inside_violation(np.array([x, y]), ocx, ocy, side, ang_deg)
        if o > 0:
            viol += o
    return viol


def _hex_lattice_centers():
    """12 centers in a hexagonal lattice, side-1 hexagons touching."""
    # Distance between centers of touching unit hexagons = 2 * apothem = sqrt(3)
    d = math.sqrt(3)
    # Hex lattice basis: a1 = (d, 0), a2 = (d/2, d*sqrt(3)/2)
    a1 = np.array([d, 0.0])
    a2 = np.array([d / 2.0, d * math.sqrt(3) / 2.0])
    # Choose 12 lattice points in a compact cluster
    coords = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
              (2, -1), (-2, 1), (1, 1), (-1, -1), (2, 0)]
    pts = []
    for (i, j) in coords:
        p = i * a1 + j * a2
        pts.append(p)
    return pts


def _optimize_rotation(inner, max_iter=200):
    """Optimize outer hexagon (cx, cy, side, ang) to enclose inner with min side.
    Uses coordinate descent on ang and a bisection on side with center optimization.
    """
    # First, compute the minimal enclosing for a given orientation by finding
    # the required side such that all points are inside, centered at centroid.
    def side_for_orientation(ang_deg, cx, cy):
        # Find minimal side so that all inner points are inside hexagon centered (cx,cy)
        # Bisection on side
        lo = 1.0
        hi = 20.0
        for _ in range(60):
            mid = (lo + hi) / 2.0
            ok = True
            for (x, y, a) in inner:
                if _inside_violation(np.array([x, y]), cx, cy, mid, ang_deg) > 1e-9:
                    ok = False
                    break
            if ok:
                hi = mid
            else:
                lo = mid
        return hi

    # Optimize over ang and center
    best_side = 1e9
    best = (0.0, 0.0, 0.0, 0.0)
    centroid = np.mean([np.array([x, y]) for (x, y, a) in inner], axis=0)
    cx, cy = centroid[0], centroid[1]
    for ang in np.linspace(0, 60, 13):
        s = side_for_orientation(ang, cx, cy)
        if s < best_side:
            best_side = s
            best = (cx, cy, s, ang)
    return best


def _local_refine(inner, ocx, ocy, side, ang_deg, budget=10.0):
    """Penalty-based local refinement of inner positions to reduce violations.
    Returns improved inner if feasible side can be reduced.
    """
    start = time.time()
    rng = np.random.default_rng(42)
    inner = [list(p) for p in inner]
    best = _score_of(inner, ocx, ocy, side, ang_deg)
    best_inner = [list(p) for p in inner]
    step = 0.3
    it = 0
    while time.time() - start < budget and it < 2000:
        it += 1
        i = rng.integers(0, len(inner))
        p = inner[i]
        cand = [p[0] + rng.normal(0, step), p[1] + rng.normal(0, step), p[2]]
        new_inner = [list(q) for q in inner]
        new_inner[i] = cand
        s = _score_of(new_inner, ocx, ocy, side, ang_deg)
        if s < best:
            best = s
            best_inner = new_inner
            inner = new_inner
            step = max(0.05, step * 0.95)
        else:
            step = max(0.05, step * 0.9)
    return best_inner, best


def construct():
    t0 = time.time()
    # Build hexagonal lattice
    centers = _hex_lattice_centers()
    inner = [[float(p[0]), float(p[1]), 0.0] for p in centers]

    # Optimize outer hexagon to enclose the lattice
    ocx, ocy, side, ang = _optimize_rotation(inner)

    # Local refinement to tighten
    if time.time() - t0 < DEADLINE - 15:
        inner, viol = _local_refine(inner, ocx, ocy, side, ang, budget=min(20.0, DEADLINE - (time.time() - t0) - 10))
        # After refinement, re-optimize the outer hexagon to shrink side
        ocx, ocy, side, ang = _optimize_rotation(inner)

    # Final safety: ensure feasibility by enlarging side if needed
    if time.time() - t0 < DEADLINE - 5:
        # verify and enlarge if any violation
        for _ in range(200):
            if time.time() - t0 > DEADLINE - 2:
                break
            verts = [_hex_vertices(x, y, 1.0, a) for (x, y, a) in inner]
            ok = True
            for i in range(len(inner)):
                for j in range(i + 1, len(inner)):
                    if _penetration(verts[i], verts[j]) > 1e-7:
                        ok = False
                        break
                if not ok:
                    break
            for (x, y, a) in inner:
                if _inside_violation(np.array([x, y]), ocx, ocy, side, ang) > 1e-7:
                    ok = False
                    break
            if ok:
                break
            side *= 1.02
    return inner, (ocx, ocy), side, ang

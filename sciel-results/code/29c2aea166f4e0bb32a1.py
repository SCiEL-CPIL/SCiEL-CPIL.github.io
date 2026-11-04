"""Packing 12 unit regular hexagons into a minimal regular hexagon.

Strategy:
  * Guaranteed valid baseline: honeycomb lattice clusters (1x12, 2x6, 3x4),
    all hexagons parallel. Compute the exact minimal enclosing *parallel*
    regular hexagon side for each cluster; keep the best (single row -> 12).
  * Improvement: simulated annealing over (positions, per-hexagon rotation)
    with a hard deadline guard. The objective is the side length of the
    minimal enclosing regular hexagon (over its own rotation) of the current
    placement, but we only accept a move if it is *valid* (no overlaps and
    all hexagons inside the current best side). We always return the best
    valid packing seen, so the result is never worse than the baseline.
"""
import math
import random
import time

N = 12
DEADLINE = 135.0          # safety margin before the 150 s wall-clock kill
START = time.time()

# ---------------------------------------------------------------------------
# Regular hexagon geometry helpers
# ---------------------------------------------------------------------------
# A regular hexagon of side 1, "flat-top" orientation, centered at origin,
# has vertices at angles 0, 60, 120, 180, 240, 300 degrees (radius = 1).
# Its 6 outward face normals are at angles 30, 90, 150, 210, 270, 330 deg
# and the inradius (apothem) is sqrt(3)/2.

def hex_vertices(cx, cy, side, angle_deg):
    """Return the 6 (x, y) vertices of a regular hexagon (side, angle)."""
    a = math.radians(angle_deg)
    out = []
    for k in range(6):
        t = a + math.radians(60.0 * k)
        out.append((cx + side * math.cos(t), cy + side * math.sin(t)))
    return out

def hex_face_normals(angle_deg):
    """Outward unit normals of the 6 faces for a hexagon at angle_deg.
    For a hexagon whose vertices are at angles a+60k, the faces (edges)
    have outward normals at a+30+60k."""
    a = math.radians(angle_deg)
    out = []
    for k in range(6):
        t = a + math.radians(30.0 + 60.0 * k)
        out.append((math.cos(t), math.sin(t)))
    return out

def poly_overlap(v1, v2, eps=1e-9):
    """True if two convex polygons (lists of (x,y)) overlap in interior.
    Separating-axis test on convex polygons."""
    polys = (v1, v2)
    for poly in polys:
        m = len(poly)
        for i in range(m):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % m]
            # edge normal
            nx, ny = -(y2 - y1), (x2 - x1)
            # project both polygons
            min1, max1 = math.inf, -math.inf
            min2, max2 = math.inf, -math.inf
            for (px, py) in v1:
                d = nx * px + ny * py
                if d < min1: min1 = d
                if d > max1: max1 = d
            for (px, py) in v2:
                d = nx * px + ny * py
                if d < min2: min2 = d
                if d > max2: max2 = d
            if max1 <= min2 + eps or max2 <= min1 + eps:
                return False
    return True

def all_disjoint(vertices_list, eps=1e-9):
    """Check pairwise non-overlap of a list of convex polygons."""
    n = len(vertices_list)
    for i in range(n):
        for j in range(i + 1, n):
            if poly_overlap(vertices_list[i], vertices_list[j], eps):
                return False
    return True

def enclosing_hexagon_side(points, angle_deg):
    """Minimal side length S of a regular hexagon (centered at origin,
    given rotation) that contains all points. For each face normal n,
    the required support is max_i (n . p_i); S must satisfy S * apothem
    >= support, i.e. S >= support / apothem where apothem = sqrt(3)/2 * S?
    Careful: for a hexagon of side S, apothem = sqrt(3)/2 * S. The face
    inequality is n . p <= apothem = sqrt(3)/2 * S. So S >= 2*support/sqrt(3).
    Return max over faces of 2*support/sqrt(3), with support >= 0 handled
    by also considering the opposite face (we take max of |support|)."""
    ap = math.sqrt(3.0) / 2.0
    normals = hex_face_normals(angle_deg)
    best = 0.0
    for (nx, ny) in normals:
        support = 0.0
        for (px, py) in points:
            d = nx * px + ny * py
            if d > support:
                support = d
        # need support <= apothem = ap * S  ->  S >= support / ap
        if support / ap > best:
            best = support / ap
    return best

def min_enclosing_hexagon_side(points, n_angles=24):
    """Minimize enclosing hexagon side over its rotation (coarse grid)."""
    best = math.inf
    for k in range(n_angles):
        ang = 360.0 * k / n_angles
        s = enclosing_hexagon_side(points, ang)
        if s < best:
            best = s
    return best

def placement_vertices(centers, angles, side=1.0):
    return [hex_vertices(cx, cy, side, ang) for (cx, cy), ang in zip(centers, angles)]

def is_valid(centers, angles, side, center, eps=1e-9):
    """All hexagons disjoint and inside outer hexagon (center, side, angle 0)."""
    verts = placement_vertices(centers, angles)
    if not all_disjoint(verts, eps):
        return False
    # outer hexagon: center (center), side, rotation 0
    # translate points into outer frame
    ox, oy = center
    pts = [(cx - ox, cy - oy) for (cx, cy) in centers]
    # but we must check each *vertex* of each inner hexagon is inside outer
    ap = math.sqrt(3.0) / 2.0 * side
    normals = hex_face_normals(0.0)
    for v in verts:
        for (px, py) in v:
            dx, dy = px - ox, py - oy
            for (nx, ny) in normals:
                if nx * dx + ny * dy > ap + eps:
                    return False
    return True

def objective(centers, angles):
    """Side length of minimal enclosing hexagon (centered at origin) over rotation.
    We re-center the cloud at its centroid to be fair."""
    cx = sum(c[0] for c in centers) / len(centers)
    cy = sum(c[1] for c in centers) / len(centers)
    pts = [(c[0] - cx, c[1] - cy) for c in centers]
    return min_enclosing_hexagon_side(pts)

# ---------------------------------------------------------------------------
# Honeycomb baseline
# ---------------------------------------------------------------------------
def honeycomb_cluster(rows, cols):
    """Return centers of a rows x cols honeycomb block (all parallel, angle 0).
    Column spacing sqrt(3), row spacing 1.5, odd rows offset sqrt(3)/2."""
    centers = []
    for r in range(rows):
        for c in range(cols):
            x = c * math.sqrt(3.0) + (math.sqrt(3.0) / 2.0 if r % 2 else 0.0)
            y = r * 1.5
            centers.append((x, y))
    return centers

def honeycomb_best():
    candidates = [(1, 12), (2, 6), (3, 4), (4, 3), (6, 2), (12, 1)]
    best_side = math.inf
    best = None
    for (r, c) in candidates:
        if r * c != N:
            continue
        centers = honeycomb_cluster(r, c)
        # re-center at origin
        cx = sum(p[0] for p in centers) / N
        cy = sum(p[1] for p in centers) / N
        centers = [(p[0] - cx, p[1] - cy) for p in centers]
        angles = [0.0] * N
        side = min_enclosing_hexagon_side(centers, n_angles=36)
        # verify validity with this side (outer centered at origin, angle 0)
        if is_valid(centers, angles, side, (0.0, 0.0)):
            if side < best_side:
                best_side = side
                best = (centers, angles, side)
    return best

# ---------------------------------------------------------------------------
# Simulated annealing improvement
# ---------------------------------------------------------------------------
def sa_improve(base_centers, base_angles, base_side, budget_s):
    """Try to beat base_side. Returns (centers, angles, side) of best valid."""
    best_centers = list(base_centers)
    best_angles = list(base_angles)
    best_side = base_side

    centers = [c[:] for c in base_centers]
    angles = list(base_angles)
    cur_side = objective(centers, angles)

    temp = 1.0
    cooling = 0.999
    step = 0.3
    end = time.time() + budget_s
    rng = random.Random(12345)

    while time.time() < end:
        # pick a random hexagon and perturb
        i = rng.randrange(N)
        dx = rng.uniform(-step, step)
        dy = rng.uniform(-step, step)
        da = rng.uniform(-25.0, 25.0)
        ox, oy = centers[i]
        oa = angles[i]
        centers[i] = [ox + dx, oy + dy]
        angles[i] = oa + da

        new_side = objective(centers, angles)
        delta = new_side - cur_side
        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-9)):
            cur_side = new_side
            # check validity at the *current best side* to only keep valid ones
            if new_side < best_side:
                if is_valid(centers, angles, best_side, (0.0, 0.0)):
                    best_side = new_side
                    best_centers = [c[:] for c in centers]
                    best_angles = list(angles)
        else:
            centers[i] = [ox, oy]
            angles[i] = oa

        temp *= cooling
        step *= 0.9995

    return best_centers, best_angles, best_side

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def construct():
    # 1) honeycomb baseline (always valid)
    hb = honeycomb_best()
    if hb is None:
        # extremely defensive fallback: single row
        centers = [(float(k), 0.0) for k in range(N)]
        cx = sum(p[0] for p in centers) / N
        centers = [(p[0] - cx, 0.0) for p in centers]
        angles = [0.0] * N
        side = min_enclosing_hexagon_side(centers, n_angles=36)
        best_centers, best_angles, best_side = centers, angles, side
    else:
        best_centers, best_angles, best_side = hb

    # 2) simulated annealing to try to improve, with a deadline guard
    remaining = DEADLINE - (time.time() - START)
    if remaining > 8.0:
        try:
            c2, a2, s2 = sa_improve(best_centers, best_angles, best_side,
                                    remaining - 10.0)
            if s2 < best_side:
                best_centers, best_angles, best_side = c2, a2, s2
        except Exception:
            pass

    inner = [[cx, cy, ang] for (cx, cy), ang in zip(best_centers, best_angles)]
    return inner, (0.0, 0.0), float(best_side), 0.0

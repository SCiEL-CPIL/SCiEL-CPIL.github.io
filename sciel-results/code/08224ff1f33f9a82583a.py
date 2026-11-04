"""Packing 12 unit hexagons in a minimal regular hexagon via lattice topologies + 1D rotation search."""
import math
import time

N = 12

_DEADLINE = time.time() + 145.0  # safety margin under 150s kill


def _hex_vertices(cx, cy, s, ang_deg):
    """Vertices of regular hexagon, flat-top convention at ang=0."""
    a = math.radians(ang_deg)
    vs = []
    for k in range(6):
        t = a + math.radians(60 * k)
        vs.append((cx + s * math.cos(t), cy + s * math.sin(t)))
    return vs


def _inner_points(positions):
    """All vertices of all inner unit hexagons (positions: list of (x,y,ang_deg))."""
    pts = []
    for (x, y, a) in positions:
        vs = _hex_vertices(x, y, 1.0, a)
        pts.extend(vs)
    return pts


def _min_side_for_points(pts, theta_deg):
    """Minimal side of regular hexagon (center free) with rotation theta containing pts."""
    th = math.radians(theta_deg)
    # 6 outward normals of a regular hexagon with rotation theta
    # For a regular hexagon, the 6 face normals are spaced 60 deg apart.
    # A regular hexagon (pointy or flat) has normals at angles th + 30 + 60k? 
    # We define hexagon by 6 supporting lines with normals n_i at angle th + 60*i,
    # and half-width w = s*sqrt(3)/2. This gives a regular hexagon of side s.
    normals = []
    for i in range(6):
        ang = th + math.radians(60 * i)
        normals.append((math.cos(ang), math.sin(ang)))
    widths = []
    for (nx, ny) in normals:
        proj = [px * nx + py * ny for (px, py) in pts]
        widths.append(max(proj) - min(proj))
    return max(widths) / math.sqrt(3.0)


def _best_side_for_points(pts):
    """Minimize side over rotation via golden section over [0, 30) degrees (period 30 due to symmetry? hexagon period is 60, width function period 30)."""
    # width function is periodic with period 30 degrees? normals rotate, set of 6 normals repeats every 60 deg, but max over 6 is periodic 30? Actually set {th+60i} for i=0..5 equals set {th+30+60i} shifted, which is a different orientation (rotated 30). The hexagon rotated 30 deg is the same shape (regular hexagon has 6-fold symmetry, so rotation by 60 is identity; rotation by 30 is a different orientation). The width function period is 60. But due to symmetry of max over 6 directions spaced 60, width(th+60)=width(th). Also width(th+30) corresponds to the dual orientation. We search [0,60).
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    lo, hi = 0.0, 60.0
    c = hi - gr * (hi - lo)
    d = lo + gr * (hi - lo)
    fc = _min_side_for_points(pts, c)
    fd = _min_side_for_points(pts, d)
    for _ in range(40):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - gr * (hi - lo)
            fc = _min_side_for_points(pts, c)
        else:
            lo, c, fc = c, d, fd
            d = lo + gr * (hi - lo)
            fd = _min_side_for_points(pts, d)
    return min(fc, fd)


def _build_lattice_clusters():
    """Generate connected lattice clusters of 12 unit hexagons on triangular lattice (spacing 2)."""
    # Triangular lattice basis: e1=(2,0), e2=(1, sqrt(3))
    e1 = (2.0, 0.0)
    e2 = (1.0, math.sqrt(3.0))

    def coord_to_xy(i, j):
        return (i * e1[0] + j * e2[0], i * e1[1] + j * e2[1])

    # Generate all sets of 12 lattice points that are connected and "compact"
    # Use BFS from origin, grow shapes, prune by bounding box to keep count manageable.
    # We'll generate shapes by choosing a set of 12 coords within a small radius, connected.
    # Simpler: enumerate via recursive growth with a fixed ordering to avoid duplicates.
    shapes = []
    seen = set()

    # candidate coordinate pool within radius ~4
    pool = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x, y = coord_to_xy(i, j)
            if math.hypot(x, y) <= 9.0:
                pool.append((i, j))
    pool_set = set(pool)

    def neighbors(i, j):
        res = []
        for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]:
            ni, nj = i + di, j + dj
            if (ni, nj) in pool_set:
                res.append((ni, nj))
        return res

    # canonical form: translate so min i then min j is at (0,0), then frozenset
    def canonical(coords):
        mi = min(c[0] for c in coords)
        mj = min(c[1] for c in coords)
        shifted = frozenset((c[0] - mi, c[1] - mj) for c in coords)
        return shifted

    # DFS growth: start with (0,0), add cells from frontier, keep connected, stop at 12.
    # To limit, only grow within pool and use a visited set of canonical states at size 12.
    stack = [frozenset([(0, 0)])]
    found = set()
    max_iter = 200000
    iters = 0
    while stack and iters < max_iter and time.time() < _DEADLINE - 5:
        iters += 1
        cur = stack.pop()
        if len(cur) == 12:
            canon = canonical(set(cur))
            if canon not in found:
                found.add(canon)
                shapes.append(set(canon))
            continue
        if len(cur) > 12:
            continue
        # frontier: cells adjacent to cur but not in cur
        frontier = set()
        for (i, j) in cur:
            for nb in neighbors(i, j):
                if nb not in cur:
                    frontier.add(nb)
        # deterministic order
        for nb in sorted(frontier):
            stack.append(cur | {nb})

    # Fallback: also add a few hand-crafted compact shapes
    # 3x3 hex cluster (19 cells) minus 7 -> we want 12; use a "hexagon of side 2" = 7 cells, plus 5.
    # We'll just rely on DFS; if too few, add a linear and a 2x6 block.
    if len(shapes) < 5:
        # 2x6 block
        coords = [(i, j) for i in range(6) for j in range(2)]
        shapes.append(set(coords))
        # 3x4 block
        coords = [(i, j) for i in range(4) for j in range(3)]
        shapes.append(set(coords))
        # line
        coords = [(i, 0) for i in range(12)]
        shapes.append(set(coords))

    # Convert to positions (all angle 0)
    result = []
    for coords in shapes:
        positions = [coord_to_xy(i, j) + (0.0,) for (i, j) in coords]
        result.append(positions)
    return result


def construct():
    start = time.time()
    best_positions = None
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # Fallback trivial packing (spread out) to always return something
    fallback_positions = [(4.0 * k, 0.0, 0.0) for k in range(-(N // 2), N - N // 2)]
    fallback_pts = _inner_points(fallback_positions)
    fallback_side = _best_side_for_points(fallback_pts)

    try:
        clusters = _build_lattice_clusters()
    except Exception:
        clusters = []

    # Evaluate each cluster: compute minimal side over rotation; also try a few inner rotations
    for positions in clusters:
        if time.time() > _DEADLINE - 10:
            break
        # Try inner angle 0 and 30 (rotating each inner hex by 30 may help fit)
        for inner_ang in (0.0, 30.0):
            pos2 = [(x, y, inner_ang) for (x, y, a) in positions]
            pts = _inner_points(pos2)
            side = _best_side_for_points(pts)
            if side < best_side:
                best_side = side
                best_positions = pos2
                # compute optimal center for the best theta
                # re-find best theta
                gr = (math.sqrt(5.0) - 1.0) / 2.0
                lo, hi = 0.0, 60.0
                c = hi - gr * (hi - lo)
                d = lo + gr * (hi - lo)
                fc = _min_side_for_points(pts, c)
                fd = _min_side_for_points(pts, d)
                for _ in range(40):
                    if fc < fd:
                        hi, d, fd = d, c, fc
                        c = hi - gr * (hi - lo)
                        fc = _min_side_for_points(pts, c)
                    else:
                        lo, c, fc = c, d, fd
                        d = lo + gr * (hi - lo)
                        fd = _min_side_for_points(pts, d)
                best_theta = (lo + hi) / 2.0
                # optimal center = midpoint of projections for the binding direction
                th = math.radians(best_theta)
                normals = []
                for i in range(6):
                    ang = th + math.radians(60 * i)
                    normals.append((math.cos(ang), math.sin(ang)))
                best_w = -1
                best_n = normals[0]
                for (nx, ny) in normals:
                    proj = [px * nx + py * ny for (px, py) in pts]
                    w = max(proj) - min(proj)
                    if w > best_w:
                        best_w = w
                        best_n = (nx, ny)
                proj = [px * best_n[0] + py * best_n[1] for (px, py) in pts]
                cx = (max(proj) + min(proj)) / 2.0 * best_n[0]
                cy = (max(proj) + min(proj)) / 2.0 * best_n[1]
                # But center must also satisfy other directions; using the binding direction center is optimal for that direction; overall optimal center is intersection - we approximate by averaging over all 6 direction midpoints.
                cx2 = 0.0
                cy2 = 0.0
                for (nx, ny) in normals:
                    proj = [px * nx + py * ny for (px, py) in pts]
                    mid = (max(proj) + min(proj)) / 2.0
                    cx2 += mid * nx
                    cy2 += mid * ny
                cx2 /= 6.0
                cy2 /= 6.0
                best_center = (cx2, cy2)
                best_angle = best_theta

    # If no cluster found or side is inf, use fallback
    if best_positions is None or not math.isfinite(best_side):
        best_positions = fallback_positions
        best_side = fallback_side
        # compute center/angle for fallback
        pts = _inner_points(fallback_positions)
        gr = (math.sqrt(5.0) - 1.0) / 2.0
        lo, hi = 0.0, 60.0
        c = hi - gr * (hi - lo)
        d = lo + gr * (hi - lo)
        fc = _min_side_for_points(pts, c)
        fd = _min_side_for_points(pts, d)
        for _ in range(40):
            if fc < fd:
                hi, d, fd = d, c, fc
                c = hi - gr * (hi - lo)
                fc = _min_side_for_points(pts, c)
            else:
                lo, c, fc = c, d, fd
                d = lo + gr * (hi - lo)
                fd = _min_side_for_points(pts, d)
        best_theta = (lo + hi) / 2.0
        th = math.radians(best_theta)
        normals = []
        for i in range(6):
            ang = th + math.radians(60 * i)
            normals.append((math.cos(ang), math.sin(ang)))
        cx2 = 0.0
        cy2 = 0.0
        for (nx, ny) in normals:
            proj = [px * nx + py * ny for (px, py) in pts]
            mid = (max(proj) + min(proj)) / 2.0
            cx2 += mid * nx
            cy2 += mid * ny
        cx2 /= 6.0
        cy2 /= 6.0
        best_center = (cx2, cy2)
        best_angle = best_theta

    # Ensure side is a bit above the computed minimum to avoid floating point edge violations
    side_out = best_side * 1.0000001 + 1e-9

    # Return inner rows (x, y, angle_degrees), outer center, side, angle
    inner = [[x, y, a] for (x, y, a) in best_positions]
    return inner, best_center, side_out, best_angle


# EVOLVE-BLOCK-START
# (construct defined above)
# EVOLVE-BLOCK-END

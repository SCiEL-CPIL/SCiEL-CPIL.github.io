"""Hexagonal-lattice packing of unit hexagons inside a minimal enclosing regular hexagon."""
import math
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side, rot_deg):
    """Vertices of a flat-top regular hexagon, rotated by rot_deg about (cx, cy)."""
    th = math.radians(rot_deg)
    cos_t, sin_t = math.cos(th), math.sin(th)
    verts = []
    for k in range(6):
        ang = math.radians(60.0 * k)
        x = side * math.cos(ang)
        y = side * math.sin(ang)
        # rotate
        xr = x * cos_t - y * sin_t
        yr = x * sin_t + y * cos_t
        verts.append((cx + xr, cy + yr))
    return np.array(verts)


def _convex_hull(points):
    """Andrew's monotone chain convex hull. points: (M,2). Returns hull vertices CCW."""
    pts = np.unique(points, axis=0)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    if len(pts) <= 1:
        return pts
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1])
    return hull


def _min_enclosing_hex(hull, max_iter=2000):
    """Find minimal side length of a regular hexagon enclosing the convex hull.
    Tries many orientations; for each, computes required side via support functions.
    Returns (best_side, best_center, best_rot_deg)."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_rot = 0.0
    # The enclosing hexagon orientation: we can assume the inner hexagons are axis-aligned
    # (rot 0). The outer hexagon orientation is a free parameter.
    # For a regular hexagon with outward normal directions at angles phi_k, the width
    # in direction n is max over hull of dot(p, n) - min over hull of dot(p, n).
    # The hexagon of side s with center c and normal angle phi has, in each normal dir,
    # support = s * cos(30 deg) = s * sqrt(3)/2 (apothem).
    # So we need: for all 6 normals, (max dot - min dot)/2 <= s * sqrt(3)/2  AND center must be midpoint.
    # Actually the center must satisfy: for each normal n_k, c.n_k = (max_k + min_k)/2.
    # These 6 equations are consistent only if the hexagon is positioned right. In general
    # the minimal enclosing regular hexagon: we pick orientation phi, then the required
    # side is max over k of width_k / sqrt(3), where width_k is hull width in normal dir.
    # And center = mean of (max_k+min_k)/2 projections -> but they must be consistent.
    # Simpler: for a given orientation, the minimal enclosing regular hexagon has side
    # s = max_k width_k / sqrt(3) and center is the centroid that centers all widths.
    # We compute center as the average of the 6 midpoints projected back.
    for phi_deg in np.linspace(0.0, 60.0, 601):
        phi = math.radians(phi_deg)
        normals = []
        for k in range(6):
            a = phi + math.radians(60.0 * k)
            normals.append((math.cos(a), math.sin(a)))
        widths = []
        mids = []
        for nx, ny in normals:
            d = hull[:, 0] * nx + hull[:, 1] * ny
            mx = d.max()
            mn = d.min()
            widths.append(mx - mn)
            mids.append((mx + mn) / 2.0)
        s = max(widths) / math.sqrt(3.0)
        # center: solve c such that c.n_k = mid_k for all k. Use least squares.
        A = np.array(normals)
        b = np.array(mids)
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
        # verify by checking actual containment is satisfied (it is by construction of widths)
        if s < best_side - 1e-9:
            best_side = s
            best_center = (float(c[0]), float(c[1]))
            best_rot = float(phi_deg)
    return best_side, best_center, best_rot


def construct():
    # Hexagonal lattice of N unit hexagons (side 1). Flat-top orientation.
    # Horizontal spacing between same-row centers: 3 (since width = 2, gap 1 -> 3).
    # Vertical spacing between rows: sqrt(3). Alternate rows offset by 1.5 horizontally.
    # Arrange in a hexagonal-shaped cluster of N hexagons (like a bigger hexagon of hexagons).
    # For a hexagonal number cluster of "radius" r (in hex units), count = 3r(r+1)+1.
    # We want approximately N hexagons; build the smallest hexagonal cluster with >= N,
    # or just build a compact cluster. Simpler: build rows.
    # Let's build a hexagonal cluster of side length (in hexagons) m, total = 3m(m-1)+1.
    # Choose m so that total is close to N.
    # For N=11, m=2 gives 7, m=3 gives 19. Use m=3 (19) or m=2 (7)? We must place exactly N.
    # Place exactly N: build rows of a hexagonal cluster and take first N.
    # Hexagonal cluster rows (centered): row lengths for radius r: (r+1), r+1, ..., then decreasing.
    # For radius r: rows from -r..r, length = r - |i| + 1.
    # total = sum_{i=-r}^{r} (r - |i| + 1) = 3r(r+1)+1.
    # Choose smallest r with total >= N.
    r = 0
    while 3 * r * (r + 1) + 1 < N:
        r += 1
    # Build positions in hex-lattice coordinates (row i, col j) and convert to Cartesian.
    # Use axial-ish: for row i in -r..r, cols j in -(r-|i|)..(r-|i|).
    positions = []
    for i in range(-r, r + 1):
        maxj = r - abs(i)
        for j in range(-maxj, maxj + 1):
            # Convert (i, j) to Cartesian for flat-top hex tiling.
            # x = 3 * (j + (i % 2) / 2.0) ... but i can be negative; use offset by row parity.
            # Standard offset coords: x = 3*(j + 0.5*(i&1)) but with negative i, parity: use (i%2).
            # y = sqrt(3) * i
            x = 3.0 * (j + 0.5 * (i % 2))
            y = math.sqrt(3.0) * i
            positions.append((x, y))
    # Take exactly N positions (the cluster is centered at origin).
    positions = positions[:N]
    inner = [[x, y, 0.0] for (x, y) in positions]

    # Compute convex hull of all vertices of all inner hexagons.
    all_pts = []
    for (x, y, a) in inner:
        v = _hex_vertices(x, y, 1.0, a)
        all_pts.append(v)
    all_pts = np.vstack(all_pts)
    hull = _convex_hull(all_pts)
    side, center, rot = _min_enclosing_hex(hull)
    return inner, center, side, rot
# EVOLVE-BLOCK-END

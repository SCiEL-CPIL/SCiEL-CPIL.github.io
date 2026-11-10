# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, s, a_deg):
    """Return (6,2) array of vertices for hexagon centered at (cx,cy) with side s, angle a_deg."""
    a = np.deg2rad(a_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _edge_normals(a_deg=0.0):
    """Return 6 outward edge normals for a regular hexagon with given angle."""
    a = np.deg2rad(a_deg)
    # Vertices at angles a, a+60, ..., a+300
    # Edges between consecutive vertices; outward normal bisects the angle between
    # the two adjacent vertex directions
    # For vertex at angle theta, the edge to next vertex has normal at theta + 30
    return np.array([[np.cos(a + np.deg2rad(30 + k * 60)), np.sin(a + np.deg2rad(30 + k * 60))]
                     for k in range(6)])


def _separation_ok(v1, v2, gap=1e-6):
    """Check if two hexagons are separated using SAT.
    v1, v2: (6,2) vertex arrays.
    Returns True if they are NOT intersecting (separated)."""
    # Collect all 12 edge normals (6 from each hex, but for same orientation they're the same)
    # For safety, use 12 normals: 6 from each
    n1 = _edge_normals(0.0)
    n2 = _edge_normals(0.0)
    all_normals = np.vstack([n1, n2])  # 12 normals (with duplicates for same orientation)
    
    for n in all_normals:
        p1 = v1 @ n
        p2 = v2 @ n
        min1, max1 = np.min(p1), np.max(p1)
        min2, max2 = np.min(p2), np.max(p2)
        # Intervals are apart if max1 < min2 - gap or max2 < min1 - gap
        if max1 < min2 - gap or max2 < min1 - gap:
            return True
    return False


def _inside_outer(v, side, tol=-1e-6):
    """Check if all points v (n,2) are inside or on outer hexagon (centered at origin, side=side, angle=0)."""
    normals = _edge_normals(0.0)
    # Distance from center to each edge is side * sqrt(3) / 2
    d = side * np.sqrt(3) / 2
    projections = v @ normals.T  # (n, 6)
    if np.any(projections > d - tol):
        return False
    return True


def _is_valid(inner_xy, side, outer_center=np.array([0.0, 0.0])):
    """Full validity check."""
    if not np.all(np.isfinite(inner_xy)):
        return False
    if side < np.sqrt(11) - 1e-9:
        return False
    verts = [_hex_vertices(x, y, 1.0, 0.0) for x, y in inner_xy]
    # Check separation
    for i in range(11):
        for j in range(i + 1, 11):
            if not _separation_ok(verts[i], verts[j]):
                return False
    # Check containment
    outer_v = _hex_vertices(outer_center[0], outer_center[1], side, 0.0)
    all_verts = np.vstack(verts)
    if not _inside_outer(all_verts, side):
        return False
    # Check edge lengths
    for v in verts:
        for i in range(6):
            d = np.linalg.norm(v[(i+1)%6] - v[i])
            if abs(d - 1.0) > 1e-8:
                return False
    return True


def _min_side_for_positions(inner_xy):
    """Find minimum side of outer hex (centered at origin, angle 0) containing all inner hexes."""
    normals = _edge_normals(0.0)
    all_verts = np.vstack([_hex_vertices(x, y, 1.0, 0.0) for x, y in inner_xy])
    projections = all_verts @ normals.T  # (66, 6)
    max_proj = np.max(projections, axis=1)
    s_min = np.max(max_proj) / (np.sqrt(3) / 2)
    return s_min


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon side length.
    """
    t_start = time.time()
    deadline = 620.0

    best_side = np.inf
    best_inner = None

    s3 = np.sqrt(3)

    # Multiple initial configurations
    configs = [
        # Flower: center + 6 ring + 4 outer
        np.array([
            [0, 0], [2, 0], [-2, 0], [1, s3], [-1, s3], [1, -s3], [-1, -s3],
            [3, s3], [3, -s3], [-3, s3], [-3, -s3],
        ]),
        # Compact: center + 6 ring + 4 at cardinal directions
        np.array([
            [0, 0], [2, 0], [-2, 0], [1, s3], [-1, s3], [1, -s3], [-1, -s3],
            [0, 2*s3], [0, -2*s3], [4, 0], [-4, 0],
        ]),
        # Linear row with offsets
        np.array([
            [0, 0], [2, 0], [-2, 0], [4, 0], [-4, 0],
            [1, s3], [-1, s3], [1, -s3], [-1, -s3],
            [3, s3], [-3, -s3],
        ]),
        # Tighter: try to pack more densely
        np.array([
            [0, 0], [2, 0], [-2, 0],
            [1, s3], [-1, s3], [1, -s3], [-1, -s3],
            [0, 2*s3], [0, -2*s3],
            [3, 0], [-3, 0],
        ]),
    ]

    for ci, init_xy in enumerate(configs):
        if time.time() > deadline:
            break

        # Check initial
        side0 = _min_side_for_positions(init_xy)
        if _is_valid(init_xy, side0):
            if side0 < best_side:
                best_side = side0
                best_inner = init_xy.copy()

        # Optimize with Nelder-Mead
        def obj(params):
            xy = params.reshape(11, 2)
            if not np.all(np.isfinite(xy)):
                return 1e10
            return _min_side_for_positions(xy)

        try:
            res = minimize(obj, init_xy.flatten(), method='Nelder-Mead',
                          options={'maxiter': 10000, 'xatol': 1e-9, 'fatol': 1e-11,
                                   'adaptive': True})
            if res.fun < best_side:
                opt_xy = res.x.reshape(11, 2)
                opt_side = _min_side_for_positions(opt_xy)
                if _is_valid(opt_xy, opt_side):
                    if opt_side < best_side:
                        best_side = opt_side
                        best_inner = opt_xy.copy()
        except Exception:
            pass

    if best_inner is not None:
        inner_hex_data = np.column_stack([best_inner, np.zeros(11)])
        outer_hex_data = np.array([0.0, 0.0, 0.0])
        return inner_hex_data, outer_hex_data, float(best_side)

    # Fallback
    config = configs[0]
    side = 10.0
    inner_hex_data = np.column_stack([config, np.zeros(11)])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    return inner_hex_data, outer_hex_data, float(side)


# EVOLVE-BLOCK-END

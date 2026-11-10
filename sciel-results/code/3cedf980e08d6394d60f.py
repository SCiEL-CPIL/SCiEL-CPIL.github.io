# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution


def _hex_vertices(cx, cy, d, angle_deg):
    a = np.deg2rad(angle_deg)
    i = np.arange(6)
    return np.column_stack([
        cx + d * np.cos(a + np.deg2rad(60) * i),
        cy + d * np.sin(a + np.deg2rad(60) * i),
    ])


def _outer_vertices(cx, cy, d, angle_deg):
    return _hex_vertices(cx, cy, d, angle_deg)


def _separation_ok(c1, c2, a1, a2, tol=1e-6):
    """SAT for two hexagons: returns True if separated (not overlapping).
    Hexagons same side=1. Returns True if there exists an edge normal (from either)
    such that projection intervals are disjoint, OR they are actually separated.
    We return False if they overlap (intervals overlap for ALL axes)."""
    v1 = _hex_vertices(c1[0], c1[1], 1.0, a1)
    v2 = _hex_vertices(c2[0], c2[1], 1.0, a2)
    edges = []
    for v in (v1, v2):
        for i in range(6):
            p1 = v[i]
            p2 = v[(i + 1) % 6]
            e = p2 - p1
            # outward normal (for convex polygon, normal = perp of edge)
            n = np.array([-e[1], e[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
                edges.append(n)
    # For each axis, check projection intervals
    for n in edges:
        proj1 = n @ v1
        proj2 = n @ v2
        lo1, hi1 = proj1.min(), proj1.max()
        lo2, hi2 = proj2.min(), proj2.max()
        # disjoint if hi1 + tol < lo2 or hi2 + tol < lo1
        if hi1 + tol < lo2 or hi2 + tol < lo1:
            return True  # separated
    return False  # overlap (no separating axis found)


def _inside_outer(v, outer_v, tol=1e-6):
    """Check vertex v is inside (or on) convex polygon outer_v (CCW)."""
    n = len(outer_v)
    for i in range(n):
        p1 = outer_v[i]
        p2 = outer_v[(i + 1) % n]
        e = p2 - p1
        cr = e[0] * (v[1] - p1[1]) - e[1] * (v[0] - p1[0])
        if cr < -tol:
            return False
    return True


def _objective(params, n=12):
    # params: 12*(2) centers + 12 angles + outer(cx,cy,angle) + s
    # layout: [cx0..cx11, cy0..cy11, a0..a11, ocx, ocy, oang, s]
    cx = params[0:12]
    cy = params[12:24]
    ang = params[24:36]
    ocx, ocy, oang, s = params[36], params[37], params[38], params[39]

    if s < np.sqrt(12) - 1e-9:
        return 1e6 + (np.sqrt(12) - s)

    # objective: minimize s
    obj = s

    # separation penalty
    penalty = 0.0
    centers = np.column_stack([cx, cy])
    for i in range(12):
        for j in range(i + 1, 12):
            if not _separation_ok(centers[i], centers[j], ang[i], ang[j]):
                # measure overlap depth roughly via min distance between vertices
                vi = _hex_vertices(cx[i], cy[i], 1.0, ang[i])
                vj = _hex_vertices(cx[j], cy[j], 1.0, ang[j])
                # min distance between any vertex of i and any edge of j and vice versa
                dmin = _min_hex_distance(cx[i], cy[i], ang[i], cx[j], cy[j], ang[j])
                penalty += max(0.0, -dmin) ** 2

    # containment penalty
    outer_v = _outer_vertices(ocx, ocy, s, oang)
    for i in range(12):
        vi = _hex_vertices(cx[i], cy[i], 1.0, ang[i])
        for v in vi:
            if not _inside_outer(v, outer_v):
                # distance to outside: find max violation
                viol = 0.0
                for k in range(6):
                    p1 = outer_v[k]
                    p2 = outer_v[(k + 1) % 6]
                    e = p2 - p1
                    cr = e[0] * (v[1] - p1[1]) - e[1] * (v[0] - p1[0])
                    if cr < 0:
                        viol = max(viol, -cr)
                penalty += viol ** 2

    return obj + 100.0 * penalty


def _min_hex_distance(c1, c2, a1, a2):
    """Return min distance between two hexagons (positive if separated, negative approx if overlapping via vertex-edge)."""
    v1 = _hex_vertices(c1, c1, 1.0, a1) if False else _hex_vertices(c1, c2, 1.0, a1)  # placeholder
    # recompute properly
    v1 = _hex_vertices(c1, c2 if False else c1, 1.0, a1)
    v1 = _hex_vertices(c1, c1, 1.0, a1)
    v2 = _hex_vertices(c2, c2, 1.0, a2)
    best = np.inf
    # vertex of v1 to edges of v2
    for v in v1:
        for i in range(6):
            p1 = v2[i]
            p2 = v2[(i + 1) % 6]
            e = p2 - p1
            L2 = np.dot(e, e)
            if L2 < 1e-12:
                d = np.linalg.norm(v - p1)
            else:
                t = np.dot(v - p1, e) / L2
                t = max(0.0, min(1.0, t))
                q = p1 + t * e
                d = np.linalg.norm(v - q)
            best = min(best, d)
    for v in v2:
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i + 1) % 6]
            e = p2 - p1
            L2 = np.dot(e, e)
            if L2 < 1e-12:
                d = np.linalg.norm(v - p1)
            else:
                t = np.dot(v - p1, e) / L2
                t = max(0.0, min(1.0, t))
                q = p1 + t * e
                d = np.linalg.norm(v - q)
            best = min(best, d)
    return best


def hexagon_packing_12():
    # Start from a good heuristic: honeycomb-like cluster of 12 hexagons.
    # For flat-top hexagons (0 deg), horizontal spacing = 3 (2*1.5), vertical = sqrt(3).
    # Arrange in a compact hexagonal cluster.
    # Positions for a hexagonal number pattern of 12: center + ring of 6 + partial ring of 5.
    # Ring1 (6): at distance 3 in 6 directions (flat-top: directions at 0,60,120,...)
    # Actually for flat-top, neighbors are at (±3,0), (±1.5, ±sqrt(3)), (0, ±2*sqrt(3))? 
    # Let's use axial coords. Flat-top hex side 1: 
    # neighbor offsets (flat-top, pointy direction is vertical):
    #   E: (3, 0)
    #   W: (-3, 0)
    #   NE: (1.5, sqrt(3))
    #   NW: (-1.5, sqrt(3))
    #   SE: (1.5, -sqrt(3))
    #   SW: (-1.5, -sqrt(3))
    sqrt3 = np.sqrt(3)
    base = np.array([
        [0, 0],
        [3, 0],
        [-3, 0],
        [1.5, sqrt3],
        [-1.5, sqrt3],
        [1.5, -sqrt3],
        [-1.5, -sqrt3],
        [0, 2 * sqrt3],
        [0, -2 * sqrt3],
        [3, sqrt3],
        [3, -sqrt3],
        [-3, sqrt3],
    ])
    # That's 12 hexagons in a nice cluster. All angle 0.
    cx0 = base[:, 0]
    cy0 = base[:, 1]
    ang0 = np.zeros(12)
    # outer hex centered at origin, angle 0 (flat-top), side ~ enough
    ocx0, ocy0, oang0 = 0.0, 0.0, 0.0
    s0 = 9.0
    x0 = np.concatenate([cx0, cy0, ang0, [ocx0, ocy0, oang0, s0]])

    bounds = [(-15, 15)] * 24 + [(-180, 180)] * 12 + [(-15, 15), (-15, 15), (-180, 180), (3.5, 15)]

    result = differential_evolution(
        _objective,
        bounds=bounds,
        x0=x0,
        seed=42,
        maxiter=200,
        tol=1e-6,
        popsize=20,
        polish=True,
        init='latinhypercube',
    )

    p = result.x
    cx = p[0:12]
    cy = p[12:24]
    ang = p[24:36]
    ocx, ocy, oang, s = p[36], p[37], p[38], p[39]

    # Build inner hex data
    inner_hex_data = np.column_stack([cx, cy, ang])
    outer_hex_data = np.array([ocx, ocy, oang])

    # Ensure s is at least sqrt(12)
    s = max(s, np.sqrt(12) + 1e-9)

    return inner_hex_data, outer_hex_data, float(s)


# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 630.0


def _vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(th), cy + s * np.sin(th)])


def _hexagon_vertices(hx, hy, ha):
    return _vertices(hx, hy, 1.0, ha)


def _separated(v1, v2):
    """Return True if two unit hexagons (given as 6x2 vertex arrays) are
    separated: on at least one of the 12 edge normals the projected intervals
    are apart by more than 1e-6."""
    for V in (v1, v2):
        for i in range(6):
            a = V[i]
            b = V[(i + 1) % 6]
            e = b - a
            n = np.array([-e[1], e[0]])
            n /= (np.linalg.norm(n) + 1e-30)
            p1 = V @ n
            p2 = (other := v2) @ n
            if p1.max() - p2.min() > 1e-6 or p2.max() - p1.min() > 1e-6:
                return True
    return False


def _contained(verts, outer_verts):
    """All points verts (N,2) inside (or on) convex polygon outer_verts (6,2)."""
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        e = p2 - p1
        n = np.array([-e[1], e[0]])
        cross = np.cross(e, verts - p1)
        if (cross < -1e-6).any():
            return False
    return True


def _side_for_center_angle(verts, cx, cy, a_deg):
    """Min side length of a regular hexagon (center cx,cy, angle a_deg) that
    contains all points in verts (N,2)."""
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0 * np.arange(6))
    N = np.column_stack([-np.sin(th), np.cos(th)])  # outward normals
    R = np.column_stack([np.cos(th), np.sin(th)])
    rel = verts - np.array([cx, cy])
    proj = rel @ N.T  # (N,6)
    return proj.max() / (R @ R.T).max()


def _is_valid(inner, outer, side, tol=1e-6):
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        return False
    if not np.isfinite(side) or side < np.sqrt(11.0) - 1e-9:
        return False
    ox, oy, oa = outer[0], outer[1], outer[2]
    # shifted edge-length check
    for i in range(11):
        hx, hy, ha = inner[i]
        hx -= ox
        hy -= oy
        v = _hexagon_vertices(hx, hy, ha)
        for j in range(6):
            L = np.linalg.norm(v[(j + 1) % 6] - v[j])
            if abs(L - 1.0) > 1e-8:
                return False
    # pairwise separation
    verts = [_hexagon_vertices(inner[i][0], inner[i][1], inner[i][2]) for i in range(11)]
    for i in range(11):
        for j in range(i + 1, 11):
            if not _separated(verts[i], verts[j]):
                return False
    # containment
    outer_verts = _vertices(ox, oy, side, oa)
    allv = np.vstack(verts)
    if not _contained(allv, outer_verts):
        return False
    return True


def hexagon_packing_11():
    t0 = time.time()

    # Compact hexagonal cluster of 11 unit hexagons (flat-top, angle 0).
    # Center + 6 ring + 4 far corners.
    s3 = np.sqrt(3.0)
    inner = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, s3 / 2, 0.0],
        [-0.5, s3 / 2, 0.0],
        [-1.0, 0.0, 0.0],
        [-0.5, -s3 / 2, 0.0],
        [0.5, -s3 / 2, 0.0],
        [1.5, s3 / 2, 0.0],
        [1.5, -s3 / 2, 0.0],
        [-1.5, s3 / 2, 0.0],
        [-1.5, -s3 / 2, 0.0],
    ])

    verts = np.vstack([_hexagon_vertices(inner[i][0], inner[i][1], inner[i][2]) for i in range(11)])

    # Optimize outer center (x,y) and angle to minimize enclosing side length.
    def f(params):
        if time.time() - t0 > _DEADLINE - 30:
            return 1e9
        cx, cy, a_deg = params
        return _side_for_center_angle(verts, cx, cy, a_deg)

    best = None
    # coarse grid over center and angle
    for cx in np.arange(-1.5, 1.51, 0.5):
        for cy in np.arange(-1.5, 1.51, 0.5):
            for a_deg in np.arange(0.0, 180.0, 15.0):
                val = f([cx, cy, a_deg])
                if best is None or val < best[0]:
                    best = (val, cx, cy, a_deg)
    if best is None:
        return inner, np.array([0.0, 0.0, 0.0]), 8.0

    val, cx, cy, a_deg = best
    # local refinement
    for step in (0.25, 0.1, 0.04, 0.015):
        improved = True
        while improved and time.time() - t0 < _DEADLINE - 30:
            improved = False
            for dcx in (-step, 0.0, step):
                for dcy in (-step, 0.0, step):
                    for da in (-step * 4, 0.0, step * 4):
                        v2 = f([cx + dcx, cy + dcy, a_deg + da])
                        if v2 < val - 1e-12:
                            val, cx, cy, a_deg = v2, cx + dcx, cy + dcy, a_deg + da
                            improved = True
    side = float(val)

    # small safety margin so containment is comfortably satisfied
    side += 1e-4

    outer = np.array([cx, cy, a_deg])

    if not _is_valid(inner, outer, side):
        # fallback to a guaranteed-valid loose box
        return inner, np.array([0.0, 0.0, 0.0]), 8.0

    return inner, outer, side


# EVOLVE-BLOCK-END

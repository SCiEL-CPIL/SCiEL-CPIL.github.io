# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 600.0
_EPS = 1e-6


def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack((cx + s * np.cos(th), cy + s * np.sin(th)))


def _axis_aligned_min_side(points):
    # Smallest axis-aligned regular hexagon (center free) containing points.
    # Axis-aligned unit hexagon: x in [-0.8660254, 0.8660254], y in [-0.5, 0.5].
    # A point (x,y) is inside iff (|x|/0.8660254)^2 + (|y|/0.5)^2 <= 1.
    # Center (cx,cy) must satisfy (x-cx)^2 + (1.5(y-cy))^2 <= 1 for every point
    # (unit disk in transformed space). Min enclosing disk => min side.
    X = np.asarray(points, dtype=float)
    X = np.column_stack((X[:, 0], 1.5 * X[:, 1]))
    n = X.shape[0]
    if n == 1:
        return 1.0, X[0, 0], X[0, 1] / 1.5
    if n == 2:
        c = 0.5 * (X[0] + X[1])
        r = float(np.linalg.norm(X[0] - X[1]))
        return r, c[0], c[1] / 1.5

    def _diameter():
        best = 0.0
        bc = None
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(X[i] - X[j]))
                if d > best:
                    best = d
                    bc = 0.5 * (X[i] + X[j])
        return best, bc

    r, c = _diameter()
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if i == j or j == k or i == k:
                    continue
                ax, ay = X[i]
                bx, by = X[j]
                cx, cy = X[k]
                dx = bx - ax
                dy = by - ay
                ex = cx - ax
                ey = cy - ay
                d2 = dx * dx + dy * dy
                if d2 < 1e-12:
                    continue
                num = 2.0 * (dx * ex + dy * ey)
                den = 2.0 * d2
                if num <= 1e-9 or num >= den - 1e-9:
                    continue
                t = num / den
                ux = ax + t * dx
                uy = ay + t * dy
                rr = max(float(np.hypot(p[0] - ux, p[1] - uy)) for p in X)
                if rr < r - 1e-12:
                    r = rr
                    c = np.array([ux, uy])
    return r, c[0], c[1] / 1.5


def _check_valid(inner, outer, side):
    if not (np.all(np.isfinite(inner)) and np.all(np.isfinite(outer)) and np.isfinite(side)):
        return False
    if side < np.sqrt(11.0) - 1e-6:
        return False
    ocx, ocy, oa = outer
    ov = _hex_vertices(ocx, ocy, side, oa)
    # containment: every inner vertex inside outer hexagon
    for row in inner:
        iv = _hex_vertices(row[0], row[1], 1.0, row[2])
        for p in iv:
            for e in range(6):
                p1 = ov[e]
                p2 = ov[(e + 1) % 6]
                if float(np.cross(p2 - p1, p - p1)) < -1e-6:
                    return False
    # separation of all inner pairs via edge-normal intervals
    verts = [_hex_vertices(r[0], r[1], 1.0, r[2]) for r in inner]
    for a in range(11):
        for b in range(a + 1, 11):
            ok = False
            for va in verts[a]:
                for e in range(6):
                    p1 = va[e]
                    p2 = va[(e + 1) % 6]
                    d = p2 - p1
                    L = float(np.hypot(d[0], d[1]))
                    if L < 1e-12:
                        continue
                    nrm = np.array([-d[1], d[0]]) / L
                    proj = np.dot(verts[b], nrm)
                    mn = float(proj.min())
                    mx = float(proj.max())
                    base = float(np.dot(p1, nrm))
                    if (mx < base - _EPS) or (mn > base + _EPS):
                        ok = True
                        break
                if ok:
                    break
            if not ok:
                return False
    return True


def hexagon_packing_11():
    """
    Returns
        inner_hex_data: np.ndarray shape (11,3) rows (x, y, angle_deg) of unit hexagons.
        outer_hex_data: np.ndarray shape (3,) (x, y, angle_deg) of outer hexagon.
        outer_hex_side_length: float side length.
    """
    inner_hex_data = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [-2.5, 0.0, 0.0],
            [1.25, 2.1650635094610964, 0.0],
            [-1.25, 2.1650635094610964, 0.0],
            [0.0, 4.330127018922193, 0.0],
            [1.25, -2.1650635094610964, 0.0],
            [-1.25, -2.1650635094610964, 0.0],
            [0.0, -4.330127018922193, 0.0],
            [3.75, 2.1650635094610964, 0.0],
            [-3.75, -2.1650635094610964, 0.0],
        ],
        dtype=float,
    )

    # All inner hexagons are axis-aligned (angle 0), so the optimal enclosing
    # hexagon is axis-aligned; compute its minimal side and center.
    verts = []
    for r in inner_hex_data:
        verts.append(_hex_vertices(r[0], r[1], 1.0, r[2]))
    allp = np.vstack(verts)
    side, cx, cy = _axis_aligned_min_side(allp)
    outer_hex_data = np.array([cx, cy, 0.0], dtype=float)

    if not _check_valid(inner_hex_data, outer_hex_data, side):
        # Fallback: generous enclosing hexagon at origin.
        outer_hex_data = np.array([0.0, 0.0, 0.0], dtype=float)
        side = 8.0

    return inner_hex_data, outer_hex_data, float(side)


# EVOLVE-BLOCK-END

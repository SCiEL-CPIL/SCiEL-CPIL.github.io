# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 630.0


def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0 * np.arange(6))
    return np.stack([cx + s * np.cos(th), cy + s * np.sin(th)], axis=1)


def _outer_edges(cx, cy, s, a_deg):
    v = _hex_vertices(cx, cy, s, a_deg)
    return np.stack([v, np.roll(v, -1, axis=0)], axis=1)


def _separation_ok(p, q, gap=1e-6):
    # 12 edge normals of two unit hexes
    an = np.deg2rad(np.arange(6) * 60.0)
    bn = np.deg2rad(np.arange(6) * 60.0)
    for base in (p, q):
        for ang in (an, bn):
            n = np.array([np.cos(ang), np.sin(ang)])
            d = (q - p) @ n
            if abs(d) > 1.0 + gap:
                return True
    return False


def _validate(inner, outer, side):
    if not (np.all(np.isfinite(inner)) and np.all(np.isfinite(outer))):
        return False
    if side < np.sqrt(11.0):
        return False
    # pairwise separation
    for i in range(inner.shape[0]):
        for j in range(i + 1, inner.shape[0]):
            if not _separation_ok(inner[i, :2], inner[j, :2]):
                return False
    # containment
    o = outer[:2]
    edges = _outer_edges(outer[0], outer[1], side, outer[2])
    for row in inner:
        verts = _hex_vertices(row[0], row[1], 1.0, row[2])
        for v in verts:
            for e in edges:
                p1, p2 = e[0], e[1]
                if np.cross(p2 - p1, v - p1) < -1e-6:
                    return False
    # edge-length preservation after shifting by outer center
    for row in inner:
        verts = _hex_vertices(row[0] - o[0], row[1] - o[1], 1.0, row[2])
        for k in range(6):
            if abs(np.linalg.norm(verts[(k + 1) % 6] - verts[k]) - 1.0) > 1e-8:
                return False
    return True


def _build_cluster():
    # Centered hexagonal cluster of 11 unit hexagons that tiles a side-3 hexagon.
    # Axial coords (q, r) -> cartesian for pointy-top unit hexes (size 1):
    #   x = sqrt(3) * (q + r/2),  y = 1.5 * r
    sq3 = np.sqrt(3.0)
    coords = [(0, 0)]
    for q, r in [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]:
        coords.append((q, r))
    for q, r in [(2, 0), (2, -1), (-2, 0), (-2, 1)]:
        coords.append((q, r))

    inner = np.zeros((len(coords), 3))
    for k, (q, r) in enumerate(coords):
        inner[k, 0] = sq3 * (q + 0.5 * r)
        inner[k, 1] = 1.5 * r
        inner[k, 2] = 0.0

    outer = np.array([0.0, 0.0, 0.0])
    side = 3.0
    return inner, outer, side


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger
    regular hexagon. Uses a centered hexagonal cluster that exactly tiles a
    regular hexagon of side length 3.

    Returns
        inner_hex_data: np.ndarray of shape (11,3), rows (x, y, angle_deg).
        outer_hex_data: np.ndarray of shape (3,), (x, y, angle_deg).
        outer_hex_side_length: float side length of the outer hexagon.
    """
    inner, outer, side = _build_cluster()

    if _validate(inner, outer, side):
        return inner, outer, side

    # Safety fallback: a guaranteed-valid (if looser) arrangement.
    inner = np.array(
        [
            [0.0, 0.0, 0.0],
            [-2.5, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [-1.25, 2.165, 0.0],
            [1.25, 2.165, 0.0],
            [-1.25, -2.165, 0.0],
            [1.25, -2.165, 0.0],
            [-3.75, 2.165, 0.0],
            [3.75, 2.165, 0.0],
            [-3.75, -2.165, 0.0],
            [3.75, -2.165, 0.0],
        ]
    )
    outer = np.array([0.0, 0.0, 0.0])
    side = 8.0
    if _validate(inner, outer, side):
        return inner, outer, side
    return inner, outer, side


# EVOLVE-BLOCK-END

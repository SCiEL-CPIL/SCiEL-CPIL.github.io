"""Pack 11 unit hexagons in a minimal regular hexagon (rotation-robust)."""
import math
import time

N = 11
DEADLINE = 115.0  # safety margin below the 150 s kill

def _hex_vertices(cx, cy, rot_deg):
    """6 vertices of a unit regular hexagon (circumradius 1)."""
    a0 = math.radians(rot_deg)
    out = []
    for k in range(6):
        a = a0 + math.radians(60.0 * k)
        out.append((cx + math.cos(a), cy + math.sin(a)))
    return out

def _eval_pattern(coords, inner_rot):
    """Return (outer_side, outer_center, inner_centers) for a lattice pattern."""
    s = math.sqrt(3.0)
    e1 = (s, 0.0)
    e2 = (s / 2.0, 1.5)
    centers = []
    verts = []
    for (i, j) in coords:
        cx = i * e1[0] + j * e2[0]
        cy = i * e1[1] + j * e2[1]
        centers.append((cx, cy))
        verts.extend(_hex_vertices(cx, cy, inner_rot))
    cx = sum(p[0] for p in verts) / len(verts)
    cy = sum(p[1] for p in verts) / len(verts)
    # Rotation-robust inradius: the inscribed circle (radius R) of the outer
    # hexagon contains every point within distance R of the center.
    R = 0.0
    for (x, y) in verts:
        d = math.hypot(x - cx, y - cy)
        if d > R:
            R = d
    S = 2.0 * R / s
    return S, (cx, cy), centers

def construct():
    t0 = time.time()
    # Compact 11-cell clusters on the hexagonal lattice (each has exactly 11).
    patterns = [
        [(-1, -1), (0, -1), (1, -1), (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
         (-1, 1), (0, 1), (1, 1)],
        [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0),
         (-1, 1), (0, 1), (1, 1), (-1, 2), (0, 2)],
        [(-1, -1), (0, -1), (1, -1), (2, -1), (-1, 0), (0, 0), (1, 0), (2, 0),
         (-1, 1), (0, 1), (1, 1)],
        [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0),
         (-1, 1), (0, 1), (1, 1), (-2, 0), (2, 0)],
        [(0, -1), (1, -1), (2, -1), (3, -1), (-1, 0), (0, 0), (1, 0),
         (-1, 1), (0, 1), (1, 1), (2, 1)],
    ]
    best = None
    for coords in patterns:
        if time.time() - t0 > DEADLINE:
            break
        for inner_rot in (0.0, 30.0):
            S, c, centers = _eval_pattern(coords, inner_rot)
            if best is None or S < best[0]:
                best = (S, c, centers, inner_rot)
    if best is None:
        centers = [(4.0 * k, 0.0) for k in range(-(N // 2), N - N // 2)]
        return [[x, y, 0.0] for (x, y) in centers], (0.0, 0.0), 40.0, 0.0
    S, c, centers, inner_rot = best
    inner = [[float(x), float(y), float(inner_rot)] for (x, y) in centers]
    return inner, (float(c[0]), float(c[1])), float(S), 0.0

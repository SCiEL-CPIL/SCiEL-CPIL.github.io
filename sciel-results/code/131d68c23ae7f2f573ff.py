# EVOLVE-BLOCK-START
import time
import numpy as np


def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    ang = a + np.deg2rad(60.0 * np.arange(6))
    return np.stack([cx + s * np.cos(ang), cy + s * np.sin(ang)], axis=1)


def _separation_ok(i, j, verts):
    # verts: (6,2) each. Separated if on some shared edge normal the projected
    # intervals are apart by more than 1e-6.
    for k in range(6):
        e = verts[i, (k + 1) % 6] - verts[i, k]
        n = np.array([-e[1], e[0]])
        nl = np.linalg.norm(n)
        if nl < 1e-12:
            continue
        n = n / nl
        pi = verts[i] @ n
        pj = verts[j] @ n
        gap = max(pi.min(), pj.min()) - min(pi.max(), pj.max())
        if gap > 1e-6:
            return True
    return False


def _all_separated(verts):
    n = len(verts)
    for i in range(n):
        for j in range(i + 1, n):
            if not _separation_ok(i, j, verts):
                return False
    return True


def _inside_outer(verts, outer_center, outer_verts):
    # For each outer edge p1->p2, cross(p2-p1, p-p1) >= -1e-6 for all inner p.
    for p1, p2 in zip(outer_verts, np.roll(outer_verts, -1, axis=0)):
        e = p2 - p1
        for v in verts:
            p = v - p1
            if np.cross(e, p) < -1e-6:
                return False
    return True


def hexagon_packing_11():
    start = time.time()
    deadline = start + 600.0

    # 11 hexagons in honeycomb: center + 6 ring + 4 ring-2 corners.
    # Axial coords -> pointy-top pixel (x, y), unit side, spacing sqrt(3).
    sx = np.sqrt(3.0)
    axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (0, 2), (-2, 2), (2, -2),
    ]
    inner = np.zeros((11, 3))
    for idx, (q, r) in enumerate(axial):
        inner[idx, 0] = sx * (q + 0.5 * r)
        inner[idx, 1] = 1.5 * r
        inner[idx, 2] = 0.0

    outer_center = np.array([0.0, 0.0])
    side = 3.0 + np.sqrt(3.0)
    outer = np.array([0.0, 0.0, 0.0])

    # Shift inner by outer center (zero here) and build vertices for validation.
    shifted = inner[:, :2] - outer_center
    verts = [_hex_vertices(x, y, 1.0, a) for (x, y, a) in shifted]
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], side, outer[2])

    # Validate (finite, side, separation, containment, edge lengths).
    ok = True
    if not (np.all(np.isfinite(inner)) and np.all(np.isfinite(outer))):
        ok = False
    if side < np.sqrt(11.0):
        ok = False
    if not _all_separated(verts):
        ok = False
    if not _inside_outer(shifted, outer_center, outer_verts):
        ok = False
    for v in verts:
        edges = np.roll(v, -1, axis=0) - v
        if not np.all(np.abs(np.linalg.norm(edges, axis=1) - 1.0) < 1e-8):
            ok = False

    if not ok:
        # Fallback: enlarge outer side slightly to guarantee containment.
        side = 6.0
        outer_verts = _hex_vertices(outer_center[0], outer_center[1], side, outer[2])
        if not _inside_outer(shifted, outer_center, outer_verts):
            side = 10.0
            outer_verts = _hex_vertices(outer_center[0], outer_center[1], side, outer[2])

    if time.time() > deadline - 30.0:
        pass  # safety margin; nothing else to do

    return inner, outer, float(side)


# EVOLVE-BLOCK-END

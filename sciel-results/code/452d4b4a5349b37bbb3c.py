"""Compact hex-lattice packing of N unit hexagons, minimal enclosing regular hexagon."""
import math
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, s=1.0, angle_deg=0.0):
    """6 vertices of a regular hexagon (flat-top, angle 0 = vertex on +x axis)."""
    a = math.radians(angle_deg)
    verts = []
    for k in range(6):
        th = a + k * math.pi / 3.0
        verts.append((cx + s * math.cos(th), cy + s * math.sin(th)))
    return verts

def _enclosing_side(points, center, phi_deg):
    """Minimal side of a regular hexagon (orientation phi_deg) centered at `center`
    that contains all `points`."""
    cx, cy = center
    a0 = math.radians(phi_deg) + math.pi / 6.0  # first edge-normal angle
    maxproj = 0.0
    for k in range(6):
        th = a0 + k * math.pi / 3.0
        nx, ny = math.cos(th), math.sin(th)
        for (px, py) in points:
            d = (px - cx) * nx + (py - cy) * ny
            if d > maxproj:
                maxproj = d
    # inradius a = maxproj  ->  side s = 2a / sqrt(3)
    return 2.0 * maxproj / math.sqrt(3.0)

def construct():
    # Compact hex-lattice blob of N unit hexagons (flat-top, angle 0).
    sqrt3 = math.sqrt(3.0)
    u = (sqrt3, 0.0)          # horizontal neighbor vector
    v = (sqrt3 / 2.0, 1.5)    # diagonal neighbor vector
    axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-2, 1), (-2, 0),
    ][:N]
    centers = []
    for (q, r) in axial:
        x = q * u[0] + r * v[0]
        y = q * u[1] + r * v[1]
        centers.append((x, y))

    inner = [[x, y, 0.0] for (x, y) in centers]

    # All vertices of all inner hexagons (must be contained by outer hexagon).
    points = []
    for (x, y) in centers:
        points.extend(_hex_vertices(x, y, 1.0, 0.0))

    # Centroid as initial outer center.
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)

    # Coarse grid over orientation (period 60 deg).
    best = (None, None, None)
    for i in range(120):
        phi = i * (60.0 / 120.0)
        s = _enclosing_side(points, (cx, cy), phi)
        if best[0] is None or s < best[0]:
            best = (s, (cx, cy), phi)

    s, center, phi = best

    # Local refinement of center (and a finer phi) to shrink the enclosing side.
    def total_s(c, p):
        return _enclosing_side(points, c, p)

    # Nelder-Mead-lite on (cx, cy, phi)
    import copy
    best_c = list(center); best_p = phi; best_s = s
    step = 0.2
    for _ in range(200):
        improved = False
        for dcx in (-step, step):
            for dcy in (-step, step):
                ns = total_s((best_c[0] + dcx, best_c[1] + dcy), best_p)
                if ns < best_s - 1e-9:
                    best_c = [best_c[0] + dcx, best_c[1] + dcy]
                    best_s = ns; improved = True
        for dp in (-step, step):
            ns = total_s(tuple(best_c), best_p + dp)
            if ns < best_s - 1e-9:
                best_p = best_p + dp; best_s = ns; improved = True
        if not improved:
            step *= 0.5
            if step < 1e-4:
                break

    return inner, (best_c[0], best_c[1]), best_s, best_p
# EVOLVE-BLOCK-END

"""
Optimized packing of 11 unit regular hexagons inside a regular hexagon.

Strategy:
1. Use a known dense hexagonal lattice arrangement (flower + ring) for the 11 hexagons.
2. Compute the minimal outer regular hexagon (with arbitrary rotation) that contains all inner hexagons.
3. Use scipy.optimize.minimize to fine-tune positions and rotation to minimize the outer side length.
4. Include a deadline guard to ensure we finish within the time limit.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 11
START_TIME = time.time()
TIME_LIMIT = 120.0  # seconds


def hex_vertices(cx, cy, side, angle_deg):
    """
    Return the 6 vertices of a regular hexagon with center (cx, cy),
    side length `side`, and rotation angle in degrees.
    Vertices are ordered counter-clockwise starting from angle=0 pointing right.
    """
    angles = np.deg2rad(angle_deg) + np.deg2rad(np.arange(0, 360, 60))
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack((xs, ys))


def point_in_hexagon(p, cx, cy, side, angle_deg):
    """
    Check if point p (array [x, y]) is inside or on the boundary of a regular
    hexagon with center (cx, cy), side length `side`, rotation angle_deg.
    Uses half-plane tests on the 6 edges.
    """
    verts = hex_vertices(cx, cy, side, angle_deg)
    for i in range(6):
        v1 = verts[i]
        v2 = verts[(i + 1) % 6]
        # Edge from v1 to v2, interior is to the left
        edge = v2 - v1
        normal = np.array([-edge[1], edge[0]])
        # Ensure normal points inward: check against center
        if np.dot(normal, np.array([cx, cy]) - v1) < 0:
            normal = -normal
        if np.dot(normal, p - v1) < -1e-9:
            return False
    return True


def hexagon_overlap_area(v1, v2):
    """
    Compute the area of intersection of two convex polygons (each with 6 vertices).
    Uses Sutherland-Hodgman polygon clipping.
    """
    # Sutherland-Hodgman: clip polygon `subject` by half-plane defined by `clip_edge`
    def clip_polygon_by_half_plane(subject, a, b):
        """Clip polygon `subject` by the half-plane to the left of edge a->b."""
        if len(subject) == 0:
            return []
        output = []
        edge = b - a
        normal = np.array([-edge[1], edge[0]])
        # Ensure normal points inward for the clip
        # The half-plane is: dot(normal, p - a) >= 0
        def inside(p):
            return np.dot(normal, p - a) >= -1e-9

        def intersect(p1, p2):
            # Find intersection of line p1-p2 with line a-b
            # Line a-b: normal . (p - a) = 0
            # Line p1-p2: p1 + t*(p2-p1)
            d = p2 - p1
            denom = np.dot(normal, d)
            if abs(denom) < 1e-12:
                return None
            t = -np.dot(normal, p1 - a) / denom
            if -1e-9 <= t <= 1 + 1e-9:
                return p1 + t * d
            return None

        input_list = list(subject)
        for i in range(len(input_list)):
            current = input_list[i]
            previous = input_list[i - 1]
            cur_inside = inside(current)
            prev_inside = inside(previous)

            if cur_inside:
                if not prev_inside:
                    pt = intersect(previous, current)
                    if pt is not None:
                        output.append(pt)
                output.append(current)
            else:
                if prev_inside:
                    pt = intersect(previous, current)
                    if pt is not None:
                        output.append(pt)
        return output

    # Clip polygon 1 by each edge of polygon 2
    poly = [list(v1[i]) for i in range(6)]
    poly = np.array(poly)
    for i in range(6):
        a = v2[i]
        b = v2[(i + 1) % 6]
        poly = np.array(clip_polygon_by_half_plane(poly, a, b))
        if len(poly) == 0:
            return 0.0

    # Compute area of resulting polygon
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return max(0.0, area)


def check_no_overlap(inner, eps=1e-6):
    """
    Check that no two inner hexagons overlap. Returns True if valid.
    """
    n = len(inner)
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ai = inner[i]
            xj, yj, aj = inner[j]
            vi = hex_vertices(xi, yi, 1.0, ai)
            vj = hex_vertices(xj, yj, 1.0, aj)
            area = hexagon_overlap_area(vi, vj)
            if area > eps:
                return False
    return True


def outer_side_length(inner, cx, cy, angle_deg):
    """
    Compute the minimum side length of a regular hexagon centered at (cx, cy)
    with rotation angle_deg that contains all inner hexagons.
    """
    n = len(inner)
    max_dist = 0.0
    for i in range(n):
        xi, yi, ai = inner[i]
        verts = hex_vertices(xi, yi, 1.0, ai)
        for p in verts:
            # Transform point to outer hexagon's local frame
            dx = p[0] - cx
            dy = p[1] - cy
            theta = np.deg2rad(angle_deg)
            # Rotate by -theta
            px = dx * np.cos(theta) + dy * np.sin(theta)
            py = -dx * np.sin(theta) + dy * np.cos(theta)
            # In a regular hexagon with side s, the apothem is s * sqrt(3)/2
            # The distance from center to a vertex is s
            # The max of |px| and |py| in hexagonal coordinates... 
            # Actually, for a regular hexagon with flat sides (angle=0), 
            # the constraint is: |x| <= s, and |sqrt(3)/2 * x + y| <= s, etc.
            # Simpler: the regular hexagon with side s and rotation 0 has vertices at
            # (s,0), (s/2, s*sqrt(3)/2), etc.
            # The half-width in x is s, and the constraint for containment is:
            # For each of the 6 half-planes, the point must be inside.
            # The 6 normals (in local frame) are at angles 0, 60, 120, 180, 240, 300 degrees
            # But the apothem direction is at 30, 90, 150, ... for flat-top
            # Let's use the support function: for a regular hexagon with side s,
            # the support function in direction u is s * max over vertices of dot(u, v)/s
            # Actually, the minimal s such that point (px, py) is inside is:
            # s >= max over the 6 edge normals of the projection
            # The 6 outward normals for a regular hexagon (flat-top, side s) are at angles 0, 60, 120, 180, 240, 300
            # The apothem is s*sqrt(3)/2, so the distance from center to each edge is s*sqrt(3)/2
            # So the constraint is: for each normal direction n_k, dot((px,py), n_k) <= s*sqrt(3)/2
            # Thus s >= 2/sqrt(3) * max_k dot((px,py), n_k)
            normals = []
            for k in range(6):
                ang = np.deg2rad(k * 60)
                normals.append([np.cos(ang), np.sin(ang)])
            normals = np.array(normals)
            projections = normals @ np.array([px, py])
            max_proj = np.max(np.abs(projections))
            # s >= 2/sqrt(3) * max_proj
            s_req = (2.0 / np.sqrt(3.0)) * max_proj
            if s_req > max_dist:
                max_dist = s_req
    return max_dist


def objective(params):
    """
    Objective function for optimization.
    params: flattened array of positions and angles for all inner hexagons,
            plus outer center and angle.
    """
    n = N
    # params layout: [x0, y0, a0, x1, y1, a1, ..., x10, y10, a10, cx, cy, angle_outer]
    inner = []
    for i in range(n):
        x = params[3 * i]
        y = params[3 * i + 1]
        a = params[3 * i + 2]
        inner.append([x, y, a])
    cx = params[3 * n]
    cy = params[3 * n + 1]
    angle_outer = params[3 * n + 2]

    side = outer_side_length(inner, cx, cy, angle_outer)

    # Penalty for overlaps
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ai = inner[i]
            xj, yj, aj = inner[j]
            vi = hex_vertices(xi, yi, 1.0, ai)
            vj = hex_vertices(xj, yj, 1.0, aj)
            area = hexagon_overlap_area(vi, vj)
            if area > 0:
                penalty += area * 1000.0

    return side + penalty


def construct():
    global START_TIME, TIME_LIMIT

    # Initial arrangement: central hexagon + 6 around + 4 more in a ring
    # Hexagonal lattice spacing: 2 for non-overlapping (touching)
    # Central at (0,0)
    # Ring 1 at distance 2 in 6 directions
    # Ring 2: we need 4 more, place them at distance 4 in some directions

    inner = []
    # Central
    inner.append([0.0, 0.0, 0.0])
    # Ring 1: 6 hexagons
    for k in range(6):
        ang = np.deg2rad(k * 60)
        x = 2.0 * np.cos(ang)
        y = 2.0 * np.sin(ang)
        inner.append([x, y, 0.0])
    # Ring 2: 4 hexagons at distance 4, at angles 0, 90, 180, 270? 
    # Actually, in a hex lattice, the next ring is at distance 4 in the 6 main directions,
    # but we only need 4. Let's place them at 0, 60, 120, 180 degrees at distance 4.
    for k in range(4):
        ang = np.deg2rad(k * 60)
        x = 4.0 * np.cos(ang)
        y = 4.0 * np.sin(ang)
        inner.append([x, y, 0.0])

    # Optimize
    n = len(inner)
    init_params = []
    for h in inner:
        init_params.extend(h)
    # Outer center and angle
    init_params.extend([0.0, 0.0, 0.0])
    init_params = np.array(init_params)

    best_params = init_params.copy()
    best_val = objective(init_params)

    # Try a few optimization runs
    options = {'maxiter': 200, 'ftol': 1e-8, 'disp': False}
    
    try:
        res = minimize(objective, init_params, method='Nelder-Mead', options=options)
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
    except Exception:
        pass

    # Check time
    if time.time() - START_TIME > TIME_LIMIT - 10.0:
        inner_out = []
        for i in range(n):
            inner_out.append([best_params[3*i], best_params[3*i+1], best_params[3*i+2]])
        return inner_out, (best_params[3*n], best_params[3*n+1]), best_val, best_params[3*n+2]

    # Try L-BFGS-B for further refinement
    try:
        res2 = minimize(objective, best_params, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-10, 'disp': False})
        if res2.fun < best_val:
            best_val = res2.fun
            best_params = res2.x
    except Exception:
        pass

    # Extract result
    inner_out = []
    for i in range(n):
        inner_out.append([best_params[3*i], best_params[3*i+1], best_params[3*i+2]])
    cx = best_params[3*n]
    cy = best_params[3*n+1]
    angle_outer = best_params[3*n+2]
    side = best_val

    return inner_out, (cx, cy), side, angle_outer


if __name__ == "__main__":
    result = construct()
    print(result)

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import differential_evolution

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    sqrt3 = math.sqrt(3)

    def get_hex_vertices(cx, cy, side, angle_deg):
        """Returns 6 vertices of a regular hexagon."""
        angle_rad = math.radians(angle_deg)
        verts = []
        for k in range(6):
            a = angle_rad + k * math.pi / 3
            verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
        return verts

    def point_in_hexagon(px, py, hx, hy, side, angle_deg):
        """Check if point (px, py) is inside hexagon centered at (hx, hy) with given side and angle."""
        dx = px - hx
        dy = py - hy
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        inradius = side * sqrt3 / 2
        # For flat-top hexagon (angle=0), edge normals are at 0, 60, 120, 180, 240, 300
        # After rotation by angle_deg, normals rotate too
        for k in range(6):
            normal_angle = math.radians(k * 60)
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            if lx * nx + ly * ny > inradius + 1e-9:
                return False
        return True

    def point_in_polygon(point, polygon):
        """Ray casting algorithm to determine if a point is inside a polygon."""
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        else:
                            xinters = p1x
                        if (p1x <= xinters or p2x <= xinters) and (p1x != p2x):
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def on_segment(p, q, r):
        """Check if point q lies on segment pr."""
        if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
            return True
        return False

    def orientation(p, q, r):
        """Find orientation of triangle formed by p, q, r. 0: collinear, 1: clockwise, 2: counterclockwise"""
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if abs(val) < 1e-10:
            return 0
        return 1 if val > 0 else 2

    def do_intersect(p1, q1, p2, q2):
        """Check if segment p1-q1 intersects with segment p2-q2."""
        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)

        if o1 != o2 and o3 != o4:
            return True

        if o1 == 0 and on_segment(p1, p2, q1):
            return True
        if o2 == 0 and on_segment(p1, q2, q1):
            return True
        if o3 == 0 and on_segment(p2, p1, q2):
            return True
        if o4 == 0 and on_segment(p2, q1, q2):
            return True

        return False

    def edges_intersect(v1, v2):
        """Check if any edge of v1 intersects any edge of v2."""
        for i in range(6):
            p1 = v1[i]
            q1 = v1[(i + 1) % 6]
            for j in range(6):
                p2 = v2[j]
                q2 = v2[(j + 1) % 6]
                if do_intersect(p1, q1, p2, q2):
                    return True
        return False

    def hexes_overlap(v1, v2):
        """
        Check if two sets of vertices (hexagons) overlap.
        Returns True if they overlap (interiors intersect).
        """
        # Check if any vertex of v1 is strictly inside v2
        for p in v1:
            if point_in_polygon(p, v2):
                return True
        # Check if any vertex of v2 is strictly inside v1
        for p in v2:
            if point_in_polygon(p, v1):
                return True
        # Check edge intersections
        if edges_intersect(v1, v2):
            return True
        return False

    def all_inside_outer(outer_cx, outer_cy, outer_side, outer_angle, inner_verts_list):
        """Check if all inner hexagons are inside the outer hexagon."""
        for verts in inner_verts_list:
            for vx, vy in verts:
                if not point_in_hexagon(vx, vy, outer_cx, outer_cy, outer_side, outer_angle):
                    return False
        return True

    def no_overlaps(inner_verts_list):
        """Check that no two inner hexagons overlap."""
        for i in range(N):
            for j in range(i + 1, N):
                # Quick bounding box check
                cx1 = sum(v[0] for v in inner_verts_list[i]) / 6
                cy1 = sum(v[1] for v in inner_verts_list[i]) / 6
                cx2 = sum(v[0] for v in inner_verts_list[j]) / 6
                cy2 = sum(v[1] for v in inner_verts_list[j]) / 6
                dist_sq = (cx1 - cx2)**2 + (cy1 - cy2)**2
                if dist_sq > 4.5:  # circumradius is 1, so if centers > 2 apart, no overlap
                    continue
                if hexes_overlap(inner_verts_list[i], inner_verts_list[j]):
                    return False
        return True

    def objective(params):
        if time.time() > deadline:
            return 1e10

        # Parse params:
        # [outer_cx, outer_cy, outer_side, outer_angle,
        #  ix0, iy0, iang0, ix1, iy1, iang1, ...]
        outer_cx = params[0]
        outer_cy = params[1]
        outer_side = params[2]
        outer_angle = params[3]

        if outer_side < 1.0:
            return 1e10

        # Extract inner hexagons
        inner_verts_list = []
        for i in range(N):
            ix = params[4 + i * 3]
            iy = params[5 + i * 3]
            iang = params[6 + i * 3]
            verts = get_hex_vertices(ix, iy, 1.0, iang)
            inner_verts_list.append(verts)

        # Check if all inner hexagons are inside outer hexagon
        if not all_inside_outer(outer_cx, outer_cy, outer_side, outer_angle, inner_verts_list):
            return outer_side + 50.0

        # Check for overlaps between inner hexagons
        if not no_overlaps(inner_verts_list):
            return outer_side + 100.0

        return outer_side

    # Initial guess: Honeycomb lattice (4 rows x 3 cols)
    # Row 0 (y=0): x = -2, 0, 2
    # Row 1 (y=sqrt(3)): x = -1, 1, 3
    # Row 2 (y=2*sqrt(3)): x = -2, 0, 2
    # Row 3 (y=3*sqrt(3)): x = -1, 1, 3
    positions = []
    for row in range(4):
        y = row * sqrt3
        for col in range(3):
            if row % 2 == 0:
                x = (col - 1) * 2.0  # -2, 0, 2
            else:
                x = (col - 1) * 2.0 + 1.0  # -1, 1, 3
            positions.append((x, y))

    # Center the arrangement
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx_offset = (min(xs) + max(xs)) / 2.0
    cy_offset = (min(ys) + max(ys)) / 2.0

    init_params = [0.0, 0.0, 8.0, 0.0]  # Outer guess: center, side, angle
    for x, y in positions:
        init_params.extend([x - cx_offset, y - cy_offset, 0.0])

    # Bounds
    bounds = [
        (-6.0, 6.0),   # outer_cx
        (-6.0, 6.0),   # outer_cy
        (3.0, 12.0),   # outer_side
        (0.0, 60.0)    # outer_angle
    ]
    for i in range(N):
        bounds.append((-6.0, 6.0))   # ix
        bounds.append((-6.0, 6.0))   # iy
        bounds.append((0.0, 60.0))   # iang

    best_result = None
    best_side = float('inf')

    # First, evaluate the initial honeycomb configuration
    init_fun = objective(init_params)
    if init_fun < best_side:
        best_side = init_fun
        best_result = init_params[:]

    # Run differential evolution
    try:
        result = differential_evolution(
            objective, bounds,
            maxiter=150,
            popsize=12,
            tol=1e-6,
            mutation=(0.3, 1.0),
            recombination=0.8,
            seed=42,
            polish=True,
            init='latinhypercube',
            x0=init_params,
            callback=None
        )

        if result.fun < best_side:
            best_side = result.fun
            best_result = result.x[:]

    except Exception as e:
        pass

    # If we have time, do a second round with tighter bounds around best
    if time.time() < deadline - 30 and best_result is not None:
        try:
            # Tighten bounds around best result
            tight_bounds = []
            for i in range(len(bounds)):
                lo, hi = bounds[i]
                if i < 4:  # outer params
                    center = best_result[i]
                    half_range = (hi - lo) / 4
                    tight_bounds.append((center - half_range, center + half_range))
                else:  # inner params
                    center = best_result[i]
                    half_range = 1.5
                    tight_bounds.append((center - half_range, center + half_range))

            result2 = differential_evolution(
                objective, tight_bounds,
                maxiter=100,
                popsize=10,
                tol=1e-7,
                mutation=(0.2, 0.8),
                recombination=0.7,
                seed=123,
                polish=True,
                init='latinhypercube',
                x0=best_result
            )

            if result2.fun < best_side:
                best_side = result2.fun
                best_result = result2.x[:]

        except Exception as e:
            pass

    if best_result is None:
        # Fallback to initial honeycomb
        best_result = init_params

    # Extract final answer
    outer_cx = best_result[0]
    outer_cy = best_result[1]
    outer_side = best_result[2]
    outer_angle = best_result[3]

    inner = []
    for i in range(N):
        ix = best_result[4 + i * 3]
        iy = best_result[5 + i * 3]
        iang = best_result[6 + i * 3]
        inner.append([ix, iy, iang])

    return inner, (outer_cx, outer_cy), outer_side, outer_angle
# EVOLVE-BLOCK-END

"""Unit hexagons in regular hexagon - optimized packing for N=11."""
import time
import math
import numpy as np

N = 11


# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # safety margin before 150s kill

    # --- Geometry helpers ---
    def hex_vertices(cx, cy, side, angle_deg):
        a = math.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = a + k * math.pi / 3.0
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            verts.append((x, y))
        return verts

    def point_in_convex(p, poly):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) < -1e-12:
                return False
        return True

    def polygon_area_signed(verts):
        n = len(verts)
        area = 0.0
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]
            area += x1 * y2 - x2 * y1
        return area / 2.0

    def line_intersect(p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-15:
            return p1
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    def polygon_overlap_area(v1, v2):
        def clip_polygon(subject, clip):
            output = list(subject)
            for i in range(len(clip)):
                if not output:
                    break
                inp = output
                output = []
                S = inp[-1]
                for E in inp:
                    if point_in_convex(E, clip):
                        if not point_in_convex(S, clip):
                            output.append(line_intersect(S, E, clip[i], clip[(i + 1) % len(clip)]))
                        output.append(E)
                    elif point_in_convex(S, clip):
                        output.append(line_intersect(S, E, clip[i], clip[(i + 1) % len(clip)]))
                    S = E
            return output

        clipped = clip_polygon(v1, v2)
        if len(clipped) < 3:
            return 0.0
        return abs(polygon_area_signed(clipped))

    def hex_in_hexagon(inner_verts, outer_verts):
        return all(point_in_convex(v, outer_verts) for v in inner_verts)

    def fast_containment(inner_params, outer_center, outer_side, outer_angle):
        cx, cy = outer_center
        outer_verts = hex_vertices(cx, cy, outer_side, outer_angle)
        for (ix, iy, iang) in inner_params:
            inner_verts = hex_vertices(ix, iy, 1.0, iang)
            if not hex_in_hexagon(inner_verts, outer_verts):
                return False
        return True

    def outer_side_required_fast(inner_params, outer_center, outer_angle):
        cx, cy = outer_center
        apothem_factor = math.sqrt(3) / 2.0
        normal_angles = [math.radians(outer_angle + 30 + k * 60) for k in range(6)]
        normals = [(math.cos(a), math.sin(a)) for a in normal_angles]
        max_val = 0.0
        for (ix, iy, iang) in inner_params:
            inner_verts = hex_vertices(ix, iy, 1.0, iang)
            for vx, vy in inner_verts:
                dx = vx - cx
                dy = vy - cy
                for nx, ny in normals:
                    proj = dx * nx + dy * ny
                    if proj > max_val:
                        max_val = proj
        return max_val / apothem_factor

    def is_valid(inner_params, outer_center, outer_side, outer_angle):
        if not fast_containment(inner_params, outer_center, outer_side, outer_angle):
            return False
        for i in range(N):
            for j in range(i + 1, N):
                vi = hex_vertices(inner_params[i][0], inner_params[i][1], 1.0, inner_params[i][2])
                vj = hex_vertices(inner_params[j][0], inner_params[j][1], 1.0, inner_params[j][2])
                if polygon_overlap_area(vi, vj) > 1e-8:
                    return False
        return True

    # --- Analytic seed: concentric ring packing with 0-degree outer orientation ---
    def analytic_solution():
        """
        Concentric ring packing:
        - 1 center hex at origin
        - 6 ring-1 hexagons at distance sqrt(3) along 0,60,120,180,240,300 degrees
        - 4 ring-2 hexagons at distance 3 along 30,90,150,210,270,330 (4 of them)
        All inner hexagons rotated by 30 degrees.
        Outer hexagon at 0 degrees.
        """
        eps = 1e-9
        centers = []
        centers.append((0.0, 0.0))

        for k in range(6):
            angle = math.radians(k * 60)
            x = (math.sqrt(3) + eps) * math.cos(angle)
            y = (math.sqrt(3) + eps) * math.sin(angle)
            centers.append((x, y))

        # 4 ring-2 hexagons at 90, 150, 270, 330 (top, upper-left, bottom, lower-right)
        # Actually place them at the 4 "corner" positions: 30, 90, 210, 270 won't work
        # Let me place at 90, 150, 270, 330 to be symmetric
        for k in range(4):
            angle = math.radians(90 + k * 90)  # 90, 180, 270, 0 -- no
        # Better: place 4 at 30, 90, 210, 270
        # Actually for symmetry with the 6 ring-1, place at the 4 positions that are
        # not directly opposite to ring-1. Ring-1 is at 0,60,120,180,240,300.
        # Ring-2 at 30,90,150,210,270,330 would be 6, but we only need 4.
        # Place at 90, 150, 270, 330 (top, upper-left, bottom, lower-right)
        for angle_deg in [90, 150, 270, 330]:
            angle = math.radians(angle_deg)
            x = (3.0 + 2 * eps) * math.cos(angle)
            y = (3.0 + 2 * eps) * math.sin(angle)
            centers.append((x, y))

        inner = []
        for x, y in centers:
            inner.append((x, y, 30.0))

        # With outer at 0 degrees, the side is determined by max projection
        # Let's compute it properly
        outer_side = outer_side_required_fast(inner, (0.0, 0.0), 0.0)

        return inner, (0.0, 0.0), outer_side, 0.0

    # Get the analytic solution as baseline
    inner_seed, outer_center_seed, side_seed, angle_seed = analytic_solution()
    best_score = side_seed
    best_result = (inner_seed, outer_center_seed, side_seed, angle_seed)

    # --- Optimization using smooth surrogate + coordinate descent ---
    def compute_objective(params, outer_angle=0.0):
        """
        Compute objective: outer side length + smooth overlap penalty.
        params: 33-dimensional vector (x, y, angle_rad) for each of 11 hexagons.
        """
        inner_params = []
        for i in range(N):
            x = params[3 * i]
            y = params[3 * i + 1]
            a = params[3 * i + 2] * 180.0 / math.pi
            inner_params.append((x, y, a))

        outer_side = outer_side_required_fast(inner_params, (0.0, 0.0), outer_angle)

        # Overlap penalty using smooth surrogate
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                vi = hex_vertices(inner_params[i][0], inner_params[i][1], 1.0, inner_params[i][2])
                vj = hex_vertices(inner_params[j][0], inner_params[j][1], 1.0, inner_params[j][2])
                ov = polygon_overlap_area(vi, vj)
                if ov > 1e-10:
                    overlap_penalty += ov ** 2

        return outer_side + 1000.0 * overlap_penalty

    def initial_guess_from_seed():
        inner, _, _, _ = analytic_solution()
        params = np.zeros(3 * N)
        for i, (x, y, a) in enumerate(inner):
            params[3 * i] = x
            params[3 * i + 1] = y
            params[3 * i + 2] = a * math.pi / 180.0
        return params

    # Try optimization to improve
    if time.time() < deadline - 30:
        params0 = initial_guess_from_seed()
        try:
            from scipy.optimize import minimize

            # L-BFGS-B with finite differences
            result = minimize(
                lambda p: compute_objective(p, 0.0),
                params0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )

            if time.time() < deadline - 15:
                best_params = result.x
                inner_positions = []
                for i in range(N):
                    x = best_params[3 * i]
                    y = best_params[3 * i + 1]
                    a = best_params[3 * i + 2] * 180.0 / math.pi
                    inner_positions.append((x, y, a))

                outer_side = outer_side_required_fast(inner_positions, (0.0, 0.0), 0.0)

                if is_valid(inner_positions, (0.0, 0.0), outer_side, 0.0):
                    if outer_side < best_score:
                        best_score = outer_side
                        best_result = (inner_positions, (0.0, 0.0), outer_side, 0.0)
        except Exception:
            pass

    # Coordinate descent polish
    if time.time() < deadline - 10:
        inner_positions, outer_center, outer_side, outer_angle = best_result
        # Convert to flat array
        params = np.zeros(3 * N)
        for i, (x, y, a) in enumerate(inner_positions):
            params[3 * i] = x
            params[3 * i + 1] = y
            params[3 * i + 2] = a * math.pi / 180.0

        # Coordinate descent with small steps
        step_sizes = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
        for step in step_sizes:
            if time.time() > deadline - 5:
                break
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for i in range(3 * N):
                    for delta in [-step, step]:
                        trial = params.copy()
                        trial[i] += delta
                        obj = compute_objective(trial, 0.0)
                        if obj < compute_objective(params, 0.0) - 1e-12:
                            params = trial
                            improved = True
                            break

        # Check if the polished result is valid and better
        inner_polished = []
        for i in range(N):
            x = params[3 * i]
            y = params[3 * i + 1]
            a = params[3 * i + 2] * 180.0 / math.pi
            inner_polished.append((x, y, a))

        outer_side_polished = outer_side_required_fast(inner_polished, (0.0, 0.0), 0.0)

        if is_valid(inner_polished, (0.0, 0.0), outer_side_polished, 0.0):
            if outer_side_polished < best_score:
                best_score = outer_side_polished
                best_result = (inner_polished, (0.0, 0.0), outer_side_polished, 0.0)

    # Return the best result
    inner_positions, outer_center, outer_side, outer_angle = best_result
    inner = [[p[0], p[1], p[2]] for p in inner_positions]
    return inner, (outer_center[0], outer_center[1]), outer_side, outer_angle
# EVOLVE-BLOCK-END

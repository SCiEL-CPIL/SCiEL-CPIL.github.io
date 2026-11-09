"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    sqrt3 = math.sqrt(3.0)
    rng = random.Random(42)

    # Unit hexagon vertices (side=1, centered at origin, angle=0)
    unit_hex = np.array([[math.cos(i * math.pi / 3.0), math.sin(i * math.pi / 3.0)] for i in range(6)])

    def get_hex_vertices(cx, cy, angle_deg, side=1.0):
        """Get vertices of a regular hexagon in CCW order as numpy array."""
        angle_rad = math.radians(angle_deg)
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        return (R @ unit_hex.T).T * side + np.array([cx, cy])

    def min_distance_between_hexes(c1, a1, c2, a2):
        """Compute minimum distance between two unit hexagons (negative if overlapping)."""
        v1 = get_hex_vertices(*c1, a1)
        v2 = get_hex_vertices(*c2, a2)
        min_dist = float('inf')
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i + 1) % 6]
            for j in range(6):
                q1 = v2[j]
                q2 = v2[(j + 1) % 6]
                # Check point-segment distances
                for pt in [p1, p2, q1, q2]:
                    # Distance from point to segment
                    d = _point_segment_dist(pt, p1, p2)
                    d2 = _point_segment_dist(pt, q1, q2)
                    min_dist = min(min_dist, d, d2)
        # Also check point-in-polygon
        for p in v1:
            if _point_in_convex(p, v2):
                return -1.0
        for p in v2:
            if _point_in_convex(p, v1):
                return -1.0
        return min_dist

    def _point_segment_dist(p, a, b):
        """Distance from point p to segment ab."""
        ab = b - a
        t = np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-15)
        t = max(0, min(1, t))
        proj = a + t * ab
        return np.linalg.norm(p - proj)

    def _point_in_convex(p, poly):
        """Check if point is inside convex polygon."""
        n = len(poly)
        for i in range(n):
            a = poly[i]
            b = poly[(i + 1) % n]
            cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
            if cross < -1e-10:
                return False
        return True

    def enclosing_side_length(centers, angles, outer_angle=0.0):
        """Compute minimal side length of outer hexagon containing all inner hexagons."""
        normals = []
        for k in range(6):
            angle = math.radians(outer_angle + 30 + k * 60)
            normals.append((math.cos(angle), math.sin(angle)))

        max_extent = 0.0
        for i in range(N):
            cx, cy = centers[i]
            inner_angle_rad = math.radians(angles[i])

            for nk in range(6):
                nx, ny = normals[nk]
                n_angle = math.atan2(ny, nx)

                max_dot = -2.0
                for j in range(6):
                    v_angle = inner_angle_rad + j * math.pi / 3.0
                    dot_val = math.cos(v_angle - n_angle)
                    if dot_val > max_dot:
                        max_dot = dot_val

                extent = cx * nx + cy * ny + max_dot
                if extent > max_extent:
                    max_extent = extent

        return 2.0 * max_extent / sqrt3

    def check_overlap(centers, angles):
        """Check if any two inner hexagons overlap."""
        for i in range(N):
            vi = get_hex_vertices(centers[i][0], centers[i][1], angles[i])
            for j in range(i + 1, N):
                vj = get_hex_vertices(centers[j][0], centers[j][1], angles[j])
                # Check point-in-polygon
                for p in vi:
                    if _point_in_convex(p, vj):
                        return False
                for p in vj:
                    if _point_in_convex(p, vi):
                        return False
                # Check edge intersections
                for ii in range(6):
                    p1 = vi[ii]
                    p2 = vi[(ii + 1) % 6]
                    for jj in range(6):
                        p3 = vj[jj]
                        p4 = vj[(jj + 1) % 6]
                        if _segments_intersect(p1, p2, p3, p4):
                            return False
        return True

    def _segments_intersect(p1, p2, p3, p4):
        """Check if segment p1-p2 intersects segment p3-p4."""
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        c1 = cross(p3, p4, p1)
        c2 = cross(p3, p4, p2)
        c3 = cross(p1, p2, p3)
        c4 = cross(p1, p2, p4)

        if ((c1 > 0 and c2 < 0) or (c1 < 0 and c2 > 0)) and \
           ((c3 > 0 and c4 < 0) or (c3 < 0 and c4 > 0)):
            return True
        return False

    def grid_to_cart(q, r):
        x = sqrt3 * q + sqrt3 / 2 * r
        y = 1.5 * r
        return (x, y)

    # Initial hexagonal lattice configuration
    init_grid_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]

    centers = [grid_to_cart(q, r) for q, r in init_grid_coords]
    angles = [0.0] * N

    if not check_overlap(centers, angles):
        centers = [(5.0 * k, 0.0) for k in range(-(N // 2), N - N // 2)]
        angles = [0.0] * N

    best_L = enclosing_side_length(centers, angles, 0.0)
    best_centers = [c[:] for c in centers]
    best_angles = angles[:]
    best_outer_angle = 0.0

    def params_to_config(params):
        """Convert parameter vector to (centers, angles, outer_angle)."""
        centers_opt = []
        angles_opt = []
        for i in range(N):
            centers_opt.append((params[i * 3], params[i * 3 + 1]))
            angles_opt.append(params[i * 3 + 2])
        outer_angle = params[36]
        return centers_opt, angles_opt, outer_angle

    def config_to_params(centers, angles, outer_angle):
        """Convert configuration to parameter vector."""
        params = []
        for i in range(N):
            params.append(centers[i][0])
            params.append(centers[i][1])
            params.append(angles[i])
        params.append(outer_angle)
        return params

    def objective(params):
        """Objective: minimize enclosing side length, penalize overlaps."""
        if time.time() > deadline:
            return 1e10
        centers_opt, angles_opt, outer_angle = params_to_config(params)

        # Check overlaps
        overlap_penalty = 0.0
        for i in range(N):
            vi = get_hex_vertices(centers_opt[i][0], centers_opt[i][1], angles_opt[i])
            for j in range(i + 1, N):
                vj = get_hex_vertices(centers_opt[j][0], centers_opt[j][1], angles_opt[j])
                for p in vi:
                    if _point_in_convex(p, vj):
                        overlap_penalty += 1.0
                for p in vj:
                    if _point_in_convex(p, vi):
                        overlap_penalty += 1.0
                for ii in range(6):
                    p1 = vi[ii]
                    p2 = vi[(ii + 1) % 6]
                    for jj in range(6):
                        p3 = vj[jj]
                        p4 = vj[(jj + 1) % 6]
                        if _segments_intersect(p1, p2, p3, p4):
                            overlap_penalty += 0.5

        L = enclosing_side_length(centers_opt, angles_opt, outer_angle)
        return L + 50.0 * overlap_penalty

    # Multi-start with SLSQP optimizer
    n_starts = 8
    for s in range(n_starts):
        if time.time() > deadline - 15.0:
            break

        if s == 0:
            current_params = config_to_params(centers, angles, 0.0)
        elif s == 1:
            # Perturb lattice
            current_params = config_to_params(centers, angles, 0.0)
            for i in range(N):
                current_params[i * 3] += rng.uniform(-0.3, 0.3)
                current_params[i * 3 + 1] += rng.uniform(-0.3, 0.3)
                current_params[i * 3 + 2] += rng.uniform(-10.0, 10.0)
            current_params[36] = rng.uniform(-10.0, 10.0)
        elif s == 2:
            # Rotated lattice
            current_params = config_to_params(centers, [30.0] * N, 0.0)
        elif s == 3:
            # Circular arrangement
            circ_centers = []
            circ_angles = []
            for i in range(N):
                angle = 2 * math.pi * i / N
                r = 2.5
                circ_centers.append((r * math.cos(angle), r * math.sin(angle)))
                circ_angles.append(0.0)
            current_params = config_to_params(circ_centers, circ_angles, 0.0)
        elif s == 4:
            # Two rows
            row_centers = []
            row_angles = []
            for i in range(6):
                x = -4.5 + i * 1.8
                row_centers.append((x, -1.2))
                row_angles.append(0.0)
            for i in range(6):
                x = -4.5 + i * 1.8
                row_centers.append((x, 1.2))
                row_angles.append(0.0)
            current_params = config_to_params(row_centers, row_angles, 0.0)
        elif s == 5:
            # Random
            current_params = []
            for i in range(N):
                current_params.append(rng.uniform(-3, 3))
                current_params.append(rng.uniform(-3, 3))
                current_params.append(rng.uniform(-60, 60))
            current_params.append(rng.uniform(-60, 60))
        elif s == 6:
            # Perturbed with larger steps
            current_params = config_to_params(centers, angles, 0.0)
            for i in range(N):
                current_params[i * 3] += rng.uniform(-1.0, 1.0)
                current_params[i * 3 + 1] += rng.uniform(-1.0, 1.0)
                current_params[i * 3 + 2] += rng.uniform(-30.0, 30.0)
            current_params[36] = rng.uniform(-30.0, 30.0)
        else:
            # Ring with center
            ring_centers = [(0.0, 0.0)]
            ring_angles = [0.0]
            for i in range(6):
                angle = 2 * math.pi * i / 6
                r = 3.0
                ring_centers.append((r * math.cos(angle), r * math.sin(angle)))
                ring_angles.append(0.0)
            for i in range(5):
                angle = 2 * math.pi * i / 5 + 0.3
                r = 5.0
                ring_centers.append((r * math.cos(angle), r * math.sin(angle)))
                ring_angles.append(0.0)
            current_params = config_to_params(ring_centers, ring_angles, 0.0)

        # Use SLSQP for local optimization
        try:
            result = minimize(
                objective,
                current_params,
                method='SLSQP',
                options={'maxiter': 50, 'ftol': 1e-8}
            )
            if result.success or result.fun < objective(current_params):
                final_params = result.x
                centers_opt, angles_opt, outer_angle = params_to_config(final_params)
                if check_overlap(centers_opt, angles_opt):
                    L = enclosing_side_length(centers_opt, angles_opt, outer_angle)
                    if L < best_L:
                        best_L = L
                        best_centers = centers_opt
                        best_angles = angles_opt
                        best_outer_angle = outer_angle
        except Exception:
            pass

    # Final local search refinement on best solution
    if time.time() < deadline - 5.0:
        current_params = config_to_params(best_centers, best_angles, best_outer_angle)
        step_size = 0.3
        for it in range(100):
            if time.time() > deadline - 3.0:
                break
            improved = False
            best_obj = objective(current_params)
            for _ in range(30):
                trial = current_params[:]
                idx = rng.randint(0, 35)
                trial[idx] += rng.uniform(-step_size, step_size)
                obj = objective(trial)
                if obj < best_obj - 1e-8:
                    current_params = trial
                    best_obj = obj
                    improved = True
                    step_size = max(0.05, step_size * 0.9)
                    break
            if not improved:
                step_size *= 0.85
                if step_size < 0.01:
                    break

        centers_opt, angles_opt, outer_angle = params_to_config(current_params)
        if check_overlap(centers_opt, angles_opt):
            L = enclosing_side_length(centers_opt, angles_opt, outer_angle)
            if L < best_L:
                best_L = L
                best_centers = centers_opt
                best_angles = angles_opt
                best_outer_angle = outer_angle

    # Final validation
    if not check_overlap(best_centers, best_angles):
        best_centers = [grid_to_cart(q, r) for q, r in init_grid_coords]
        best_angles = [0.0] * N
        best_L = enclosing_side_length(best_centers, best_angles, 0.0)
        best_outer_angle = 0.0

    inner = []
    for i in range(N):
        inner.append([best_centers[i][0], best_centers[i][1], best_angles[i]])

    center = (0.0, 0.0)
    side = best_L
    angle = best_outer_angle

    return inner, center, side, angle
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin

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

    def get_objective_value(params):
        inner_params = []
        for i in range(N):
            x = params[3 * i]
            y = params[3 * i + 1]
            a = params[3 * i + 2]
            inner_params.append((x, y, a))

        outer_center = (0.0, 0.0)
        outer_angle = 30.0 # Fixed outer angle
        
        outer_side = outer_side_required_fast(inner_params, outer_center, outer_angle)

        overlap_penalty = 0.0
        for i in range(N):
            vi = hex_vertices(inner_params[i][0], inner_params[i][1], 1.0, inner_params[i][2])
            for j in range(i + 1, N):
                vj = hex_vertices(inner_params[j][0], inner_params[j][1], 1.0, inner_params[j][2])
                ov = polygon_overlap_area(vi, vj)
                if ov > 1e-10:
                    overlap_penalty += ov ** 2

        return outer_side + 100.0 * overlap_penalty

    def analytic_solution():
        eps = 1e-9
        centers = []
        centers.append((0.0, 0.0))
        for k in range(6):
            angle = math.radians(k * 60)
            x = (math.sqrt(3) + eps) * math.cos(angle)
            y = (math.sqrt(3) + eps) * math.sin(angle)
            centers.append((x, y))
        for k in range(4):
            angle = math.radians(30 + k * 60)
            x = (3.0 + 2*eps) * math.cos(angle)
            y = (3.0 + 2*eps) * math.sin(angle)
            centers.append((x, y))
        
        inner = []
        for x, y in centers:
            inner.append((x, y, 30.0))
        
        max_x = (2 + eps) * math.sqrt(3)
        max_y = 4 + 2 * eps
        s_x = 2 * max_x / math.sqrt(3)
        s_y = max_y
        side = max(s_x, s_y) + 1e-9
        return inner, (0.0, 0.0), side, 30.0

    # Get baseline
    inner_seed, outer_center_seed, side_seed, angle_seed = analytic_solution()
    best_params = np.zeros(3 * N)
    for i, (x, y, a) in enumerate(inner_seed):
        best_params[3 * i] = x
        best_params[3 * i + 1] = y
        best_params[3 * i + 2] = a
    best_val = get_objective_value(best_params)
    best_side = side_seed

    # Multi-start Nelder-Mead
    rng = np.random.default_rng(42)
    noise_scale = 1.0
    
    # Run optimization
    while time.time() < deadline - 10:
        # Perturb
        perturbed = best_params + rng.normal(0, noise_scale, size=3 * N)
        
        try:
            res = minimize(
                get_objective_value,
                perturbed,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
            )
            
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
                noise_scale *= 0.9 # Reduce noise on success
            else:
                # If no improvement, keep noise or increase slightly?
                # For now, keep it or reduce slightly to converge
                noise_scale *= 0.95
        except Exception:
            pass

    # Final polish
    if time.time() < deadline - 5:
        try:
            res = minimize(
                get_objective_value,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass

    # Extract final result
    inner_positions = []
    for i in range(N):
        x = best_params[3 * i]
        y = best_params[3 * i + 1]
        a = best_params[3 * i + 2]
        inner_positions.append((x, y, a))

    outer_side = outer_side_required_fast(inner_positions, (0.0, 0.0), 30.0)
    
    # Verify validity
    # Check overlaps
    valid = True
    for i in range(N):
        vi = hex_vertices(inner_positions[i][0], inner_positions[i][1], 1.0, inner_positions[i][2])
        for j in range(i + 1, N):
            vj = hex_vertices(inner_positions[j][0], inner_positions[j][1], 1.0, inner_positions[j][2])
            if polygon_overlap_area(vi, vj) > 1e-8:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to seed
        inner_positions = inner_seed
        outer_side = side_seed

    inner = [[p[0], p[1], p[2]] for p in inner_positions]
    return inner, (0.0, 0.0), outer_side, 30.0

# EVOLVE-BLOCK-END

import math
import time
import random
import numpy as np
from scipy.optimize import differential_evolution

N = 11
SQRT3 = math.sqrt(3)


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s evaluator limit

    # Helper: vertices of unit hexagon centered at origin, rotated by angle_deg
    def unit_hex_vertices(angle_deg):
        verts = []
        for i in range(6):
            theta = math.radians(angle_deg + i * 60.0)
            verts.append((math.cos(theta), math.sin(theta)))
        return verts

    # Helper: get the 6 outward normal directions for a hexagon rotated by angle_deg
    def hex_normal_dirs(angle_deg):
        dirs = []
        for i in range(6):
            theta = math.radians(angle_deg + i * 60.0 + 30.0)
            dirs.append((math.cos(theta), math.sin(theta)))
        return dirs

    # Precompute unit hex vertices for a given angle (centered at origin)
    def get_hex_verts(cx, cy, angle_deg):
        uverts = unit_hex_vertices(angle_deg)
        return [(vx + cx, vy + cy) for vx, vy in uverts]

    # Check if two hexagons are disjoint using SAT
    def are_disjoint(h1_verts, h1_angle, h2_verts, h2_angle):
        axes = hex_normal_dirs(h1_angle) + hex_normal_dirs(h2_angle)
        for nx, ny in axes:
            min1 = min(vx * nx + vy * ny for vx, vy in h1_verts)
            max1 = max(vx * nx + vy * ny for vx, vy in h1_verts)
            min2 = min(vx * nx + vy * ny for vx, vy in h2_verts)
            max2 = max(vx * nx + vy * ny for vx, vy in h2_verts)
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return True
        return False

    # Calculate minimum outer hexagon side length (centered at origin, rotated by outer_angle)
    def calc_outer_side(hexagons, outer_angle_deg):
        base_angles = [30, 90, 150, 210, 270, 330]
        axis_dirs = []
        for a in base_angles:
            rad = math.radians(a + outer_angle_deg)
            axis_dirs.append((math.cos(rad), math.sin(rad)))

        max_proj = 0.0
        for hex_verts in hexagons:
            for vx, vy in hex_verts:
                for nx, ny in axis_dirs:
                    proj = abs(vx * nx + vy * ny)
                    if proj > max_proj:
                        max_proj = proj

        return 2.0 * max_proj / SQRT3

    # Objective function for the optimizer
    # params: [x1,y1,a1, x2,y2,a2, ..., xN,yN,aN, ox, oy, oa]
    def objective(params):
        if time.time() > deadline - 2.0:
            return 1e6

        centers = []
        angles = []
        for i in range(N):
            centers.append((params[i * 3], params[i * 3 + 1]))
            angles.append(params[i * 3 + 2])
        
        ox = params[N * 3]
        oy = params[N * 3 + 1]
        oa = params[N * 3 + 2]

        # Shift all centers so outer hexagon center is at origin for calculation
        shifted = [(cx - ox, cy - oy) for cx, cy in centers]

        hexagons = []
        for i in range(N):
            hexagons.append(get_hex_verts(shifted[i][0], shifted[i][1], angles[i]))

        # Check overlaps
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                if not are_disjoint(hexagons[i], angles[i], hexagons[j], angles[j]):
                    overlap_penalty += 500.0

        if overlap_penalty > 0:
            return overlap_penalty + calc_outer_side(hexagons, oa)

        return calc_outer_side(hexagons, oa)

    # Initial hexagonal lattice configuration (flower + extensions)
    def axial_to_cart(q, r, spacing=1.0):
        x = SQRT3 * spacing * (q + r / 2.0)
        y = 1.5 * spacing * r
        return x, y

    # Create initial parameter vector
    def make_initial_params(config):
        params = []
        for q, r in config:
            x, y = axial_to_cart(q, r)
            params.extend([x, y, 0.0])
        params.extend([0.0, 0.0, 0.0])  # outer center and angle
        return params

    # Config 1: Flower + 4 extensions
    coords1 = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (2, 0), (2, -1), (2, 1), (1, 1)]
    # Config 2: Compact block
    coords2 = [(-1, -1), (0, -1), (1, -1), (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0), (-1, 1), (0, 1), (1, 1)]

    best_score = float('inf')
    best_params = None

    # Use differential evolution with multiple restarts if time permits
    bounds = []
    for i in range(N):
        bounds.append((x_min := -6.0, x_max := 6.0))
        bounds.append((-6.0, 6.0))
        bounds.append((-30.0, 30.0))  # angle in degrees (hexagon has 60° symmetry, so -30 to 30 is sufficient)
    bounds.append((-3.0, 3.0))  # outer center x
    bounds.append((-3.0, 3.0))  # outer center y
    bounds.append((-30.0, 30.0))  # outer angle

    # Run differential evolution
    configs = [coords1, coords2]
    
    for ci, config in enumerate(configs):
        if time.time() > deadline - 60.0:
            break
            
        initial = make_initial_params(config)
        
        try:
            result = differential_evolution(
                objective, bounds,
                seed=42 + ci,
                maxiter=100,
                popsize=15,
                tol=1e-4,
                init='sobol',
                polish=True,
                maxfev=5000
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
                print(f"Config {ci}: DE found score {result.fun:.6f}")
        except Exception as e:
            print(f"DE failed for config {ci}: {e}")
            # Fallback: use initial config as best
            if best_params is None:
                best_params = np.array(initial)
                best_score = objective(best_params)

    # If DE didn't find anything, use the first config
    if best_params is None:
        best_params = np.array(make_initial_params(coords1))
        best_score = objective(best_params)

    # Extract final parameters
    final_params = best_params
    centers = []
    angles = []
    for i in range(N):
        centers.append((final_params[i * 3], final_params[i * 3 + 1]))
        angles.append(final_params[i * 3 + 2])
    
    ox = final_params[N * 3]
    oy = final_params[N * 3 + 1]
    oa = final_params[N * 3 + 2]

    # Shift inner centers so outer hexagon center is at origin
    inner_list = []
    for i in range(N):
        ix = centers[i][0] - ox
        iy = centers[i][1] - oy
        inner_list.append([ix, iy, angles[i]])

    # Final verification and side calculation
    hexagons_final = []
    for i in range(N):
        hexagons_final.append(get_hex_verts(inner_list[i][0], inner_list[i][1], inner_list[i][2]))

    # Fine-tune outer angle
    best_side = float('inf')
    best_oa = oa
    for oa_deg in range(-300, 301):
        oa_test = oa_deg * 0.1
        side = calc_outer_side(hexagons_final, oa_test)
        if side < best_side:
            best_side = side
            best_oa = oa_test

    # Final validity check
    valid = True
    for i in range(N):
        for j in range(i + 1, N):
            if not are_disjoint(hexagons_final[i], inner_list[i][2], hexagons_final[j], inner_list[j][2]):
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fallback to a known valid configuration
        print("Final result invalid, using fallback")
        inner_list = []
        for q, r in coords1:
            x, y = axial_to_cart(q, r)
            inner_list.append([x, y, 0.0])
        # Center it
        cx_avg = sum(p[0] for p in inner_list) / N
        cy_avg = sum(p[1] for p in inner_list) / N
        inner_list = [[x - cx_avg, y - cy_avg, a] for x, y, a in inner_list]
        hexagons_fb = [get_hex_verts(p[0], p[1], p[2]) for p in inner_list]
        best_side = calc_outer_side(hexagons_fb, 0.0)
        best_oa = 0.0

    return (inner_list, (0.0, 0.0), best_side, best_oa)


# EVOLVE-BLOCK-END

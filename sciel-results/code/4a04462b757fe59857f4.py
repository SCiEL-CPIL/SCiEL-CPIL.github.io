"""Packing unit hexagons inside a regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)
DEADLINE = 135.0  # safety margin before 150s kill

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts

def compute_required_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_deg)
    outer_center: (x, y)
    outer_angle_deg: rotation of the outer hexagon
    """
    if not inner:
        return 0.0
    ox, oy = outer_center
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    max_projection = 0.0
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj
    return 2.0 * max_projection / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=50.0):
    """
    Smooth approximation using log-sum-exp for the max projection.
    Provides better gradient information for optimization.
    """
    if not inner:
        return 0.0
    ox, oy = outer_center
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    projections = []
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                projections.append(proj)
    if not projections:
        return 0.0
    max_proj = max(projections)
    if max_proj < -1000:
        return 0.0
    scaled = [beta * (p - max_proj) for p in projections]
    sum_exp = sum(math.exp(s) for s in scaled)
    smooth_max = max_proj + math.log(sum_exp) / beta
    return 2.0 * smooth_max / SQRT3

def sat_overlap_check(v1, v2, eps=1e-9):
    """
    Check if two convex polygons overlap using SAT.
    Returns True if they overlap.
    """
    axes = []
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    for i in range(len(v2)):
        p1 = v2[i]
        p2 = v2[(i + 1) % len(v2)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    for axis in axes:
        min1 = min(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        max1 = max(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        min2 = min(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        max2 = max(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    return True

def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    verts_list = []
    for i in range(n):
        verts_list.append(get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2])))
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap_check(verts_list[i], verts_list[j], eps):
                return False
    return True

def construct():
    """
    Place N unit hexagons inside a regular hexagon, minimizing the outer side length.
    Returns: (inner, center, side, angle)
    """
    start_time = time.time()
    deadline = start_time + DEADLINE

    # 1. Define 11 hexagons on a hexagonal lattice
    # Center + Ring 1 (6) + 4 from Ring 2 for compact cluster
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-1, 2), (-2, 1)
    ]
    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]

    # 2. Convert axial to Cartesian
    epsilon = 1e-6
    scale = 1.0 + epsilon
    centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        centers.append((x, y))

    initial_inner = [[x, y, 0.0] for x, y in centers]

    best_side = float('inf')
    best_inner = [list(x) for x in initial_inner]
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0

    cx_avg = sum(c[0] for c in centers) / len(centers)
    cy_avg = sum(c[1] for c in centers) / len(centers)

    # 3. Optimize outer hexagon using scipy with smooth surrogate + exact polish
    def objective_outer_smooth(params):
        cx, cy, angle_deg = params
        return compute_required_side_smooth(initial_inner, (cx, cy), angle_deg, beta=50.0)

    def objective_outer_exact(params):
        cx, cy, angle_deg = params
        return compute_required_side(initial_inner, (cx, cy), angle_deg)

    # Multiple diverse initial angles to avoid local minima
    init_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]

    for init_angle in init_angles:
        if time.time() > deadline - 25:
            break
        init_params = [cx_avg, cy_avg, float(init_angle)]
        try:
            # First pass: smooth optimization for basin finding
            res_smooth = minimize(objective_outer_smooth, init_params, method='Nelder-Mead',
                                  options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8, 'maxfev': 800})
            # Second pass: exact optimization for precision
            res_exact = minimize(objective_outer_exact, res_smooth.x, method='Nelder-Mead',
                                 options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 500})
            val = res_exact.fun
            if val < best_side:
                best_side = val
                best_outer_center = (res_exact.x[0], res_exact.x[1])
                best_outer_angle = res_exact.x[2]
        except Exception:
            pass

    # 4. Joint optimization of inner positions and outer container
    # Use scipy to optimize all inner positions + outer center + angle simultaneously
    # with a smooth objective that includes a penalty for overlaps

    def joint_objective_smooth(params):
        """
        params: [x1, y1, a1, x2, y2, a2, ..., xN, yN, aN, cx, cy, angle]
        Returns smooth outer side length + overlap penalty.
        """
        inner = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            a = params[3*i+2]
            inner.append((x, y, a))
        cx = params[3*N]
        cy = params[3*N+1]
        angle = params[3*N+2]

        # Smooth outer side
        side = compute_required_side_smooth(inner, (cx, cy), angle, beta=30.0)

        # Overlap penalty: use minimum distance between hexagons
        # For each pair, compute minimum separation using SAT-like approach
        # Simplified: use center-to-center distance vs minimum distance for non-overlap
        # For regular hexagons of side 1, minimum center distance for non-overlap is sqrt(3)
        # when aligned, but varies with relative rotation
        # Use a smooth penalty based on distance
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                xi, yi, ai = inner[i]
                xj, yj, aj = inner[j]
                dx = xi - xj
                dy = yi - yj
                dist = math.sqrt(dx*dx + dy*dy)
                # Minimum non-overlap distance depends on relative angle
                # For simplicity, use sqrt(3) as baseline (aligned case)
                # This is conservative for some rotations
                min_dist = SQRT3 * (1.0 - 1e-6)
                if dist < min_dist:
                    # Smooth penalty: (min_dist - dist)^2
                    penalty += (min_dist - dist) ** 2

        return side + 10.0 * penalty

    def joint_objective_exact(params):
        """Exact version for final polish."""
        inner = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            a = params[3*i+2]
            inner.append((x, y, a))
        cx = params[3*N]
        cy = params[3*N+1]
        angle = params[3*N+2]

        # Check disjointness
        if not check_all_disjoint(inner, eps=1e-5):
            return 1000.0  # Large penalty for overlap

        side = compute_required_side(inner, (cx, cy), angle)
        return side

    # Initial params for joint optimization
    init_params = []
    for x, y, a in best_inner:
        init_params.extend([x, y, a])
    init_params.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])

    # Try joint optimization from the best outer-only solution
    if time.time() < deadline - 15:
        try:
            res_joint = minimize(joint_objective_smooth, init_params, method='Nelder-Mead',
                                 options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-7, 'maxfev': 3000})
            # Extract and verify
            inner_from_res = []
            for i in range(N):
                inner_from_res.append([res_joint.x[3*i], res_joint.x[3*i+1], res_joint.x[3*i+2]])
            cx_r = res_joint.x[3*N]
            cy_r = res_joint.x[3*N+1]
            angle_r = res_joint.x[3*N+2]

            if check_all_disjoint(inner_from_res, eps=1e-5):
                side_r = compute_required_side(inner_from_res, (cx_r, cy_r), angle_r)
                if side_r < best_side - 1e-6:
                    best_side = side_r
                    best_inner = inner_from_res
                    best_outer_center = (cx_r, cy_r)
                    best_outer_angle = angle_r
        except Exception:
            pass

    # 5. Final high-precision polish pass
    if time.time() < deadline - 5:
        # Fine-tune outer container for the best inner configuration
        current_inner = best_inner
        current_center = best_outer_center
        current_angle = best_outer_angle

        def polish_objective(params):
            cx, cy, angle = params
            return compute_required_side(current_inner, (cx, cy), angle)

        init_polish = [current_center[0], current_center[1], current_angle]
        try:
            res_polish = minimize(polish_objective, init_polish, method='Nelder-Mead',
                                  options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 2000})
            if res_polish.fun < best_side:
                best_side = res_polish.fun
                best_outer_center = (res_polish.x[0], res_polish.x[1])
                best_outer_angle = res_polish.x[2]
        except Exception:
            pass

        # Also try small perturbations of inner positions with exact outer re-optimization
        if time.time() < deadline - 3:
            np.random.seed(42)
            current_side = best_side
            step_size = 0.02
            for _ in range(200):
                if time.time() > deadline - 3:
                    break
                idx = np.random.randint(N)
                orig = list(best_inner[idx])
                new_x = orig[0] + np.random.uniform(-step_size, step_size)
                new_y = orig[1] + np.random.uniform(-step_size, step_size)
                new_ang = orig[2] + np.random.uniform(-3, 3)
                trial_inner = [list(x) for x in best_inner]
                trial_inner[idx] = [new_x, new_y, new_ang]

                if check_all_disjoint(trial_inner, eps=1e-5):
                    # Quick outer optimization
                    def quick_obj(params):
                        return compute_required_side(trial_inner, (params[0], params[1]), params[2])
                    try:
                        res_q = minimize(quick_obj, [best_outer_center[0], best_outer_center[1], best_outer_angle],
                                         method='Nelder-Mead',
                                         options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 300})
                        if res_q.fun < current_side - 1e-7:
                            current_side = res_q.fun
                            best_side = res_q.fun
                            best_inner = trial_inner
                            best_outer_center = (res_q.x[0], res_q.x[1])
                            best_outer_angle = res_q.x[2]
                            step_size = max(step_size * 0.9, 0.005)
                    except Exception:
                        pass
                else:
                    step_size = max(step_size * 0.9, 0.005)

    # Add tiny epsilon to side to ensure validity
    best_side += 1e-9

    return best_inner, best_outer_center, best_side, best_outer_angle

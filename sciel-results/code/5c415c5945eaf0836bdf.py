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

def compute_required_side_fast(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    Vectorized version.
    """
    if not inner:
        return 0.0

    ox, oy = outer_center
    # Normals for a hexagon with angle 0
    thetas = np.radians(30 + np.arange(6) * 60)
    normals = np.column_stack((np.cos(thetas), np.sin(thetas))) # (6, 2)

    # Rotate outer frame
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))

    # Collect all vertices
    # inner: list of [x, y, angle_deg]
    verts_list = []
    for ix, iy, iangle in inner:
        angles = math.radians(iangle) + np.arange(6) * (math.pi / 3.0)
        vx = ix + np.cos(angles)
        vy = iy + np.sin(angles)
        verts_list.append(np.column_stack((vx, vy)))
    
    all_verts = np.vstack(verts_list) # (6*N, 2)
    
    # Translate
    all_verts[:, 0] -= ox
    all_verts[:, 1] -= oy
    
    # Rotate
    rot_mat = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    all_verts_rot = all_verts @ rot_mat.T
    
    # Project
    projs = all_verts_rot @ normals.T # (6*N, 6)
    
    max_proj = np.max(projs)
    
    return 2.0 * max_proj / SQRT3

def sat_overlap_check(v1, v2, eps=1e-9):
    """
    Check if two convex polygons (defined by their vertices) overlap using SAT.
    v1, v2: lists of (x, y) tuples.
    Returns True if they overlap (including touching).
    """
    axes = []

    # Get axes from edges of v1
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0] ** 2 + normal[1] ** 2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))

    # Get axes from edges of v2
    for i in range(len(v2)):
        p1 = v2[i]
        p2 = v2[(i + 1) % len(v2)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0] ** 2 + normal[1] ** 2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))

    for axis in axes:
        min1 = min(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        max1 = max(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        min2 = min(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        max2 = max(p[0] * axis[0] + p[1] * axis[1] for p in v2)

        # Check for gap
        if max1 < min2 + eps or max2 < min1 + eps:
            return False

    return True

def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    for i in range(n):
        for j in range(i + 1, n):
            vi = get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
            vj = get_hex_vertices(inner[j][0], inner[j][1], 1.0, math.radians(inner[j][2]))
            if sat_overlap_check(vi, vj, eps):
                return False
    return True

def check_inside_outer(inner, outer_center, outer_side, outer_angle_deg, eps=1e-6):
    """Check all inner hexagons are inside the outer hexagon."""
    ox, oy = outer_center
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))

    apothem = outer_side * SQRT3 / 2.0

    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a

            for i in range(6):
                theta = math.radians(30 + i * 60)
                nx = math.cos(theta)
                ny = math.sin(theta)
                if lx * nx + ly * ny > apothem + eps:
                    return False
    return True

def construct():
    start_time = time.time()

    # 1. Define 11 hexagons on a hexagonal lattice
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

    # Initial layout: All hexagons aligned (angle 0)
    current_inner = [[x, y, 0.0] for x, y in centers]
    
    # Initial outer estimate
    cx_avg = sum(c[0] for c in centers) / len(centers)
    cy_avg = sum(c[1] for c in centers) / len(centers)
    current_outer_center = (cx_avg, cy_avg)
    current_outer_angle = 0.0
    current_side = compute_required_side_fast(current_inner, current_outer_center, current_outer_angle)

    best_side = current_side
    best_inner = [list(x) for x in current_inner]
    best_outer_center = current_outer_center
    best_outer_angle = current_outer_angle

    # 3. Alternating Minimization
    # Step A: Optimize Outer (center, angle) for fixed Inner
    # Step B: Optimize Inner (positions, angles) for fixed Outer
    
    # We will iterate a few times
    
    def obj_outer(params, inner_list):
        cx, cy, angle_deg = params
        return compute_required_side_fast(inner_list, (cx, cy), angle_deg)

    def obj_inner(params, outer_center, outer_angle):
        # Unpack
        inner_list = []
        for i in range(N):
            x = params[i * 3]
            y = params[i * 3 + 1]
            a = params[i * 3 + 2]
            inner_list.append([x, y, a])
        
        side = compute_required_side_fast(inner_list, outer_center, outer_angle)
        
        # Penalty for overlaps (approximate)
        penalty = 0.0
        min_dist = 1.7 # Slightly less than sqrt(3) ~ 1.732 to allow rotation
        for i in range(N):
            for j in range(i + 1, N):
                xi, yi, _ = inner_list[i]
                xj, yj, _ = inner_list[j]
                dist = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
        
        return side + 10.0 * penalty

    # Initial params for inner
    init_inner_params = []
    for x, y, a in current_inner:
        init_inner_params.extend([x, y, a])

    # Run a few rounds of alternating optimization
    for round_idx in range(5):
        if time.time() - start_time > DEADLINE - 30:
            break

        # Optimize Outer
        if time.time() - start_time > DEADLINE - 10:
            break
            
        try:
            res_outer = minimize(
                lambda p: obj_outer(p, current_inner),
                [current_outer_center[0], current_outer_center[1], current_outer_angle],
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'maxfev': 500}
            )
            if res_outer.fun < current_side:
                current_outer_center = (res_outer.x[0], res_outer.x[1])
                current_outer_angle = res_outer.x[2]
                current_side = res_outer.fun
        except Exception:
            pass

        # Optimize Inner
        if time.time() - start_time > DEADLINE - 10:
            break
            
        try:
            # Use current inner as start
            res_inner = minimize(
                lambda p: obj_inner(p, current_outer_center, current_outer_angle),
                init_inner_params,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8, 'maxfev': 800}
            )
            
            # Extract new inner
            new_inner = []
            for i in range(N):
                new_inner.append([res_inner.x[i * 3], res_inner.x[i * 3 + 1], res_inner.x[i * 3 + 2]])
            
            # Check validity
            if check_all_disjoint(new_inner):
                # Recalculate side with new inner and current outer
                new_side = compute_required_side_fast(new_inner, current_outer_center, current_outer_angle)
                if check_inside_outer(new_inner, current_outer_center, new_side, current_outer_angle):
                    # Update best
                    if new_side < best_side:
                        best_side = new_side
                        best_inner = new_inner
                        best_outer_center = current_outer_center
                        best_outer_angle = current_outer_angle
                    
                    # Update current for next iteration
                    current_inner = new_inner
                    current_side = new_side
                    init_inner_params = [v for sublist in new_inner for v in sublist]
        except Exception:
            pass

    # 4. Final Polish
    # If we have a valid best, try to polish the outer again
    if check_all_disjoint(best_inner) and check_inside_outer(best_inner, best_outer_center, best_side, best_outer_angle):
        try:
            res_polish = minimize(
                lambda p: obj_outer(p, best_inner),
                [best_outer_center[0], best_outer_center[1], best_outer_angle],
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 1000}
            )
            if res_polish.fun < best_side:
                best_outer_center = (res_polish.x[0], res_polish.x[1])
                best_outer_angle = res_polish.x[2]
                best_side = res_polish.fun
        except Exception:
            pass

    # 5. Fallback
    if not check_all_disjoint(best_inner) or not check_inside_outer(best_inner, best_outer_center, best_side, best_outer_angle):
        # Use original lattice with safe margin
        scale_fallback = 1.05
        fallback_centers = []
        for q, r in axial_coords:
            x = 1.5 * scale_fallback * q
            y = SQRT3 * scale_fallback * (r + q / 2.0)
            fallback_centers.append((x, y))
        fallback_inner = [[x, y, 0.0] for x, y in fallback_centers]
        final_center = (0.0, 0.0)
        final_angle = 0.0
        final_side = compute_required_side_fast(fallback_inner, final_center, final_angle)
        return fallback_inner, final_center, final_side, final_angle

    return best_inner, best_outer_center, best_side, best_outer_angle

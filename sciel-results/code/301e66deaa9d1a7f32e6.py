import math
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 11
SQRT3 = math.sqrt(3)

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
    # Normals for a hexagon with angle 0 (vertices at 0, 60, ...):
    # Outward normals are at 30, 90, 150, 210, 270, 330 degrees.
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_projection = 0.0
    
    # Rotate outer frame to match outer_angle
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            # Translate to outer center
            dx = v[0] - ox
            dy = v[1] - oy
            # Rotate points by -outer_angle to align with standard hex axes
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj
    
    # S * sqrt(3) / 2 >= max_projection
    # S >= 2 * max_projection / sqrt(3)
    return 2.0 * max_projection / SQRT3

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
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    # Get axes from edges of v2
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

def compute_overlap_penalty(inner, threshold=1e-4):
    """
    Compute a penalty for overlapping hexagons using center distance.
    For two unit hexagons, if they are aligned, the minimum center distance to be disjoint is sqrt(3).
    For arbitrary rotations, this is conservative but safe.
    """
    penalty = 0.0
    n = len(inner)
    min_dist = SQRT3 - 0.01
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, _ = inner[i]
            xj, yj, _ = inner[j]
            dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dist < min_dist:
                overlap = min_dist - dist
                penalty += (overlap * 50.0)**2
    return penalty

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin for 150s limit

    # 1. Define 11 hexagons on a hexagonal lattice as initial guess
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
    initial_inner = [[x, y, 0.0] for x, y in centers]

    best_side = float('inf')
    best_inner = initial_inner
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0

    # 3. Joint optimization using differential evolution for global search
    # Variables: for each hexagon (x, y, angle_deg) + outer (cx, cy, angle_deg)
    # Total: N*3 + 3 = 36 variables

    def objective_joint(params):
        inner_list = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            a = params[i*3+2]
            inner_list.append([x, y, a])

        ox = params[N*3]
        oy = params[N*3+1]
        oa = params[N*3+2]

        side = compute_required_side(inner_list, (ox, oy), oa)
        penalty = compute_overlap_penalty(inner_list)
        return side + penalty

    # Define bounds for differential evolution
    # Centers roughly in range [-3, 3], angles in [0, 180] (hexagon symmetry)
    bounds = []
    for i in range(N):
        bounds.append((-3.0, 3.0))  # x
        bounds.append((-3.0, 3.0))  # y
        bounds.append((0.0, 180.0))  # angle

    # Outer hexagon center and angle
    bounds.append((-3.0, 3.0))  # ox
    bounds.append((-3.0, 3.0))  # oy
    bounds.append((0.0, 60.0))  # outer angle (hexagon has 60-degree symmetry)

    # Run differential evolution with limited iterations to stay within time
    if time.time() < deadline - 30:
        try:
            # Use moderate population size and max iterations
            de_result = differential_evolution(
                objective_joint,
                bounds=bounds,
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                polish=False,  # We'll polish separately
                init='sobol'
            )
            if de_result.fun < best_side:
                # Extract best solution
                de_inner = []
                for i in range(N):
                    x = de_result.x[i*3]
                    y = de_result.x[i*3+1]
                    a = de_result.x[i*3+2]
                    de_inner.append([x, y, a])

                de_ox = de_result.x[N*3]
                de_oy = de_result.x[N*3+1]
                de_oa = de_result.x[N*3+2]

                # Verify with SAT
                if check_all_disjoint(de_inner):
                    side_val = compute_required_side(de_inner, (de_ox, de_oy), de_oa)
                    if side_val < best_side:
                        best_side = side_val
                        best_inner = de_inner
                        best_outer_center = (de_ox, de_oy)
                        best_outer_angle = de_oa
        except Exception:
            pass

    # 4. Polish with Nelder-Mead from best solution found
    if time.time() < deadline - 15:
        init_params = []
        for x, y, a in best_inner:
            init_params.extend([x, y, a])
        init_params.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])

        try:
            nm_result = minimize(
                objective_joint,
                init_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 1000}
            )

            nm_inner = []
            for i in range(N):
                x = nm_result.x[i*3]
                y = nm_result.x[i*3+1]
                a = nm_result.x[i*3+2]
                nm_inner.append([x, y, a])

            nm_ox = nm_result.x[N*3]
            nm_oy = nm_result.x[N*3+1]
            nm_oa = nm_result.x[N*3+2]

            # Verify with SAT
            if check_all_disjoint(nm_inner):
                side_val = compute_required_side(nm_inner, (nm_ox, nm_oy), nm_oa)
                if side_val < best_side:
                    best_side = side_val
                    best_inner = nm_inner
                    best_outer_center = (nm_ox, nm_oy)
                    best_outer_angle = nm_oa
        except Exception:
            pass

    # 5. Try multiple random restarts if time permits
    if time.time() < deadline - 20:
        for trial in range(3):
            if time.time() > deadline - 10:
                break

            # Random perturbation around best solution
            trial_params = []
            for x, y, a in best_inner:
                trial_params.extend([
                    x + np.random.uniform(-0.2, 0.2),
                    y + np.random.uniform(-0.2, 0.2),
                    a + np.random.uniform(-10, 10)
                ])
            trial_params.extend([
                best_outer_center[0] + np.random.uniform(-0.2, 0.2),
                best_outer_center[1] + np.random.uniform(-0.2, 0.2),
                best_outer_angle + np.random.uniform(-5, 5)
            ])

            try:
                res = minimize(
                    objective_joint,
                    trial_params,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 600}
                )

                trial_inner = []
                for i in range(N):
                    x = res.x[i*3]
                    y = res.x[i*3+1]
                    a = res.x[i*3+2]
                    trial_inner.append([x, y, a])

                t_ox = res.x[N*3]
                t_oy = res.x[N*3+1]
                t_oa = res.x[N*3+2]

                if check_all_disjoint(trial_inner):
                    side_val = compute_required_side(trial_inner, (t_ox, t_oy), t_oa)
                    if side_val < best_side:
                        best_side = side_val
                        best_inner = trial_inner
                        best_outer_center = (t_ox, t_oy)
                        best_outer_angle = t_oa
            except Exception:
                pass

    # Add tiny epsilon to side to ensure validity
    best_side += 1e-9

    return best_inner, best_outer_center, best_side, best_outer_angle

"""Packing unit hexagons inside a regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)
DEADLINE = 135.0  # safety margin before 150s kill


def logsumexp(a, b=1.0):
    """Stable log-sum-exp for array a with temperature b."""
    a = np.asarray(a, dtype=np.float64)
    amax = np.max(a)
    if not np.isfinite(amax):
        return amax
    return amax + math.log(np.sum(np.exp((a - amax) / b))) * b

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, temp=0.05):
    """
    Compute a smooth approximation of the minimum outer side length using log-sum-exp.
    The required side is 2*max(|proj|)/sqrt(3) where proj ranges over all vertex-normal pairs.
    We use soft-max over signed projections to smoothly approximate max(|proj|).
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

    # Collect signed projections; we need max of |proj| = max(max(proj), max(-proj))
    pos_projections = []
    neg_projections = []

    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a

            for nx, ny in normals:
                proj = lx * nx + ly * ny
                pos_projections.append(proj)
                neg_projections.append(-proj)

    if not pos_projections:
        return 0.0

    # max(|proj|) ≈ softmax(max(pos), max(neg)) via log-sum-exp on combined set
    # Use logsumexp over all 2M values to approximate max(|proj|)
    all_signed = pos_projections + neg_projections
    max_abs_proj_smooth = logsumexp(all_signed, temp)
    
    return 2.0 * max_abs_proj_smooth / SQRT3


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


def sat_overlap_check(v1, v2, eps=1e-9):
    """
    Check if two convex polygons (defined by their vertices) overlap using SAT.
    v1, v2: lists of (x, y) tuples.
    Returns True if they overlap (including touching).
    """
    axes = []

    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0] ** 2 + normal[1] ** 2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))

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

    # Initial layout: 11 hexagons in a compact hexagonal cluster
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2),
    ]

    scale = 1.0 + 1e-6

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

    # Phase 0: Differential Evolution for global exploration
    # This helps escape local minima before SA refines the solution
    if time.time() - start_time < DEADLINE - 60:
        try:
            from scipy.optimize import differential_evolution

            def de_objective(params):
                inner_list = []
                for i in range(N):
                    x = params[i * 3]
                    y = params[i * 3 + 1]
                    a = params[i * 3 + 2]
                    inner_list.append([x, y, a])

                cx = params[N * 3]
                cy = params[N * 3 + 1]
                angle = params[N * 3 + 2]

                side = compute_required_side_smooth(inner_list, (cx, cy), angle, temp=0.05)

                penalty = 0.0
                min_dist = SQRT3 - 0.02
                for i in range(N):
                    for j in range(i + 1, N):
                        xi, yi, ai = inner_list[i]
                        xj, yj, aj = inner_list[j]
                        dist = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                        if dist < min_dist:
                            penalty += (min_dist - dist) ** 2

                return side + 200.0 * penalty

            # Bounds: inner positions in [-5,5], angles in [-180,180], outer center in [-2,2], angle in [-90,90]
            bounds = []
            for i in range(N):
                bounds.extend([(-5.0, 5.0), (-5.0, 5.0), (-180.0, 180.0)])
            bounds.extend([(-2.0, 2.0), (-2.0, 2.0), (-90.0, 90.0)])

            # Use the initial layout as a starting point
            x0 = init_params[:]

            de_result = differential_evolution(
                de_objective,
                bounds,
                x0=x0,
                maxiter=200,
                popsize=15,
                tol=1e-8,
                seed=42,
                polish=False,
                updating='immediate',
                mutation=(0.3, 1.0),
                recombination=0.8,
                init='latinhypercube',
            )

            if de_result.success and de_result.fun < current_obj:
                current_params = de_result.x.tolist()
                current_obj = de_result.fun

                # Validate DE result
                inner_de = []
                for i in range(N):
                    inner_de.append([current_params[i*3], current_params[i*3+1], current_params[i*3+2]])
                cx_de = current_params[N*3]
                cy_de = current_params[N*3+1]
                ang_de = current_params[N*3+2]

                if fast_overlap_check(inner_de):
                    valid_de, inner_de2, center_de, angle_de = sa_validate(current_params)
                    if valid_de:
                        side_de = compute_required_side(inner_de2, center_de, angle_de)
                        if side_de < best_sa_obj:
                            best_sa_valid = True
                            best_sa_inner = inner_de2
                            best_sa_center = center_de
                            best_sa_angle = angle_de
                            best_sa_obj = side_de
        except Exception:
            pass

    # Phase 1: Simulated Annealing with direct containment objective
    # Key structural change: instead of optimizing outer params for fixed inner,
    # we directly minimize the required side length via SA moves on inner positions/rotations
    # AND outer center/angle simultaneously.

    def sa_objective(params):
        """Direct objective: compute smooth required side length + overlap penalty."""
        inner_list = []
        for i in range(N):
            x = params[i * 3]
            y = params[i * 3 + 1]
            a = params[i * 3 + 2]
            inner_list.append([x, y, a])

        cx = params[N * 3]
        cy = params[N * 3 + 1]
        angle = params[N * 3 + 2]

        # Use smooth approximation for the side length to guide SA
        side = compute_required_side_smooth(inner_list, (cx, cy), angle, temp=0.05)

        # Overlap penalty: use minimum distance between hexagon edges as proxy.
        # For two unit hexagons with arbitrary rotations, the minimum center distance
        # for non-overlap depends on relative orientation. A safe lower bound is 2*apothem = sqrt(3).
        # We use a tighter threshold with quadratic penalty.
        penalty = 0.0
        min_dist = SQRT3 - 0.02
        for i in range(N):
            for j in range(i + 1, N):
                xi, yi, ai = inner_list[i]
                xj, yj, aj = inner_list[j]
                dist = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2

        return side + 200.0 * penalty

    def sa_validate(params):
        """Check if solution is valid (no overlaps, all inside)."""
        inner_list = []
        for i in range(N):
            inner_list.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])

        cx = params[N * 3]
        cy = params[N * 3 + 1]
        angle = params[N * 3 + 2]

        side = compute_required_side(inner_list, (cx, cy), angle)

        if not check_all_disjoint(inner_list):
            return False, None, None, None

        if not check_inside_outer(inner_list, (cx, cy), side, angle):
            return False, None, None, None

        return True, inner_list, (cx, cy), angle

    def fast_overlap_check(inner_list, min_dist=SQRT3 - 0.01):
        """Fast approximate overlap check using center distances."""
        for i in range(N):
            for j in range(i + 1, N):
                dist = math.sqrt((inner_list[i][0] - inner_list[j][0]) ** 2 + 
                                  (inner_list[i][1] - inner_list[j][1]) ** 2)
                if dist < min_dist:
                    return False
        return True

    # Initialize SA state
    cx_avg = sum(c[0] for c in centers) / len(centers)
    cy_avg = sum(c[1] for c in centers) / len(centers)

    init_params = []
    for x, y, a in initial_inner:
        init_params.extend([x, y, a])
    init_params.extend([cx_avg, cy_avg, 0.0])

    current_params = init_params[:]
    current_obj = sa_objective(current_params)

    best_sa_params = current_params[:]
    best_sa_obj = current_obj
    best_sa_valid = False
    best_sa_inner = None
    best_sa_center = None
    best_sa_angle = None

    # Check initial validity
    valid, inner_l, center_l, angle_l = sa_validate(current_params)
    if valid:
        best_sa_valid = True
        best_sa_inner = inner_l
        best_sa_center = center_l
        best_sa_angle = angle_l
        best_sa_obj = compute_required_side(inner_l, center_l, angle_l)

    # SA parameters
    T0 = 0.8
    T_end = 0.001
    total_steps = 60000
    steps_per_check = 200

    np.random.seed(42)

    for step in range(total_steps):
        if time.time() - start_time > DEADLINE - 30:
            break

        # Cool temperature
        T = T0 * (T_end / T0) ** (step / total_steps)

        # Generate a random perturbation
        new_params = current_params[:]
        move_type = np.random.randint(0, 5)

        if move_type < 3:
            # Perturb one inner hexagon's position
            idx = np.random.randint(0, N)
            base = idx * 3
            new_params[base] += np.random.uniform(-0.12, 0.12)
            new_params[base + 1] += np.random.uniform(-0.12, 0.12)
        elif move_type == 3:
            # Perturb one inner hexagon's rotation
            idx = np.random.randint(0, N)
            new_params[idx * 3 + 2] += np.random.uniform(-8, 8)
        else:
            # Perturb outer center or angle
            if np.random.random() < 0.7:
                new_params[N * 3] += np.random.uniform(-0.12, 0.12)
                new_params[N * 3 + 1] += np.random.uniform(-0.12, 0.12)
            else:
                new_params[N * 3 + 2] += np.random.uniform(-8, 8)

        new_obj = sa_objective(new_params)

        # Accept or reject
        if new_obj < current_obj or np.random.random() < math.exp(-(new_obj - current_obj) / max(T, 1e-10)):
            current_params = new_params
            current_obj = new_obj

            # Track best valid solution - use fast pre-check then full validation
            if step % steps_per_check == 0:
                inner_l = []
                for i in range(N):
                    inner_l.append([current_params[i*3], current_params[i*3+1], current_params[i*3+2]])
                if fast_overlap_check(inner_l):
                    valid, inner_l2, center_l, angle_l = sa_validate(current_params)
                    if valid:
                        sa_side = compute_required_side(inner_l2, center_l, angle_l)
                        if sa_side < best_sa_obj:
                            best_sa_valid = True
                            best_sa_inner = inner_l2
                            best_sa_center = center_l
                            best_sa_angle = angle_l
                            best_sa_obj = sa_side
        else:
            # Rejected: revert
            pass

    # Update global best from SA
    if best_sa_valid and best_sa_obj < best_side:
        best_side = best_sa_obj
        best_inner = best_sa_inner
        best_outer_center = best_sa_center
        best_outer_angle = best_sa_angle

    # Phase 2: Local polish with Nelder-Mead on best SA solution
    # Use the smooth objective for the polish as well, as it is more robust for Nelder-Mead
    if time.time() - start_time < DEADLINE - 25:
        try:
            if best_sa_valid:
                # Flatten for optimizer
                flat = []
                for i, j, a in best_sa_inner:
                    flat.extend([i, j, a])
                flat.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])

                def polish_obj(p):
                    inner_l = []
                    for i in range(N):
                        inner_l.append([p[i*3], p[i*3+1], p[i*3+2]])
                    cx = p[N*3]
                    cy = p[N*3+1]
                    ang = p[N*3+2]
                    # Use smooth side for optimization (now correctly handles |proj|)
                    side = compute_required_side_smooth(inner_l, (cx, cy), ang, temp=0.01)
                    # Penalty for overlaps
                    pen = 0.0
                    min_d = SQRT3 - 0.01
                    for i in range(N):
                        for j in range(i+1, N):
                            d = math.sqrt((inner_l[i][0]-inner_l[j][0])**2 + (inner_l[i][1]-inner_l[j][1])**2)
                            if d < min_d:
                                pen += (min_d - d)**2
                    return side + 500.0 * pen

                polish_res = minimize(
                    polish_obj,
                    flat,
                    method='Nelder-Mead',
                    options={'maxiter': 8000, 'xatol': 1e-13, 'fatol': 1e-15, 'maxfev': 15000}
                )

                inner_l = []
                for i in range(N):
                    inner_l.append([polish_res.x[i*3], polish_res.x[i*3+1], polish_res.x[i*3+2]])
                cx = polish_res.x[N*3]
                cy = polish_res.x[N*3+1]
                ang = polish_res.x[N*3+2]

                # Validate with hard checks
                if check_all_disjoint(inner_l):
                    side = compute_required_side(inner_l, (cx, cy), ang)
                    if check_inside_outer(inner_l, (cx, cy), side, ang):
                        if side < best_side:
                            best_side = side
                            best_inner = inner_l
                            best_outer_center = (cx, cy)
                            best_outer_angle = ang
        except Exception:
            pass

    # Phase 2b: Second polish pass with even tighter tolerances
    if time.time() - start_time < DEADLINE - 15:
        try:
            if best_sa_valid:
                flat2 = []
                for i, j, a in best_inner:
                    flat2.extend([i, j, a])
                flat2.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])

                def polish2_obj(p):
                    inner_l = []
                    for i in range(N):
                        inner_l.append([p[i*3], p[i*3+1], p[i*3+2]])
                    cx = p[N*3]
                    cy = p[N*3+1]
                    ang = p[N*3+2]
                    side = compute_required_side_smooth(inner_l, (cx, cy), ang, temp=0.005)
                    pen = 0.0
                    min_d = SQRT3 - 0.005
                    for i in range(N):
                        for j in range(i+1, N):
                            d = math.sqrt((inner_l[i][0]-inner_l[j][0])**2 + (inner_l[i][1]-inner_l[j][1])**2)
                            if d < min_d:
                                pen += (min_d - d)**2
                    return side + 800.0 * pen

                polish2_res = minimize(
                    polish2_obj,
                    flat2,
                    method='Nelder-Mead',
                    options={'maxiter': 6000, 'xatol': 1e-14, 'fatol': 1e-16, 'maxfev': 12000}
                )

                inner_l2 = []
                for i in range(N):
                    inner_l2.append([polish2_res.x[i*3], polish2_res.x[i*3+1], polish2_res.x[i*3+2]])
                cx2 = polish2_res.x[N*3]
                cy2 = polish2_res.x[N*3+1]
                ang2 = polish2_res.x[N*3+2]

                if check_all_disjoint(inner_l2):
                    side2 = compute_required_side(inner_l2, (cx2, cy2), ang2)
                    if check_inside_outer(inner_l2, (cx2, cy2), side2, ang2):
                        if side2 < best_side:
                            best_side = side2
                            best_inner = inner_l2
                            best_outer_center = (cx2, cy2)
                            best_outer_angle = ang2
        except Exception:
            pass

    # Phase 3: Multi-start with different initial topologies
    if time.time() - start_time < DEADLINE - 20:
        # Try a different initial arrangement: more spread out with some rotated
        # Use a slightly larger spacing and add rotations to break symmetry
        alt_coords = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (1, 1), (0, 2), (-1, 2),
        ]
        alt_angles = [0.0, 30.0, -30.0, 15.0, -15.0, 0.0, 30.0, 0.0, 15.0, -30.0, 10.0]
        alt_scale = 1.0 + 2e-6

        alt_centers = []
        for q, r in alt_coords:
            x = 1.5 * alt_scale * q
            y = SQRT3 * alt_scale * (r + q / 2.0)
            alt_centers.append((x, y))

        alt_inner = [[x, y, alt_angles[i]] for i, (x, y) in enumerate(alt_centers)]

        # Quick SA on this topology
        cx_a = sum(c[0] for c in alt_centers) / len(alt_centers)
        cy_a = sum(c[1] for c in alt_centers) / len(alt_centers)

        init_p2 = []
        for x, y, a in alt_inner:
            init_p2.extend([x, y, a])
        init_p2.extend([cx_a, cy_a, 0.0])

        cur_p2 = init_p2[:]
        cur_obj2 = sa_objective(cur_p2)
        best_p2_params = cur_p2[:]
        best_p2_obj = cur_obj2
        best_p2_valid = False
        best_p2_inner = None
        best_p2_center = None
        best_p2_angle = None

        valid2, il2, cl2, al2 = sa_validate(cur_p2)
        if valid2:
            best_p2_valid = True
            best_p2_inner = il2
            best_p2_center = cl2
            best_p2_angle = al2
            best_p2_obj = compute_required_side(il2, cl2, al2)

        T0_2 = 0.5
        T_end_2 = 0.002
        steps2 = 25000

        np.random.seed(123)

        for step2 in range(steps2):
            if time.time() - start_time > DEADLINE - 15:
                break

            T2 = T0_2 * (T_end_2 / T0_2) ** (step2 / steps2)

            new_p2 = cur_p2[:]
            mt = np.random.randint(0, 4)
            if mt < 2:
                idx = np.random.randint(0, N)
                base = idx * 3
                new_p2[base] += np.random.uniform(-0.1, 0.1)
                new_p2[base + 1] += np.random.uniform(-0.1, 0.1)
            elif mt == 2:
                idx = np.random.randint(0, N)
                new_p2[idx * 3 + 2] += np.random.uniform(-8, 8)
            else:
                new_p2[N * 3] += np.random.uniform(-0.1, 0.1)
                new_p2[N * 3 + 1] += np.random.uniform(-0.1, 0.1)

            new_obj2 = sa_objective(new_p2)

            if new_obj2 < cur_obj2 or np.random.random() < math.exp(-(new_obj2 - cur_obj2) / max(T2, 1e-10)):
                cur_p2 = new_p2
                cur_obj2 = new_obj2

                if step2 % 400 == 0:
                    inner_l2 = []
                    for i in range(N):
                        inner_l2.append([cur_p2[i*3], cur_p2[i*3+1], cur_p2[i*3+2]])
                    if fast_overlap_check(inner_l2):
                        valid2, il2, cl2, al2 = sa_validate(cur_p2)
                        if valid2:
                            s2 = compute_required_side(il2, cl2, al2)
                            if s2 < best_p2_obj:
                                best_p2_valid = True
                                best_p2_inner = il2
                                best_p2_center = cl2
                                best_p2_angle = al2
                                best_p2_obj = s2

        if best_p2_valid and best_p2_obj < best_side:
            best_side = best_p2_obj
            best_inner = best_p2_inner
            best_outer_center = best_p2_center
            best_outer_angle = best_p2_angle

    # Phase 3b: Another multi-start with wider spacing
    if time.time() - start_time < DEADLINE - 12:
        # Wider initial spacing to explore different configurations
        wide_coords = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (1, 1), (0, 2), (-1, 2),
        ]
        wide_scale = 1.0 + 5e-6
        wide_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        wide_centers = []
        for q, r in wide_coords:
            x = 1.5 * wide_scale * q
            y = SQRT3 * wide_scale * (r + q / 2.0)
            wide_centers.append((x, y))

        wide_inner = [[x, y, wide_angles[i]] for i, (x, y) in enumerate(wide_centers)]

        cx_w = sum(c[0] for c in wide_centers) / len(wide_centers)
        cy_w = sum(c[1] for c in wide_centers) / len(wide_centers)

        init_p3 = []
        for x, y, a in wide_inner:
            init_p3.extend([x, y, a])
        init_p3.extend([cx_w, cy_w, 0.0])

        cur_p3 = init_p3[:]
        cur_obj3 = sa_objective(cur_p3)
        best_p3_obj = cur_obj3
        best_p3_valid = False
        best_p3_inner = None
        best_p3_center = None
        best_p3_angle = None

        valid3, il3, cl3, al3 = sa_validate(cur_p3)
        if valid3:
            best_p3_valid = True
            best_p3_inner = il3
            best_p3_center = cl3
            best_p3_angle = al3
            best_p3_obj = compute_required_side(il3, cl3, al3)

        T0_3 = 0.3
        T_end_3 = 0.002
        steps3 = 15000

        np.random.seed(456)

        for step3 in range(steps3):
            if time.time() - start_time > DEADLINE - 10:
                break

            T3 = T0_3 * (T_end_3 / T0_3) ** (step3 / steps3)

            new_p3 = cur_p3[:]
            mt3 = np.random.randint(0, 4)
            if mt3 < 2:
                idx = np.random.randint(0, N)
                base = idx * 3
                new_p3[base] += np.random.uniform(-0.08, 0.08)
                new_p3[base + 1] += np.random.uniform(-0.08, 0.08)
            elif mt3 == 2:
                idx = np.random.randint(0, N)
                new_p3[idx * 3 + 2] += np.random.uniform(-6, 6)
            else:
                new_p3[N * 3] += np.random.uniform(-0.08, 0.08)
                new_p3[N * 3 + 1] += np.random.uniform(-0.08, 0.08)

            new_obj3 = sa_objective(new_p3)

            if new_obj3 < cur_obj3 or np.random.random() < math.exp(-(new_obj3 - cur_obj3) / max(T3, 1e-10)):
                cur_p3 = new_p3
                cur_obj3 = new_obj3

                if step3 % 500 == 0:
                    inner_l3 = []
                    for i in range(N):
                        inner_l3.append([cur_p3[i*3], cur_p3[i*3+1], cur_p3[i*3+2]])
                    if fast_overlap_check(inner_l3):
                        valid3, il3, cl3, al3 = sa_validate(cur_p3)
                        if valid3:
                            s3 = compute_required_side(il3, cl3, al3)
                            if s3 < best_p3_obj:
                                best_p3_valid = True
                                best_p3_inner = il3
                                best_p3_center = cl3
                                best_p3_angle = al3
                                best_p3_obj = s3

        if best_p3_valid and best_p3_obj < best_side:
            best_side = best_p3_obj
            best_inner = best_p3_inner
            best_outer_center = best_p3_center
            best_outer_angle = best_p3_angle

    # Phase 4: Final fine polish
    if time.time() - start_time < DEADLINE - 3:
        try:
            flat_final = []
            for i, j, a in best_inner:
                flat_final.extend([i, j, a])
            flat_final.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])

            def final_obj(p):
                inner_l = []
                for i in range(N):
                    inner_l.append([p[i*3], p[i*3+1], p[i*3+2]])
                cx = p[N*3]
                cy = p[N*3+1]
                ang = p[N*3+2]
                # Use very low temperature for near-exact max (now correctly handles |proj|)
                side = compute_required_side_smooth(inner_l, (cx, cy), ang, temp=0.001)
                pen = 0.0
                min_d = SQRT3 - 0.003
                for i in range(N):
                    for j in range(i+1, N):
                        d = math.sqrt((inner_l[i][0]-inner_l[j][0])**2 + (inner_l[i][1]-inner_l[j][1])**2)
                        if d < min_d:
                            pen += (min_d - d)**2
                return side + 1000.0 * pen

            final_res = minimize(
                final_obj,
                flat_final,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-15, 'fatol': 1e-17, 'maxfev': 20000}
            )

            inner_l = []
            for i in range(N):
                inner_l.append([final_res.x[i*3], final_res.x[i*3+1], final_res.x[i*3+2]])
            cx = final_res.x[N*3]
            cy = final_res.x[N*3+1]
            ang = final_res.x[N*3+2]

            # Validate with hard checks
            if check_all_disjoint(inner_l):
                side = compute_required_side(inner_l, (cx, cy), ang)
                if check_inside_outer(inner_l, (cx, cy), side, ang):
                    if side < best_side:
                        best_side = side
                        best_inner = inner_l
                        best_outer_center = (cx, cy)
                        best_outer_angle = ang
        except Exception:
            pass

    # Final verification and fallback
    final_inner = [list(x) for x in best_inner]
    final_center = best_outer_center
    final_side = best_side
    final_angle = best_outer_angle

    if not check_all_disjoint(final_inner) or not check_inside_outer(final_inner, final_center, final_side, final_angle):
        scale_fallback = 1.05
        fallback_centers = []
        for q, r in axial_coords:
            x = 1.5 * scale_fallback * q
            y = SQRT3 * scale_fallback * (r + q / 2.0)
            fallback_centers.append((x, y))

        final_inner = [[x, y, 0.0] for x, y in fallback_centers]
        final_center = (0.0, 0.0)
        final_angle = 0.0
        final_side = compute_required_side(final_inner, final_center, final_angle)

    return final_inner, final_center, final_side, final_angle

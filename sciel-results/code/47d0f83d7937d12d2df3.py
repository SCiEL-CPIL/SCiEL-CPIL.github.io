"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

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

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=80.0):
    """
    Smooth approximation using log-sum-exp for the max projection.
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
    """Check if two convex polygons overlap using SAT."""
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
    for i in range(n):
        vi = get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
        for j in range(i + 1, n):
            vj = get_hex_vertices(inner[j][0], inner[j][1], 1.0, math.radians(inner[j][2]))
            if sat_overlap_check(vi, vj, eps):
                return False
    return True

def compute_overlap_penalty(inner, min_dist=SQRT3 - 0.005):
    """Smooth penalty for overlaps based on center distances."""
    penalty = 0.0
    n = len(inner)
    for i in range(n):
        xi, yi, _ = inner[i]
        for j in range(i + 1, n):
            xj, yj, _ = inner[j]
            dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dist < min_dist:
                diff = min_dist - dist
                penalty += diff * diff
    return penalty

def build_contact_graph(inner):
    """
    Build a contact graph: for each pair of hexagons, record the closest
    center-to-center distance (approximating face-to-face adjacency).
    Returns list of (i, j, dist).
    """
    contacts = []
    n = len(inner)
    for i in range(n):
        xi, yi, _ = inner[i]
        for j in range(i + 1, n):
            xj, yj, _ = inner[j]
            d = math.sqrt((xi - xj)**2 + (yi - yj)**2)
            contacts.append((i, j, d))
    return contacts

def compute_rigid_body_side(inner, global_offset, global_angle_deg):
    """
    Rigid-body formulation: treat all inner hexagons as a single composite
    object. global_offset is the translation of the composite, global_angle_deg
    is the rotation applied to the whole composite (in addition to each
    hexagon's own angle). The container is then placed at origin with the
    given rotation.

    inner: list of (x, y, angle_deg) -- local positions of the composite
    global_offset: (tx, ty) -- translation of the composite
    global_angle_deg: rotation of the composite

    Returns the minimum outer side length with container centered at origin
    and rotated by global_angle_deg.
    """
    if not inner:
        return 0.0

    tx, ty = global_offset
    ga = math.radians(global_angle_deg)
    cos_g = math.cos(ga)
    sin_g = math.sin(ga)

    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))

    max_projection = 0.0

    for (ix, iy, iangle) in inner:
        # Rotate the local position by global angle, then translate
        rx = cos_g * ix - sin_g * iy
        ry = sin_g * ix + cos_g * iy
        cx = rx + tx
        cy = ry + ty
        # The hexagon's own angle is also rotated by global angle
        total_angle = math.radians(iangle) + ga
        verts = get_hex_vertices(cx, cy, 1.0, total_angle)
        for v in verts:
            for nx, ny in normals:
                proj = v[0] * nx + v[1] * ny
                if proj > max_projection:
                    max_projection = proj

    return 2.0 * max_projection / SQRT3

def compute_rigid_body_side_smooth(inner, global_offset, global_angle_deg, beta=80.0):
    """Smooth version of compute_rigid_body_side using log-sum-exp with centering."""
    if not inner:
        return 0.0

    tx, ty = global_offset
    ga = math.radians(global_angle_deg)
    cos_g = math.cos(ga)
    sin_g = math.sin(ga)

    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))

    projections = []
    for (ix, iy, iangle) in inner:
        rx = cos_g * ix - sin_g * iy
        ry = sin_g * ix + cos_g * iy
        cx = rx + tx
        cy = ry + ty
        total_angle = math.radians(iangle) + ga
        verts = get_hex_vertices(cx, cy, 1.0, total_angle)
        for v in verts:
            for nx, ny in normals:
                proj = v[0] * nx + v[1] * ny
                projections.append(proj)

    if not projections:
        return 0.0

    max_proj = max(projections)
    if max_proj < -1000:
        return 0.0

    # Use a more stable log-sum-exp computation
    scaled = [beta * (p - max_proj) for p in projections]
    # Clip to avoid overflow
    scaled = [max(s, -100) for s in scaled]
    sum_exp = sum(math.exp(s) for s in scaled)
    smooth_max = max_proj + math.log(sum_exp) / beta

    return 2.0 * smooth_max / SQRT3

def compute_contact_penalty(inner, min_dist=SQRT3 - 0.002):
    """
    Differentiable penalty that enforces face-to-face adjacency:
    for each pair, penalize if the center distance is less than min_dist.
    This is the differentiable contact constraint.
    """
    penalty = 0.0
    n = len(inner)
    for i in range(n):
        xi, yi, _ = inner[i]
        for j in range(i + 1, n):
            xj, yj, _ = inner[j]
            d = math.sqrt((xi - xj)**2 + (yi - yj)**2)
            if d < min_dist:
                diff = min_dist - d
                penalty += diff * diff
    return penalty

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Hexagonal lattice: center + 6 neighbors + 4 more for compact cluster
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, 1), (-2, 1), (-1, -1)
    ]

    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]

    epsilon = 1e-6
    scale = 1.0 + epsilon

    centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        centers.append((x, y))

    initial_inner = [[x, y, 0.0] for x, y in centers]

    cx_avg = sum(c[0] for c in centers) / len(centers)
    cy_avg = sum(c[1] for c in centers) / len(centers)

    best_side = float('inf')
    best_inner = initial_inner
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0

    # Phase 1: Rigid-body global orientation + container optimization
    # The composite object (11 hexagons) is treated as a single body.
    # We optimize: (tx, ty, global_angle) to minimize the container side.
    # The container is centered at origin with rotation = global_angle.

    def objective_rigid_smooth(params):
        tx, ty, ga = params
        return compute_rigid_body_side_smooth(initial_inner, (tx, ty), ga, beta=50.0)

    def objective_rigid_exact(params):
        tx, ty, ga = params
        return compute_rigid_body_side(initial_inner, (tx, ty), ga)

    best_rigid_result = None
    best_rigid_val = float('inf')

    # More refined initial angles for better coverage - focus on 0-60 due to hexagonal symmetry
    init_angles = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

    for init_angle in init_angles:
        if time.time() > deadline - 55:
            break

        # Multiple starting offsets for better basin exploration
        for offset_x in [-0.1, 0.0, 0.1]:
            for offset_y in [-0.1, 0.0, 0.1]:
                if time.time() > deadline - 55:
                    break

                init_params = [offset_x, offset_y, float(init_angle)]

                try:
                    # Smooth pass for basin finding
                    res_smooth = minimize(objective_rigid_smooth, init_params, method='Nelder-Mead',
                                          options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8, 'maxfev': 600})

                    # Exact refinement with tighter tolerances
                    res_exact = minimize(objective_rigid_exact, res_smooth.x, method='Nelder-Mead',
                                         options={'maxiter': 600, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 1200})

                    if res_exact.fun < best_rigid_val:
                        best_rigid_val = res_exact.fun
                        best_rigid_result = res_exact.x.copy()
                except Exception:
                    pass

    if best_rigid_result is not None:
        best_tx = best_rigid_result[0]
        best_ty = best_rigid_result[1]
        best_ga = best_rigid_result[2]
        best_side = best_rigid_val
        best_outer_angle = best_ga
        best_outer_center = (0.0, 0.0)
    else:
        best_tx, best_ty, best_ga = 0.0, 0.0, 0.0
        best_side = compute_required_side(initial_inner, (cx_avg, cy_avg), 0.0)
        best_outer_center = (cx_avg, cy_avg)
        best_outer_angle = 0.0

    # Phase 2: Joint optimization with contact-graph constraints
    # Now we allow the internal positions to vary slightly while enforcing
    # the contact graph (face-to-face adjacency) via a differentiable penalty.

    def objective_joint(params):
        inner_list = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            a = params[i*3+2]
            inner_list.append([x, y, a])

        tx = params[N*3]
        ty = params[N*3+1]
        ga = params[N*3+2]

        side = compute_rigid_body_side(inner_list, (tx, ty), ga)
        # Contact penalty: enforce minimum distance between centers
        penalty = compute_contact_penalty(inner_list, min_dist=SQRT3 - 0.002)

        return side + 500.0 * penalty

    initial_params = []
    for x, y, a in initial_inner:
        initial_params.extend([x, y, a])
    initial_params.extend([best_tx, best_ty, best_ga])

    current_params = np.array(initial_params)
    current_val = objective_joint(current_params)
    best_params = current_params.copy()
    best_val_joint = current_val

    # Simulated annealing with adaptive step size
    T = 0.4
    T_min = 0.0005
    cooling = 0.9997

    max_iter = 18000
    for i in range(max_iter):
        if time.time() > deadline - 30:
            break

        neighbor = current_params.copy()
        # Adaptive step scale based on temperature
        step_scale = 0.25 * (T / T_min) ** 0.25

        if np.random.random() < 0.7:
            idx = np.random.randint(0, N)
            dim = np.random.randint(0, 3)
            if dim < 2:
                delta = np.random.normal(0, 0.12 * step_scale)
            else:
                delta = np.random.normal(0, 2.0 * step_scale)
            neighbor[idx*3 + dim] += delta
        elif np.random.random() < 0.5:
            # Rigid body move of two hexagons
            idx1 = np.random.randint(0, N)
            idx2 = np.random.randint(0, N)
            if idx1 != idx2:
                dx = np.random.normal(0, 0.10 * step_scale)
                dy = np.random.normal(0, 0.10 * step_scale)
                neighbor[idx1*3] += dx
                neighbor[idx1*3+1] += dy
                neighbor[idx2*3] += dx
                neighbor[idx2*3+1] += dy
        else:
            dim = np.random.randint(N*3, N*3+3)
            if dim == N*3+2:
                delta = np.random.normal(0, 2.0 * step_scale)
            else:
                delta = np.random.normal(0, 0.10 * step_scale)
            neighbor[dim] += delta

        neighbor_val = objective_joint(neighbor)

        if neighbor_val < current_val or np.random.random() < math.exp(-(neighbor_val - current_val) / T):
            current_params = neighbor
            current_val = neighbor_val

            if current_val < best_val_joint:
                best_params = current_params.copy()
                best_val_joint = current_val

        T *= cooling
        if T < T_min:
            break

    best_inner_joint = []
    for i in range(N):
        x = best_params[i*3]
        y = best_params[i*3+1]
        a = best_params[i*3+2]
        best_inner_joint.append([x, y, a])

    best_tx_j = best_params[N*3]
    best_ty_j = best_params[N*3+1]
    best_ga_j = best_params[N*3+2]

    # Convert rigid-body solution to standard format
    # The rigid-body formulation places the container at origin with rotation ga.
    # The inner hexagons are at (tx, ty) + rotated positions.
    # We need to convert to the standard format: inner hexagons at absolute positions,
    # container at some center with some angle.

    # In the rigid-body formulation, the container is at origin with angle ga.
    # The inner hexagons are at positions (tx, ty) + R(ga) * local_pos.
    # So in the standard format:
    #   outer_center = (0, 0), outer_angle = ga
    #   inner[i] = (tx + R(ga)*local_x, ty + R(ga)*local_y, local_angle + ga)

    ga_rad = math.radians(best_ga_j)
    cos_ga = math.cos(ga_rad)
    sin_ga = math.sin(ga_rad)

    standard_inner = []
    for i in range(N):
        lx, ly, la = best_inner_joint[i]
        rx = cos_ga * lx - sin_ga * ly
        ry = sin_ga * lx + cos_ga * ly
        ax = rx + best_tx_j
        ay = ry + best_ty_j
        aa = la + best_ga_j
        standard_inner.append([ax, ay, aa])

    if check_all_disjoint(standard_inner, eps=1e-5):
        best_inner = standard_inner
        best_outer_center = (0.0, 0.0)
        best_outer_angle = best_ga_j
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)
    else:
        # Fall back to the rigid-body solution from Phase 1
        ga_rad = math.radians(best_ga)
        cos_ga = math.cos(ga_rad)
        sin_ga = math.sin(ga_rad)
        standard_inner = []
        for i in range(N):
            lx, ly, la = initial_inner[i]
            rx = cos_ga * lx - sin_ga * ly
            ry = sin_ga * lx + cos_ga * ly
            ax = rx + best_tx
            ay = ry + best_ty
            aa = la + best_ga
            standard_inner.append([ax, ay, aa])
        best_inner = standard_inner
        best_outer_center = (0.0, 0.0)
        best_outer_angle = best_ga
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)

    # Phase 3: Final high-precision polish on outer container with more starts
    def objective_outer_polish(params):
        cx, cy, angle_deg = params
        return compute_required_side(best_inner, (cx, cy), angle_deg)

    # Generate more diverse starting points for the polish - focus on small perturbations
    polish_starts = []
    base_angle = best_outer_angle
    base_cx = best_outer_center[0]
    base_cy = best_outer_center[1]

    # Angle perturbations (hexagonal symmetry means 60° period)
    for da in [0, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.5, -0.5, 1.0, -1.0]:
        polish_starts.append([base_cx, base_cy, base_angle + da])

    # Center perturbations
    for dx in [-0.02, -0.01, 0, 0.01, 0.02]:
        for dy in [-0.02, -0.01, 0, 0.01, 0.02]:
            if dx == 0 and dy == 0:
                continue
            polish_starts.append([base_cx + dx, base_cy + dy, base_angle])

    best_polish_val = best_side
    best_polish_params = [best_outer_center[0], best_outer_center[1], best_outer_angle]

    for ps in polish_starts:
        if time.time() > deadline - 3:
            break
        try:
            # Use very tight tolerances for high-precision polish
            res = minimize(objective_outer_polish, ps, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-15, 'fatol': 1e-17, 'maxfev': 10000})
            if res.fun < best_polish_val:
                best_polish_val = res.fun
                best_polish_params = res.x.copy()
        except Exception:
            pass

    if best_polish_params is not None:
        best_outer_center = (best_polish_params[0], best_polish_params[1])
        best_outer_angle = best_polish_params[2]
        best_side = best_polish_val

    # Final validation: ensure the solution is valid
    if not check_all_disjoint(best_inner, eps=1e-6):
        # Fall back to the initial lattice if there's overlap
        initial_inner_valid = []
        for x, y, a in initial_inner:
            initial_inner_valid.append([x, y, a])
        best_inner = initial_inner_valid
        best_outer_center = (cx_avg, cy_avg)
        best_outer_angle = 0.0
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)

    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END

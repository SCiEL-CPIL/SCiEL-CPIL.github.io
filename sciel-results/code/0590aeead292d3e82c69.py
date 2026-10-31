"""Packing unit hexagons inside a regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)
DEADLINE = 135.0  # safety margin before 150s kill


def hex_vertex_positions(cx, cy, angle_rad):
    """Return 6 vertex positions as a (6,2) numpy array for a unit hexagon."""
    thetas = angle_rad + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(thetas), cy + np.sin(thetas)])


def min_dist_between_hexagons(v1, v2):
    """
    Compute minimum distance between two convex hexagons (6x2 arrays).
    Returns 0 if they overlap or touch, positive distance otherwise.
    Uses SAT: if any separating axis found, return the gap; else return 0.
    """
    # Edge normals from both hexagons
    axes = []
    for v in (v1, v2):
        for i in range(6):
            p1 = v[i]
            p2 = v[(i + 1) % 6]
            edge = p2 - p1
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                axes.append(n / norm)
    if not axes:
        return 0.0

    max_gap = 0.0
    for axis in axes:
        proj1 = v1 @ axis
        proj2 = v2 @ axis
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        gap = min(max1 - min2, max2 - min1)
        if gap < -1e-9:
            return 0.0  # Overlapping on this axis
        max_gap = max(max_gap, gap)
    return max_gap


def force_directed_compact(initial_centers, initial_angles, time_budget=25.0):
    """
    Force-directed granular packing: treat hexagons as particles with
    repulsive forces and a soft confining potential.
    
    Returns list of [x, y, angle_deg] for N hexagons.
    """
    n = len(initial_centers)
    positions = np.array([[c[0], c[1]] for c in initial_centers], dtype=float)
    angles = np.array(initial_angles, dtype=float)  # radians

    # Precompute vertex arrays
    def get_verts(i):
        return hex_vertex_positions(positions[i, 0], positions[i, 1], angles[i])

    # Simulation parameters
    repulse_strength = 50.0
    repulse_radius = 3.5
    centering_strength = 0.1
    dt = 0.005
    damping = 0.92
    max_steps = int(time_budget / dt)

    # Track best configuration by extent
    def compute_extent():
        all_pts = []
        for i in range(n):
            all_pts.append(get_verts(i))
        all_pts = np.vstack(all_pts)
        cx = all_pts[:, 0].mean()
        cy = all_pts[:, 1].mean()
        return np.max(np.sqrt((all_pts[:, 0] - cx)**2 + (all_pts[:, 1] - cy)**2))

    best_extent = compute_extent()
    best_positions = positions.copy()
    best_angles = angles.copy()

    start_sim = time.time()
    step = 0
    while step < max_steps and (time.time() - start_sim) < time_budget:
        # Compute forces
        forces = np.zeros_like(positions)
        torques = np.zeros(n)

        # Pairwise repulsion
        for i in range(n):
            vi = get_verts(i)
            for j in range(i + 1, n):
                vj = get_verts(j)
                # Quick rejection
                d = positions[j] - positions[i]
                dist = np.linalg.norm(d)
                if dist > repulse_radius:
                    continue
                gap = min_dist_between_hexagons(vi, vj)
                if gap < 0.01:
                    # Strong repulsion
                    force_mag = repulse_strength * max(0, 0.01 - gap) ** 2 / max(dist, 0.1)
                    if dist > 1e-6:
                        f = force_mag * d / dist
                    else:
                        f = np.array([force_mag * 0.01, 0.0])
                    forces[i] -= f
                    forces[j] += f
                    # Torque: align to reduce overlap (simple heuristic)
                    # Rotate j slightly to reduce overlap
                    if gap < 0.005:
                        torques[j] += 0.02
                        torques[i] -= 0.02

        # Centering force (soft boundary)
        centroid = positions.mean(axis=0)
        for i in range(n):
            r = positions[i] - centroid
            forces[i] -= centering_strength * r

        # Update positions with velocity damping (implicit velocity)
        # Use simple Euler with damping
        vel = forces * dt
        vel *= damping
        positions += vel
        angles += torques * dt

        # Check improvement
        if step % 50 == 0:
            ext = compute_extent()
            if ext < best_extent:
                best_extent = ext
                best_positions = positions.copy()
                best_angles = angles.copy()

        step += 1

    # Convert back to degrees
    result = []
    for i in range(n):
        result.append([best_positions[i, 0], best_positions[i, 1], math.degrees(best_angles[i])])
    return result, best_extent


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


def compute_cluster_extent(inner):
    """
    Compute the extent of the inner hexagon cluster.
    Returns the maximum distance from centroid to any vertex.
    """
    if not inner:
        return 0.0

    all_verts = []
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        all_verts.extend(verts)

    if not all_verts:
        return 0.0

    cx = sum(v[0] for v in all_verts) / len(all_verts)
    cy = sum(v[1] for v in all_verts) / len(all_verts)

    max_dist = 0.0
    for v in all_verts:
        d = math.sqrt((v[0] - cx) ** 2 + (v[1] - cy) ** 2)
        if d > max_dist:
            max_dist = d

    return max_dist


def construct():
    start_time = time.time()

    # 1. Define 11 hexagons on a hexagonal lattice - tighter configuration
    # Use a more compact hexagonal arrangement
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-1, 2), (-2, 1)
    ]

    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]

    # 2. Convert axial to Cartesian with tight spacing
    epsilon = 1e-4  # Slightly larger epsilon for numerical stability
    scale = 1.0 + epsilon

    centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        centers.append((x, y))

    # Initial layout: All hexagons aligned (angle 0)
    initial_inner = [[x, y, 0.0] for x, y in centers]

    best_side = float('inf')
    best_inner = [list(x) for x in initial_inner]
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0

    # 3. Force-directed granular packing compaction
    # Use physics simulation to find dense equilibrium
    initial_centers_list = [(x, y) for x, y, a in initial_inner]
    initial_angles_rad = [math.radians(a) for x, y, a in initial_inner]

    best_compact_extent = compute_cluster_extent(initial_inner)
    best_compact_inner = [list(x) for x in initial_inner]

    # Run force-directed simulation with different initial conditions
    num_fd_restarts = 5
    for restart in range(num_fd_restarts):
        if time.time() - start_time > DEADLINE - 65:
            break

        # Perturb initial conditions for diversity
        if restart == 0:
            fc = initial_centers_list
            fa = initial_angles_rad
        else:
            np.random.seed(123 + restart * 17)
            fc = []
            fa = []
            for i in range(N):
                cx, cy = initial_centers_list[i]
                ca = initial_angles_rad[i]
                pc = np.random.uniform(-0.05, 0.05, size=2)
                pa = np.random.uniform(-15, 15) * math.pi / 180
                fc.append((cx + pc[0], cy + pc[1]))
                fa.append(ca + pa)

        # Allocate time budget for this restart
        remaining = DEADLINE - 65 - (time.time() - start_time)
        time_per_restart = max(3.0, min(15.0, remaining / (num_fd_restarts - restart)))

        try:
            fd_result, fd_extent = force_directed_compact(fc, fa, time_budget=time_per_restart)
            # Verify no overlaps
            if check_all_disjoint(fd_result):
                if fd_extent < best_compact_extent:
                    best_compact_extent = fd_extent
                    best_compact_inner = fd_result
        except Exception:
            pass

    # Also try a quick Nelder-Mead polish on the best FD result
    if time.time() - start_time < DEADLINE - 55:
        nm_params = []
        for x, y, a in best_compact_inner:
            nm_params.extend([x, y, a])

        def nm_compact_obj(params):
            il = [[params[i*3], params[i*3+1], params[i*3+2]] for i in range(N)]
            ext = compute_cluster_extent(il)
            pen = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    d = math.sqrt((il[i][0]-il[j][0])**2 + (il[i][1]-il[j][1])**2)
                    if d < SQRT3 - 0.01:
                        pen += (SQRT3 - 0.01 - d)**2 * 50.0
            return ext + pen

        try:
            res = minimize(nm_compact_obj, nm_params, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 600})
            il = [[res.x[i*3], res.x[i*3+1], res.x[i*3+2]] for i in range(N)]
            if check_all_disjoint(il):
                ext = compute_cluster_extent(il)
                if ext < best_compact_extent:
                    best_compact_extent = ext
                    best_compact_inner = il
        except Exception:
            pass

    # Use compacted layout as starting point for outer optimization
    compact_inner = best_compact_inner

    # 4. Optimize Outer Hexagon using scipy with multiple starts
    # Variables: [cx, cy, angle_deg]

    def objective_outer(params, inner_list):
        cx, cy, angle_deg = params
        side = compute_required_side(inner_list, (cx, cy), angle_deg)
        return side

    # Initial guess: centroid of compacted layout and 0
    compact_centers = [(c[0], c[1]) for c in compact_inner]
    cx_avg = sum(c[0] for c in compact_centers) / len(compact_centers)
    cy_avg = sum(c[1] for c in compact_centers) / len(compact_centers)

    # Try multiple diverse initial angles to avoid local minima
    best_result = None
    best_val = float('inf')

    # Reduced angle sampling to save time for joint optimization
    init_angles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]

    for init_angle in init_angles:
        if time.time() - start_time > DEADLINE - 50:
            break

        init_params = [cx_avg, cy_avg, float(init_angle)]

        try:
            res = minimize(
                lambda p: objective_outer(p, compact_inner),
                init_params,
                method='Nelder-Mead',
                options={'maxiter': 250, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 400}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_result = res.x
        except Exception:
            pass

    if best_result is not None:
        best_outer_center = (best_result[0], best_result[1])
        best_outer_angle = best_result[2]
        best_side = best_val
    else:
        # Fallback
        best_outer_center = (cx_avg, cy_avg)
        best_outer_angle = 0.0
        best_side = compute_required_side(compact_inner, best_outer_center, best_outer_angle)

    # 5. Joint optimization of inner positions and outer parameters
    # Variables: [x1,y1,a1, x2,y2,a2, ..., xN,yN,aN, cx, cy, angle]
    # Total: 3*N + 3 = 36 variables

    def objective_joint(params):
        inner_list = []
        for i in range(N):
            x = params[i * 3]
            y = params[i * 3 + 1]
            a = params[i * 3 + 2]
            inner_list.append([x, y, a])

        cx = params[N * 3]
        cy = params[N * 3 + 1]
        angle = params[N * 3 + 2]

        side = compute_required_side(inner_list, (cx, cy), angle)

        # Penalty for overlaps: use center distance as proxy
        penalty = 0.0
        min_dist = SQRT3 - 0.01  # Safe margin
        for i in range(N):
            for j in range(i + 1, N):
                xi, yi, _ = inner_list[i]
                xj, yj, _ = inner_list[j]
                dist = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2 * 100.0

        # Penalty for going outside (soft constraint)
        ox, oy = cx, cy
        cos_a = math.cos(math.radians(angle))
        sin_a = math.sin(math.radians(angle))
        apothem = side * SQRT3 / 2.0
        for (ix, iy, iangle) in inner_list:
            verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
            for v in verts:
                dx = v[0] - ox
                dy = v[1] - oy
                lx = dx * cos_a + dy * sin_a
                ly = -dx * sin_a + dy * cos_a
                for k in range(6):
                    theta = math.radians(30 + k * 60)
                    proj = lx * math.cos(theta) + ly * math.sin(theta)
                    if proj > apothem:
                        penalty += (proj - apothem) ** 2 * 50.0

        return side + penalty

    # Initial joint params - use compacted layout
    init_params = []
    for x, y, a in compact_inner:
        init_params.extend([x, y, a])
    init_params.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])

    # Try joint optimization with more restarts
    best_joint_side = best_side
    best_joint_inner = [list(x) for x in compact_inner]
    best_joint_center = best_outer_center
    best_joint_angle = best_outer_angle

    # More restarts since we saved time in outer optimization
    num_restarts = 8
    for restart in range(num_restarts):
        if time.time() - start_time > DEADLINE - 15:
            break

        current_params = list(init_params)
        if restart > 0:
            # Add random perturbation with increasing magnitude
            np.random.seed(42 + restart * 7)
            perturb_scale = 0.02 + restart * 0.008
            for i in range(len(current_params)):
                if i < N * 3:
                    current_params[i] += np.random.uniform(-perturb_scale, perturb_scale)
                elif i == N * 3 + 2:  # angle
                    current_params[i] += np.random.uniform(-5, 5)

        try:
            res = minimize(
                objective_joint,
                current_params,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9, 'maxfev': 800}
            )

            # Extract result
            inner_list = []
            for i in range(N):
                inner_list.append([res.x[i * 3], res.x[i * 3 + 1], res.x[i * 3 + 2]])
            oc = (res.x[N * 3], res.x[N * 3 + 1])
            oa = res.x[N * 3 + 2]

            side = compute_required_side(inner_list, oc, oa)

            # Check validity
            if check_all_disjoint(inner_list) and check_inside_outer(inner_list, oc, side, oa):
                if side < best_joint_side:
                    best_joint_side = side
                    best_joint_inner = inner_list
                    best_joint_center = oc
                    best_joint_angle = oa
        except Exception:
            pass

    # 6. Final high-precision polish pass
    if time.time() - start_time < DEADLINE - 5:
        # Refine the best solution found with tighter tolerances
        polish_params = []
        for x, y, a in best_joint_inner:
            polish_params.extend([x, y, a])
        polish_params.extend([best_joint_center[0], best_joint_center[1], best_joint_angle])

        try:
            res = minimize(
                objective_joint,
                polish_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-11, 'fatol': 1e-13, 'maxfev': 3000}
            )

            inner_list = []
            for i in range(N):
                inner_list.append([res.x[i * 3], res.x[i * 3 + 1], res.x[i * 3 + 2]])
            oc = (res.x[N * 3], res.x[N * 3 + 1])
            oa = res.x[N * 3 + 2]

            side = compute_required_side(inner_list, oc, oa)

            if check_all_disjoint(inner_list) and check_inside_outer(inner_list, oc, side, oa):
                if side < best_joint_side:
                    best_joint_side = side
                    best_joint_inner = inner_list
                    best_joint_center = oc
                    best_joint_angle = oa
        except Exception:
            pass

    # 7. If joint optimization didn't find valid, try local search on initial layout
    if best_joint_side >= best_side:
        # Use the outer-optimized result with original inner positions
        best_joint_inner = [list(x) for x in compact_inner]
        best_joint_side = best_side
        best_joint_center = best_outer_center
        best_joint_angle = best_outer_angle

    # 8. Final verification and fallback
    final_inner = best_joint_inner
    final_center = best_joint_center
    final_side = best_joint_side
    final_angle = best_joint_angle

    if not check_all_disjoint(final_inner) or not check_inside_outer(final_inner, final_center, final_side, final_angle):
        # Fallback: use original layout with safe spacing
        scale_fallback = 1.02
        fallback_centers = []
        for q, r in axial_coords:
            x = 1.5 * scale_fallback * q
            y = SQRT3 * scale_fallback * (r + q / 2.0)
            fallback_centers.append((x, y))

        fallback_inner = [[x, y, 0.0] for x, y in fallback_centers]
        final_center = (0.0, 0.0)
        final_angle = 0.0
        final_side = compute_required_side(fallback_inner, final_center, final_angle)
        final_inner = fallback_inner

    return final_inner, final_center, final_side, final_angle

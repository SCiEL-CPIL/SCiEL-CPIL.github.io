import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 145.0
    random.seed(7)

    def get_hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        thetas = angle + np.arange(6) * (math.pi / 3)
        xs = cx + side * np.cos(thetas)
        ys = cy + side * np.sin(thetas)
        return np.column_stack((xs, ys))

    def are_disjoint_sat(v1, v2, eps=1e-9):
        e1 = np.roll(v1, -1, axis=0) - v1
        n1 = np.column_stack((-e1[:, 1], e1[:, 0]))
        p1 = v1 @ n1.T
        p2 = v2 @ n1.T
        if np.any(np.max(p1, axis=0) < np.min(p2, axis=0) + eps) or np.any(np.max(p2, axis=0) < np.min(p1, axis=0) + eps):
            return True
        e2 = np.roll(v2, -1, axis=0) - v2
        n2 = np.column_stack((-e2[:, 1], e2[:, 0]))
        p1 = v1 @ n2.T
        p2 = v2 @ n2.T
        if np.any(np.max(p1, axis=0) < np.min(p2, axis=0) + eps) or np.any(np.max(p2, axis=0) < np.min(p1, axis=0) + eps):
            return True
        return False

    def sat_gap(poly1, poly2):
        """
        Returns the minimum separation distance between two convex hexagons.
        If they overlap, returns a negative value (penetration depth).
        If they are separated, returns the positive distance.
        """
        axes1 = []
        for i in range(6):
            edge = poly1[(i+1)%6] - poly1[i]
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-9:
                axes1.append(n / norm)
        
        axes2 = []
        for i in range(6):
            edge = poly2[(i+1)%6] - poly2[i]
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-9:
                axes2.append(n / norm)
        
        min_gap = float('inf')
        
        for axis in axes1:
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            gap = max(np.min(proj1) - np.max(proj2), np.min(proj2) - np.max(proj1))
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap
        
        for axis in axes2:
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            gap = max(np.min(proj1) - np.max(proj2), np.min(proj2) - np.max(proj1))
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap
        
        return min_gap

    def compute_outer_side(centers, angles, ocx, ocy, otheta):
        max_proj = 0.0
        otheta_rad = math.radians(otheta)
        normal_angles = otheta_rad + math.pi / 6 + np.arange(6) * (math.pi / 3)
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        for i in range(len(centers)):
            cx, cy = centers[i]
            angle_rad = math.radians(angles[i])
            thetas = angle_rad + np.arange(6) * (math.pi / 3)
            vx = cx + np.cos(thetas)
            vy = cy + np.sin(thetas)
            dx = vx - ocx
            dy = vy - ocy
            projs = dx[:, np.newaxis] * nx[np.newaxis, :] + dy[:, np.newaxis] * ny[np.newaxis, :]
            current_max = np.max(projs)
            if current_max > max_proj:
                max_proj = current_max
        return 2.0 * max_proj / math.sqrt(3)

    def calculate_cost(centers, angles, ocx, ocy, otheta):
        side = compute_outer_side(centers, angles, ocx, ocy, otheta)
        penalty = 0.0
        verts = []
        for i in range(12):
            v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            verts.append(v)
        for i in range(12):
            for j in range(i + 1, 12):
                if not are_disjoint_sat(verts[i], verts[j]):
                    penalty += 100.0
        return side + penalty

    def optimize_sa(centers_init, angles_init, ocx_init, ocy_init, otheta_init, time_budget):
        T = 0.5
        T_min = 0.001
        alpha = 0.98
        steps_per_temp = 20
        current_centers = [c.copy() for c in centers_init]
        current_angles = list(angles_init)
        current_ocx = ocx_init
        current_ocy = ocy_init
        current_otheta = otheta_init
        current_cost = calculate_cost(current_centers, current_angles, current_ocx, current_ocy, current_otheta)
        best_centers = [c.copy() for c in current_centers]
        best_angles = list(current_angles)
        best_ocx = current_ocx
        best_ocy = current_ocy
        best_otheta = current_otheta
        best_cost = current_cost
        while T > T_min:
            for _ in range(steps_per_temp):
                if time.time() > deadline - time_budget:
                    return best_centers, best_angles, best_ocx, best_ocy, best_otheta, best_cost
                new_centers = [c.copy() for c in current_centers]
                new_angles = list(current_angles)
                new_ocx = current_ocx
                new_ocy = current_ocy
                new_otheta = current_otheta
                i = random.randint(0, 11)
                step_size = T * 0.3
                new_centers[i][0] += random.gauss(0, step_size)
                new_centers[i][1] += random.gauss(0, step_size)
                new_angles[i] += random.gauss(0, step_size * 5)
                if random.random() < 0.05:
                    new_ocx += random.gauss(0, T * 0.1)
                    new_ocy += random.gauss(0, T * 0.1)
                    new_otheta += random.gauss(0, T * 0.1)
                new_cost = calculate_cost(new_centers, new_angles, new_ocx, new_ocy, new_otheta)
                delta = new_cost - current_cost
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_centers = new_centers
                    current_angles = new_angles
                    current_ocx = new_ocx
                    current_ocy = new_ocy
                    current_otheta = new_otheta
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_centers = [c.copy() for c in current_centers]
                        best_angles = list(current_angles)
                        best_ocx = current_ocx
                        best_ocy = current_ocy
                        best_otheta = current_otheta
            T *= alpha
        return best_centers, best_angles, best_ocx, best_ocy, best_otheta, best_cost

    def optimize_slsqp(centers_init, angles_init, ocx_init, ocy_init, otheta_init, max_iters=80):
        n = len(centers_init)
        x0 = []
        for i in range(n):
            x0.extend([centers_init[i][0], centers_init[i][1], angles_init[i]])
        x0.extend([ocx_init, ocy_init, otheta_init])
        
        def objective(x):
            if time.time() > deadline - 10:
                return 1e10
            c = [[x[3*i], x[3*i+1]] for i in range(n)]
            a = [x[3*i+2] for i in range(n)]
            ocx, ocy, otheta = x[3*n], x[3*n+1], x[3*n+2]
            return compute_outer_side(c, a, ocx, ocy, otheta)
        
        constraints = []
        min_dist_sq = (math.sqrt(3) - 0.005)**2
        for i in range(n):
            for j in range(i+1, n):
                def dist_constraint(x, i=i, j=j):
                    xi = x[3*i]; yi = x[3*i+1]
                    xj = x[3*j]; yj = x[3*j+1]
                    return (xi - xj)**2 + (yi - yj)**2 - min_dist_sq
                constraints.append({'type': 'ineq', 'fun': dist_constraint})
        
        bounds = []
        for i in range(n):
            bounds.extend([(-6, 6), (-6, 6), (0, 60)])
        bounds.extend([(-3, 3), (-3, 3), (0, 60)])
        
        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                             constraints=constraints, options={'maxiter': max_iters, 'ftol': 1e-12})
            if result.success:
                c = [[result.x[3*i], result.x[3*i+1]] for i in range(n)]
                a = [result.x[3*i+2] for i in range(n)]
                ocx, ocy, otheta = result.x[3*n], result.x[3*n+1], result.x[3*n+2]
                side = compute_outer_side(c, a, ocx, ocy, otheta)
                verts = [get_hex_vertices(c[i][0], c[i][1], 1.0, a[i]) for i in range(n)]
                valid = True
                for i in range(n):
                    for j in range(i+1, n):
                        if not are_disjoint_sat(verts[i], verts[j]):
                            valid = False
                            break
                    if not valid: break
                if valid:
                    return c, a, ocx, ocy, otheta, side
        except Exception:
            pass
        return None

    def generate_lattice_config():
        d = math.sqrt(3) + 0.01
        u = np.array([d, 0.0])
        v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])
        centers = [np.array([0.0, 0.0])]
        ring1_vectors = [u, v, v - u, -u, -v, u - v]
        for vec in ring1_vectors:
            centers.append(vec.copy())
        ring2_candidates = []
        basis = [u, v, v - u, -u, -v, u - v]
        for i in range(6):
            p1 = ring1_vectors[i]
            for b in basis:
                p2 = p1 + b
                is_r0 = np.allclose(p2, np.array([0.0, 0.0]))
                is_r1 = False
                for p1j in ring1_vectors:
                    if np.allclose(p2, p1j):
                        is_r1 = True
                        break
                if not is_r0 and not is_r1:
                    duplicate = False
                    for pc in ring2_candidates:
                        if np.allclose(p2, pc):
                            duplicate = True
                            break
                    if not duplicate:
                        ring2_candidates.append(p2)
        ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
        for i in range(5):
            centers.append(ring2_candidates[i].copy())
        angles = [30.0] * 12
        return centers, angles, 0.0, 0.0, 0.0

    def generate_circular_config():
        centers = [np.array([0.0, 0.0])]
        r = math.sqrt(3) + 0.02
        for i in range(6):
            angle = i * math.pi / 3
            cx = r * math.cos(angle)
            cy = r * math.sin(angle)
            centers.append(np.array([cx, cy]))
        r2 = 2 * math.sqrt(3) + 0.02
        for i in range(5):
            angle = i * 2 * math.pi / 5 + math.pi / 5
            cx = r2 * math.cos(angle)
            cy = r2 * math.sin(angle)
            centers.append(np.array([cx, cy]))
        angles = [30.0] * 12
        return centers, angles, 0.0, 0.0, 0.0

    def generate_row_config():
        centers = []
        d = math.sqrt(3) + 0.01
        for i in range(6):
            centers.append(np.array([i * d, 0.0]))
        for i in range(6):
            centers.append(np.array([i * d + d / 2, d * math.sqrt(3) / 2]))
        angles = [30.0] * 12
        return centers, angles, 0.0, 0.0, 0.0

    def lattice_config_from_params(shear, scale, rot_deg, ox, oy):
        """
        Build 12 hex centers from a 3x4 hex lattice with uniform shear and scale.
        Lattice basis: e1 = scale * (1, 0), e2 = scale * (0.5 + shear, sqrt(3)/2).
        Positions: (i * e1 + j * e2) for i in 0..2, j in 0..3, centered at origin.
        All hexes share the same orientation (30 deg + rot_deg).
        """
        s = scale
        h = math.sqrt(3) / 2.0
        e1 = np.array([s, 0.0])
        e2 = np.array([s * (0.5 + shear), s * h])
        pts = []
        for i in range(3):
            for j in range(4):
                pts.append(i * e1 + j * e2)
        pts = np.array(pts)
        # Center the lattice at origin
        c = pts.mean(axis=0)
        pts = pts - c
        # Rotate
        theta = math.radians(rot_deg)
        R = np.array([[math.cos(theta), -math.sin(theta)],
                      [math.sin(theta), math.cos(theta)]])
        pts = pts @ R.T
        # Translate
        pts = pts + np.array([ox, oy])
        centers = [np.array(pts[k]) for k in range(12)]
        angles = [30.0 + rot_deg] * 12
        return centers, angles, 0.0, 0.0, 0.0

    def optimize_lattice_params():
        """
        Optimize 4 lattice parameters: shear, scale, rot_deg, and outer hex rotation.
        The lattice topology is fixed (3x4), so non-overlap is guaranteed for
        reasonable shear/scale values. We minimize the outer hex side length.
        """
        def lattice_side(shear, scale, rot_deg, otheta):
            centers, angles, _, _, _ = lattice_config_from_params(shear, scale, rot_deg, 0.0, 0.0)
            # Check non-overlap quickly
            verts = [get_hex_vertices(c[0], c[1], 1.0, a) for c, a in zip(centers, angles)]
            for i in range(12):
                for j in range(i + 1, 12):
                    if not are_disjoint_sat(verts[i], verts[j], eps=1e-6):
                        return 1e6
            return compute_outer_side(centers, angles, 0.0, 0.0, otheta)

        # Grid search over shear and scale to find a good starting point
        best_cost = float('inf')
        best_params = (0.0, math.sqrt(3) + 0.01, 0.0, 0.0)
        for shear in np.linspace(-0.3, 0.3, 13):
            for scale in np.linspace(math.sqrt(3) - 0.1, math.sqrt(3) + 0.3, 17):
                for rot_deg in [0.0, 15.0, 30.0, 45.0, 60.0]:
                    for otheta in [0.0, 15.0, 30.0, 45.0, 60.0]:
                        cost = lattice_side(shear, scale, rot_deg, otheta)
                        if cost < best_cost:
                            best_cost = cost
                            best_params = (shear, scale, rot_deg, otheta)
                        if time.time() > deadline - 20:
                            break
                    if time.time() > deadline - 20:
                        break
                if time.time() > deadline - 20:
                    break
            if time.time() > deadline - 20:
                break

        # Local refinement via Nelder-Mead on the 4 parameters
        x0 = np.array(best_params)
        def obj(x):
            if time.time() > deadline - 10:
                return 1e10
            return lattice_side(x[0], x[1], x[2], x[3])
        try:
            from scipy.optimize import minimize
            res = minimize(obj, x0, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_cost:
                best_params = tuple(res.x)
                best_cost = res.fun
        except Exception:
            pass

        # Fine polish
        for scale in [0.01, 0.002, 0.0005]:
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for ds in [-scale, 0, scale]:
                    for dd in [-scale, 0, scale]:
                        for dr in [-scale * 10, 0, scale * 10]:
                            for do in [-scale * 10, 0, scale * 10]:
                                new_cost = lattice_side(
                                    best_params[0] + ds,
                                    best_params[1] + dd,
                                    best_params[2] + dr,
                                    best_params[3] + do
                                )
                                if new_cost < best_cost - 1e-12:
                                    best_cost = new_cost
                                    best_params = (
                                        best_params[0] + ds,
                                        best_params[1] + dd,
                                        best_params[2] + dr,
                                        best_params[3] + do
                                    )
                                    improved = True

        centers, angles, _, _, _ = lattice_config_from_params(*best_params)
        return centers, angles, 0.0, 0.0, best_params[3], best_cost

    configs = [
        generate_lattice_config(),
        generate_circular_config(),
        generate_row_config()
    ]

    best_overall_cost = float('inf')
    best_centers = None
    best_angles = None
    best_ocx = 0.0
    best_ocy = 0.0
    best_otheta = 0.0

    # First try the lattice-parameter approach (fast, 4-parameter optimization)
    if time.time() < deadline - 30:
        try:
            lat_centers, lat_angles, lat_ocx, lat_ocy, lat_otheta, lat_cost = optimize_lattice_params()
            if lat_cost < best_overall_cost:
                best_overall_cost = lat_cost
                best_centers = lat_centers
                best_angles = lat_angles
                best_ocx = lat_ocx
                best_ocy = lat_ocy
                best_otheta = lat_otheta
        except Exception:
            pass

    for centers_init, angles_init, ocx_init, ocy_init, otheta_init in configs:
        if time.time() > deadline - 30:
            break

        # 1. Simulated Annealing for global exploration
        sa_centers, sa_angles, sa_ocx, sa_ocy, sa_otheta, sa_cost = optimize_sa(
            centers_init, angles_init, ocx_init, ocy_init, otheta_init, 20
        )

        # 2. SLSQP for constrained local optimization
        slsqp_result = optimize_slsqp(sa_centers, sa_angles, sa_ocx, sa_ocy, sa_otheta, max_iters=60)

        if slsqp_result is not None:
            sl_c, sl_a, sl_ocx, sl_ocy, sl_otheta, sl_cost = slsqp_result
            if sl_cost < best_overall_cost:
                best_overall_cost = sl_cost
                best_centers = sl_c
                best_angles = sl_a
                best_ocx = sl_ocx
                best_ocy = sl_ocy
                best_otheta = sl_otheta
        else:
            # Fall back to SA result if SLSQP failed
            if sa_cost < best_overall_cost:
                best_overall_cost = sa_cost
                best_centers = sa_centers
                best_angles = sa_angles
                best_ocx = sa_ocx
                best_ocy = sa_ocy
                best_otheta = sa_otheta

    if best_centers is None:
        best_centers, best_angles, best_ocx, best_ocy, best_otheta = generate_lattice_config()
        best_overall_cost = compute_outer_side(best_centers, best_angles, best_ocx, best_ocy, best_otheta)

    best_side = best_overall_cost
    final_ocx, final_ocy, final_otheta = best_ocx, best_ocy, best_otheta

    # High-precision multi-scale polish of outer hexagon parameters
    # This is the refinement machinery from the inspiration
    scales = [0.5, 0.1, 0.02, 0.005, 0.001]
    for scale in scales:
        if time.time() > deadline - 5:
            break
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # 3x3 grid around current best
            for dx in [-scale, 0, scale]:
                for dy in [-scale, 0, scale]:
                    for da_deg in [-scale*20, 0, scale*20]:
                        new_otheta = (final_otheta + da_deg) % 60
                        s = compute_outer_side(best_centers, best_angles, final_ocx + dx, final_ocy + dy, new_otheta)
                        if s < best_side - 1e-12:
                            best_side = s
                            final_ocx += dx
                            final_ocy += dy
                            final_otheta = new_otheta
                            improved = True

    # Final very fine polish
    if time.time() < deadline - 2:
        for _ in range(3):
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                for dx in [-0.001, 0, 0.001]:
                    for dy in [-0.001, 0, 0.001]:
                        for da_deg in [-0.01, 0, 0.01]:
                            new_otheta = (final_otheta + da_deg) % 60
                            s = compute_outer_side(best_centers, best_angles, final_ocx + dx, final_ocy + dy, new_otheta)
                            if s < best_side - 1e-14:
                                best_side = s
                                final_ocx += dx
                                final_ocy += dy
                                final_otheta = new_otheta
                                improved = True

    inner = [[float(c[0]), float(c[1]), float(a)] for c, a in zip(best_centers, best_angles)]
    outer_center = (float(final_ocx), float(final_ocy))
    outer_side = float(best_side)
    outer_angle = float(final_otheta)

    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END

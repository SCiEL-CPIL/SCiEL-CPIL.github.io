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
    np.random.seed(7)

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

    def compute_outer_side(centers, angles, ocx, ocy, otheta_deg):
        max_proj = 0.0
        otheta_rad = math.radians(otheta_deg)
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

    def smooth_penalty(centers, angles, min_dist_sq=3.0 - 0.0002):
        """Smooth squared-violation penalty for SA speed."""
        n = len(centers)
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist_sq = dx * dx + dy * dy
                if dist_sq < min_dist_sq:
                    violation = min_dist_sq - dist_sq
                    penalty += violation * violation
        return penalty

    def calculate_cost_fast(centers, angles, ocx, ocy, otheta_deg):
        side = compute_outer_side(centers, angles, ocx, ocy, otheta_deg)
        pen = smooth_penalty(centers, angles)
        if pen > 1e-8:
            side += 50.0 * math.sqrt(pen)
        return side

    def calculate_cost_exact(centers, angles, ocx, ocy, otheta_deg):
        side = compute_outer_side(centers, angles, ocx, ocy, otheta_deg)
        verts = [get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i]) for i in range(12)]
        for i in range(12):
            for j in range(i + 1, 12):
                if not are_disjoint_sat(verts[i], verts[j]):
                    return side + 100.0
        return side

    def exact_outer_polish(centers, angles, otheta_init):
        """
        For fixed inner hexagons, find optimal outer (ocx, ocy, otheta) exactly.
        The outer hexagon must contain all inner vertices. For a given orientation,
        the optimal center is the centroid of the bounding box of projected points
        on each of the 6 face normals. We ternary-search over orientation.
        """
        # Precompute all 72 inner vertex positions
        all_verts = []
        for i in range(12):
            cx, cy = centers[i]
            angle_rad = math.radians(angles[i])
            thetas = angle_rad + np.arange(6) * (math.pi / 3)
            vx = cx + np.cos(thetas)
            vy = cy + np.sin(thetas)
            all_verts.append(np.column_stack((vx, vy)))
        all_verts = np.vstack(all_verts)  # (72, 2)

        def outer_side_for_theta(otheta_deg):
            otheta_rad = math.radians(otheta_deg)
            normal_angles = otheta_rad + math.pi / 6 + np.arange(6) * (math.pi / 3)
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            # Project all vertices onto each normal
            projs = all_verts[:, 0][:, np.newaxis] * nx[np.newaxis, :] + \
                    all_verts[:, 1][:, np.newaxis] * ny[np.newaxis, :]
            mins = np.min(projs, axis=0)
            maxs = np.max(projs, axis=0)
            # Optimal center: for each pair of opposite normals, center is midpoint
            # The 6 normals come in 3 opposite pairs. For pair (k, k+3):
            #   center projection = (mins[k] + maxs[k]) / 2 on normal k
            #   center projection = (mins[k+3] + maxs[k+3]) / 2 on normal k+3
            # These must be consistent. The optimal center minimizes the max extent.
            # Actually, for a regular hexagon, the constraint is:
            #   for each face normal n_k: max_proj <= s*sqrt(3)/2
            #   for each opposite normal n_{k+3}: -min_proj <= s*sqrt(3)/2
            # So we need: max(maxs[k]) - center_proj_k <= s*sqrt(3)/2
            # and: center_proj_{k+3} - mins[k+3] <= s*sqrt(3)/2
            # The optimal center satisfies: center_proj_k = (maxs[k] + mins[k+3]) / 2
            # Wait, that's not quite right. Let me think again.
            #
            # A regular hexagon with center (cx,cy) and side s has the property that
            # for each of the 6 face normals n_k (pointing outward), the half-width is s*sqrt(3)/2.
            # So: (v - c) · n_k <= s*sqrt(3)/2 for all v, all k.
            # And: (v - c) · (-n_k) <= s*sqrt(3)/2, i.e., c·n_k - v·n_k <= s*sqrt(3)/2.
            # So: v·n_k - c·n_k <= s*sqrt(3)/2 and c·n_k - v·n_k <= s*sqrt(3)/2
            # => |v·n_k - c·n_k| <= s*sqrt(3)/2 for all v, k.
            # The minimal s for a given c is: max over all v,k of |v·n_k - c·n_k| / (sqrt(3)/2)
            # For a given orientation, the optimal c minimizes max_{v,k} |v·n_k - c·n_k|.
            # This is a Chebyshev center problem in 2D with 72*6=432 linear constraints.
            # But since the hexagon is regular, the 6 normals are equally spaced, and
            # the optimal center is the intersection of the midpoints of the bounding
            # projections on opposite normal pairs.
            #
            # For normals n_0 and n_3 (opposite):
            #   c·n_0 = (max(v·n_0) + min(v·n_3)) / 2  ... no wait
            #   We need: max_v(v·n_0) - c·n_0 <= H and c·n_0 - min_v(v·n_0) <= H
            #   => H >= (max_v(v·n_0) - min_v(v·n_0)) / 2, with c·n_0 = (max+min)/2
            # Similarly for n_3: H >= (max_v(v·n_3) - min_v(v·n_3)) / 2
            # But c·n_3 = -c·n_0 (since n_3 = -n_0 for a regular hexagon? No, n_3 is opposite to n_0)
            # Actually n_{k+3} = -n_k for a regular hexagon's face normals.
            # So: c·n_{k+3} = -c·n_k
            # Constraint from n_k: |v·n_k - c·n_k| <= H for all v
            #   => c·n_k in [max_v(v·n_k) - H, min_v(v·n_k) + H]
            #   => H >= (max_v(v·n_k) - min_v(v·n_k)) / 2
            #   => c·n_k = (max_v(v·n_k) + min_v(v·n_k)) / 2
            # So for each of the 3 pairs of opposite normals, the optimal c·n_k is fixed,
            # and H_k = (max - min) / 2. The overall H = max(H_0, H_1, H_2).
            # And c is determined by the 3 equations c·n_k = m_k for k=0,1,2 (independent normals).
            #
            # So: H = max over 3 pairs of (range on that normal) / 2
            # s = 2H / sqrt(3)
            #
            # This means for a fixed orientation, the optimal side is simply:
            #   s = 2/sqrt(3) * max_k (range of projections on normal k) / 2
            #     = 1/sqrt(3) * max_k (range of projections on normal k)
            # where k ranges over the 3 independent normal directions (0,1,2).
            #
            # And the center is determined by: c·n_k = (max_k + min_k)/2 for k=0,1,2.

            ranges = []
            for k in range(3):
                r = maxs[k] - mins[k]
                ranges.append(r)
            H = max(ranges) / 2.0
            s = 2.0 * H / math.sqrt(3)

            # Compute center: solve c·n_0 = m_0, c·n_1 = m_1, c·n_2 = m_2
            # where m_k = (maxs[k] + mins[k]) / 2
            m0 = (maxs[0] + mins[0]) / 2.0
            m1 = (maxs[1] + mins[1]) / 2.0
            m2 = (maxs[2] + mins[2]) / 2.0
            # n_0, n_1, n_2 are three non-collinear vectors in 2D
            # c_x * nx[0] + c_y * ny[0] = m0
            # c_x * nx[1] + c_y * ny[1] = m1
            # c_x * nx[2] + c_y * ny[2] = m2
            # Solve using first two equations:
            det = nx[0] * ny[1] - nx[1] * ny[0]
            if abs(det) < 1e-12:
                # Degenerate, use first and third
                det = nx[0] * ny[2] - nx[2] * ny[0]
                if abs(det) < 1e-12:
                    return s, 0.0, 0.0
                cx = (m0 * ny[2] - m2 * ny[0]) / det
                cy = (nx[0] * m2 - nx[2] * m0) / det
            else:
                cx = (m0 * ny[1] - m1 * ny[0]) / det
                cy = (nx[0] * m1 - nx[1] * m0) / det
            return s, cx, cy

        # Ternary search over otheta in [0, 60)
        a, b = 0.0, 60.0
        for _ in range(80):
            m1 = a + (b - a) / 3.0
            m2 = b - (b - a) / 3.0
            s1, _, _ = outer_side_for_theta(m1)
            s2, _, _ = outer_side_for_theta(m2)
            if s1 < s2:
                b = m2
            else:
                a = m1
        otheta_opt = (a + b) / 2.0
        s_opt, cx_opt, cy_opt = outer_side_for_theta(otheta_opt)
        return cx_opt, cy_opt, otheta_opt, s_opt

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
        current_cost = calculate_cost_fast(current_centers, current_angles, current_ocx, current_ocy, current_otheta)
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
                new_cost = calculate_cost_fast(new_centers, new_angles, new_ocx, new_ocy, new_otheta)
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
            c = [[x[3 * i], x[3 * i + 1]] for i in range(n)]
            a = [x[3 * i + 2] for i in range(n)]
            ocx, ocy, otheta = x[3 * n], x[3 * n + 1], x[3 * n + 2]
            return calculate_cost_fast(c, a, ocx, ocy, otheta)

        constraints = []
        min_dist_sq = (math.sqrt(3) - 0.005) ** 2
        for i in range(n):
            for j in range(i + 1, n):
                def dist_constraint(x, i=i, j=j):
                    xi = x[3 * i]; yi = x[3 * i + 1]
                    xj = x[3 * j]; yj = x[3 * j + 1]
                    return (xi - xj) ** 2 + (yi - yj) ** 2 - min_dist_sq
                constraints.append({'type': 'ineq', 'fun': dist_constraint})

        bounds = []
        for i in range(n):
            bounds.extend([(-6, 6), (-6, 6), (0, 60)])
        bounds.extend([(-3, 3), (-3, 3), (0, 60)])

        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                              constraints=constraints, options={'maxiter': max_iters, 'ftol': 1e-12})
            if result.success:
                c = [[result.x[3 * i], result.x[3 * i + 1]] for i in range(n)]
                a = [result.x[3 * i + 2] for i in range(n)]
                ocx, ocy, otheta = result.x[3 * n], result.x[3 * n + 1], result.x[3 * n + 2]
                verts = [get_hex_vertices(c[i][0], c[i][1], 1.0, a[i]) for i in range(n)]
                valid = True
                for i in range(n):
                    for j in range(i + 1, n):
                        if not are_disjoint_sat(verts[i], verts[j]):
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    return c, a, ocx, ocy, otheta
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

    configs = [
        generate_lattice_config(),
        generate_circular_config(),
        generate_row_config()
    ]

    best_overall_side = float('inf')
    best_centers = None
    best_angles = None
    best_ocx = 0.0
    best_ocy = 0.0
    best_otheta = 0.0

    for centers_init, angles_init, ocx_init, ocy_init, otheta_init in configs:
        if time.time() > deadline - 40:
            break

        # 1. Simulated Annealing for global exploration
        sa_centers, sa_angles, sa_ocx, sa_ocy, sa_otheta, sa_cost = optimize_sa(
            centers_init, angles_init, ocx_init, ocy_init, otheta_init, 25
        )

        # 2. SLSQP for constrained local optimization
        slsqp_result = optimize_slsqp(sa_centers, sa_angles, sa_ocx, sa_ocy, sa_otheta, max_iters=60)

        if slsqp_result is not None:
            sl_c, sl_a, sl_ocx, sl_ocy, sl_otheta = slsqp_result
            candidate_centers = sl_c
            candidate_angles = sl_a
        else:
            candidate_centers = sa_centers
            candidate_angles = sa_angles

        # 3. Exact outer polish: find optimal outer hexagon for these inner positions
        if time.time() < deadline - 15:
            ocx_opt, ocy_opt, otheta_opt, side_opt = exact_outer_polish(candidate_centers, candidate_angles, 0.0)
            # Verify validity
            verts = [get_hex_vertices(candidate_centers[i][0], candidate_centers[i][1], 1.0, candidate_angles[i]) for i in range(12)]
            valid = True
            for i in range(12):
                for j in range(i + 1, 12):
                    if not are_disjoint_sat(verts[i], verts[j]):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                if side_opt < best_overall_side:
                    best_overall_side = side_opt
                    best_centers = candidate_centers
                    best_angles = candidate_angles
                    best_ocx = ocx_opt
                    best_ocy = ocy_opt
                    best_otheta = otheta_opt

        # Also try SA result with exact outer polish
        if time.time() < deadline - 10:
            ocx_opt2, ocy_opt2, otheta_opt2, side_opt2 = exact_outer_polish(sa_centers, sa_angles, 0.0)
            verts2 = [get_hex_vertices(sa_centers[i][0], sa_centers[i][1], 1.0, sa_angles[i]) for i in range(12)]
            valid2 = True
            for i in range(12):
                for j in range(i + 1, 12):
                    if not are_disjoint_sat(verts2[i], verts2[j]):
                        valid2 = False
                        break
                if not valid2:
                    break
            if valid2 and side_opt2 < best_overall_side:
                best_overall_side = side_opt2
                best_centers = sa_centers
                best_angles = sa_angles
                best_ocx = ocx_opt2
                best_ocy = ocy_opt2
                best_otheta = otheta_opt2

    if best_centers is None:
        best_centers, best_angles, _, _, _ = generate_lattice_config()
        best_ocx, best_ocy, best_otheta, best_overall_side = exact_outer_polish(best_centers, best_angles, 0.0)

    # Final ultra-fine polish: nudge inner hexagons slightly and re-solve outer exactly
    if time.time() < deadline - 5:
        for _ in range(3):
            improved = False
            for i in range(12):
                for dx in [-0.01, 0.01]:
                    for dy in [-0.01, 0.01]:
                        if time.time() > deadline - 5:
                            break
                        new_centers = [c.copy() for c in best_centers]
                        new_centers[i][0] += dx
                        new_centers[i][1] += dy
                        # Check validity
                        verts = [get_hex_vertices(new_centers[k][0], new_centers[k][1], 1.0, best_angles[k]) for k in range(12)]
                        valid = True
                        for k in range(12):
                            for m in range(k + 1, 12):
                                if not are_disjoint_sat(verts[k], verts[m]):
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            continue
                        ocx_n, ocy_n, otheta_n, side_n = exact_outer_polish(new_centers, best_angles, 0.0)
                        if side_n < best_overall_side - 1e-10:
                            best_overall_side = side_n
                            best_centers = new_centers
                            best_ocx = ocx_n
                            best_ocy = ocy_n
                            best_otheta = otheta_n
                            improved = True
            if not improved:
                break

    inner = [[float(c[0]), float(c[1]), float(a)] for c, a in zip(best_centers, best_angles)]
    outer_center = (float(best_ocx), float(best_ocy))
    outer_side = float(best_overall_side)
    outer_angle = float(best_otheta)

    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END

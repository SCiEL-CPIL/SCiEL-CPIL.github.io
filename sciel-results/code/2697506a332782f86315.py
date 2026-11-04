import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0

    def get_hex_vertices(cx, cy, side, angle_rad):
        angles = angle_rad + np.arange(6) * np.pi / 3
        return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

    def get_hex_edges(verts):
        """Get edge normals and lengths for a hexagon."""
        normals = []
        for i in range(6):
            x_a, y_a = verts[i]
            x_b, y_b = verts[(i + 1) % 6]
            dx = x_b - x_a
            dy = y_b - y_a
            length = math.sqrt(dx * dx + dy * dy)
            if length > 1e-10:
                nx = -dy / length
                ny = dx / length
                normals.append((nx, ny))
        return normals

    def compute_outer_side(inner, cx, cy, theta_deg):
        """Compute minimum outer hexagon side length for given center and angle."""
        theta = math.radians(theta_deg)
        max_proj = 0.0
        for k in range(6):
            normal_angle = theta + math.pi / 6 + k * math.pi / 3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            for x, y, a in inner:
                verts = get_hex_vertices(x, y, 1.0, math.radians(a))
                proj = verts @ np.array([nx, ny])
                proj_centered = proj - (cx * nx + cy * ny)
                if proj_centered.max() > max_proj:
                    max_proj = proj_centered.max()
        return 2.0 * max_proj / math.sqrt(3)

    def min_gap_between_hexagons(hex1, hex2, eps=1e-6):
        """Compute minimum gap between two hexagons using SAT."""
        x1, y1, a1 = hex1
        x2, y2, a2 = hex2
        verts1 = get_hex_vertices(x1, y1, 1.0, math.radians(a1))
        verts2 = get_hex_vertices(x2, y2, 1.0, math.radians(a2))

        min_gap = float('inf')
        for verts in [verts1, verts2]:
            normals = get_hex_edges(verts)
            for nx, ny in normals:
                proj1 = verts1 @ np.array([nx, ny])
                proj2 = verts2 @ np.array([nx, ny])
                gap = min(proj2.max() - proj1.min(), proj1.max() - proj2.min())
                if gap < min_gap:
                    min_gap = gap
        return min_gap

    def hexagons_disjoint(inner, eps=1e-6):
        """Check if all hexagons are disjoint."""
        n = len(inner)
        for i in range(n):
            for j in range(i + 1, n):
                if min_gap_between_hexagons(inner[i], inner[j], eps) < eps:
                    return False
        return True

    def generate_ring_topology():
        """Generate 1+6+5 ring topology."""
        eps = 0.02
        d = math.sqrt(3) + eps

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
                is_r0 = np.allclose(p2, np.array([0.0, 0.0]), atol=1e-9)
                is_r1 = False
                for p1j in ring1_vectors:
                    if np.allclose(p2, p1j, atol=1e-9):
                        is_r1 = True
                        break
                if not is_r0 and not is_r1:
                    duplicate = False
                    for pc in ring2_candidates:
                        if np.allclose(p2, pc, atol=1e-9):
                            duplicate = True
                            break
                    if not duplicate:
                        ring2_candidates.append(p2)

        ring2_candidates.sort(key=lambda p: np.linalg.norm(p))

        for i in range(5):
            centers.append(ring2_candidates[i].copy())

        return [[c[0], c[1], 30.0] for c in centers]

    def generate_hex6plus6_topology():
        """Generate two clusters of 6 hexagons."""
        eps = 0.02
        d = math.sqrt(3) + eps

        u = np.array([d, 0.0])
        v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])

        centers = []

        # First cluster: center + 5 neighbors
        centers.append(np.array([0.0, 0.0]))
        ring1 = [u, v, v - u, -u, -v]
        for vec in ring1:
            centers.append(vec.copy())

        # Second cluster: offset
        offset = np.array([2.0 * d, 0.0])
        centers.append(offset + np.array([0.0, 0.0]))
        ring1_2 = [u, v, v - u, -u, -v]
        for vec in ring1_2:
            centers.append(offset + vec.copy())

        return [[c[0], c[1], 30.0] for c in centers]

    def generate_rect_topology():
        """Generate a rectangular grid."""
        eps = 0.02
        d = math.sqrt(3) + eps

        row_spacing = d * math.sqrt(3)
        col_spacing = d

        centers = []
        for row in range(3):
            for col in range(4):
                x = col * col_spacing
                y = row * row_spacing
                if row % 2 == 1:
                    x += col_spacing / 2.0
                centers.append(np.array([x, y]))

        return [[c[0], c[1], 30.0] for c in centers]

    def evaluate_topology(inner):
        """Evaluate a topology and return best configuration."""
        all_verts = []
        for x, y, a in inner:
            verts = get_hex_vertices(x, y, 1.0, math.radians(a))
            all_verts.extend(verts)

        cx = sum(p[0] for p in all_verts) / len(all_verts)
        cy = sum(p[1] for p in all_verts) / len(all_verts)

        best_side = float('inf')
        best_angle = 0.0

        for theta_deg in np.linspace(0, 60, 61):
            s = compute_outer_side(inner, cx, cy, theta_deg)
            if s < best_side:
                best_side = s
                best_angle = theta_deg

        return inner, cx, cy, best_angle, best_side

    # Generate multiple topologies
    topologies = [
        ("ring", generate_ring_topology()),
        ("hex6plus6", generate_hex6plus6_topology()),
        ("rect", generate_rect_topology()),
    ]

    best_overall = None
    best_overall_side = float('inf')

    for name, inner in topologies:
        if time.time() > deadline - 30.0:
            break

        if not hexagons_disjoint(inner):
            continue

        inner_eval, cx_eval, cy_eval, angle_eval, side_eval = evaluate_topology(inner)

        if side_eval < best_overall_side:
            best_overall = (inner_eval, cx_eval, cy_eval, angle_eval, side_eval)
            best_overall_side = side_eval

    if best_overall is None:
        inner = generate_ring_topology()
        best_overall = evaluate_topology(inner)
        best_overall_side = best_overall[4]

    inner, cx, cy, best_angle, best_side = best_overall

    # Homotopy continuation: track from large container to minimal
    # Start with a large outer hexagon and shrink
    def homotopy_continuation(inner_init, cx_init, cy_init, theta_init, s_init):
        """Use homotopy continuation to find minimal side length."""
        s_current = s_init
        s_min = s_init
        inner_current = [list(h) for h in inner_init]
        cx_current, cy_current, theta_current = cx_init, cy_init, theta_init

        # Number of homotopy steps
        n_steps = 30
        s_target = s_init * 0.7  # Try to shrink by 30%

        for step in range(n_steps):
            if time.time() > deadline - 20.0:
                break

            s_next = s_init + (s_target - s_init) * (step + 1) / n_steps

            # Try to find a valid configuration at this side length
            # Use SLSQP with constraints
            def objective(params):
                inner_local = []
                for i in range(N):
                    x, y, a = params[3 * i], params[3 * i + 1], params[3 * i + 2]
                    inner_local.append([x, y, math.degrees(a)])
                cx_l, cy_l, theta_l = params[3 * N], params[3 * N + 1], params[3 * N + 2]
                theta_l_deg = math.degrees(theta_l)

                side = compute_outer_side(inner_local, cx_l, cy_l, theta_l_deg)

                # Penalty for exceeding target side
                penalty = 0.0
                if side > s_next:
                    penalty = 100.0 * (side - s_next) ** 2

                return side + penalty

            def constraints(params):
                inner_local = []
                for i in range(N):
                    x, y, a = params[3 * i], params[3 * i + 1], params[3 * i + 2]
                    inner_local.append([x, y, math.degrees(a)])

                # Constraint: all pairs must be disjoint
                cons = []
                for i in range(N):
                    for j in range(i + 1, N):
                        x1, y1, a1 = inner_local[i]
                        x2, y2, a2 = inner_local[j]
                        gap = min_gap_between_hexagons(inner_local[i], inner_local[j])
                        cons.append({'type': 'ineq', 'fun': lambda p, i=i, j=j: min_gap_between_hexagons(
                            [p[3 * i], p[3 * i + 1], math.degrees(p[3 * i + 2])],
                            [p[3 * j], p[3 * j + 1], math.degrees(p[3 * j + 2])]
                        ) - 1e-6})
                return cons

            params = []
            for x, y, a in inner_current:
                params.extend([x, y, math.radians(a)])
            params.extend([cx_current, cy_current, math.radians(theta_current)])

            try:
                result = minimize(
                    objective,
                    params,
                    method='SLSQP',
                    constraints=constraints,
                    options={'maxiter': 200, 'ftol': 1e-10}
                )

                if result.success or result.fun < s_current:
                    inner_opt = []
                    for i in range(N):
                        x, y, a = result.x[3 * i], result.x[3 * i + 1], result.x[3 * i + 2]
                        inner_opt.append([x, y, math.degrees(a)])
                    cx_opt, cy_opt = result.x[3 * N], result.x[3 * N + 1]
                    theta_opt = math.degrees(result.x[3 * N + 2])

                    if hexagons_disjoint(inner_opt):
                        s_opt = compute_outer_side(inner_opt, cx_opt, cy_opt, theta_opt)
                        if s_opt < s_min:
                            s_min = s_opt
                            inner_current = inner_opt
                            cx_current, cy_current, theta_current = cx_opt, cy_opt, theta_opt
            except Exception:
                pass

        return inner_current, cx_current, cy_current, theta_current, s_min

    # Try homotopy continuation
    if time.time() < deadline - 25.0:
        inner_h, cx_h, cy_h, theta_h, s_h = homotopy_continuation(inner, cx, cy, best_angle, best_side)
        if s_h < best_side:
            inner, cx, cy, best_angle, best_side = inner_h, cx_h, cy_h, theta_h, s_h

    # Simulated annealing
    if time.time() < deadline - 15.0:
        T = 0.05
        T_min = 0.001
        cooling = 0.995
        max_iter = 2000
        current_inner = [list(h) for h in inner]
        current_cx, current_cy, current_theta = cx, cy, best_angle
        current_side = best_side
        best_inner = [list(h) for h in inner]
        best_cx, best_cy, best_theta = cx, cy, best_angle
        best_side_sa = best_side

        for it in range(max_iter):
            if time.time() > deadline - 18.0:
                break

            new_inner = [list(h) for h in current_inner]
            new_cx, new_cy, new_theta = current_cx, current_cy, current_theta

            r = random.random()
            if r < 0.7:
                idx = random.randint(0, N - 1)
                dx = random.gauss(0, 0.03)
                dy = random.gauss(0, 0.03)
                new_inner[idx][0] += dx
                new_inner[idx][1] += dy
            elif r < 0.85:
                idx = random.randint(0, N - 1)
                da = random.gauss(0, 1.0)
                new_inner[idx][2] += da
            elif r < 0.92:
                dx = random.gauss(0, 0.02)
                dy = random.gauss(0, 0.02)
                new_cx += dx
                new_cy += dy
            else:
                dtheta = random.gauss(0, 0.5)
                new_theta += dtheta

            if not hexagons_disjoint(new_inner):
                T *= cooling
                continue

            new_side = compute_outer_side(new_inner, new_cx, new_cy, new_theta)
            delta = new_side - current_side

            if delta < 0 or random.random() < math.exp(-delta / T):
                current_inner = new_inner
                current_cx, current_cy, current_theta = new_cx, new_cy, new_theta
                current_side = new_side

                if current_side < best_side_sa - 1e-10:
                    best_side_sa = current_side
                    best_inner = [list(h) for h in current_inner]
                    best_cx, best_cy, best_theta = current_cx, current_cy, current_theta

            T *= cooling

        if best_side_sa < best_side:
            inner = best_inner
            cx, cy = best_cx, best_cy
            best_angle = best_theta
            best_side = best_side_sa

    # Final polish with Nelder-Mead on positions only
    if time.time() < deadline - 8.0:
        def nm_objective(params):
            inner_local = []
            for i in range(N):
                x, y = params[2 * i], params[2 * i + 1]
                inner_local.append([x, y, inner[i][2]])
            return compute_outer_side(inner_local, cx, cy, best_angle)

        params0 = []
        for x, y, a in inner:
            params0.extend([x, y])

        try:
            result = minimize(
                nm_objective,
                params0,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10}
            )

            inner_opt = []
            for i in range(N):
                x, y = result.x[2 * i], result.x[2 * i + 1]
                inner_opt.append([x, y, inner[i][2]])

            if hexagons_disjoint(inner_opt):
                s_opt = compute_outer_side(inner_opt, cx, cy, best_angle)
                if s_opt < best_side:
                    inner = inner_opt
                    best_side = s_opt
        except Exception:
            pass

    # Final angle scan
    if time.time() < deadline - 3.0:
        best_angle_final = best_angle
        best_side_final = best_side
        for theta_deg in np.linspace(best_angle - 5, best_angle + 5, 21):
            s = compute_outer_side(inner, cx, cy, theta_deg)
            if s < best_side_final:
                best_side_final = s
                best_angle_final = theta_deg

        best_angle = best_angle_final
        best_side = best_side_final

    outer_center = (cx, cy)
    outer_side = best_side
    outer_angle = best_angle

    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END

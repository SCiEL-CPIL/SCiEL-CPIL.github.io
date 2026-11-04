import math
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 11

def construct():
    start_time = time.time()
    deadline = 150.0
    safety_margin = 15.0

    # Geometry constants
    d = math.sqrt(3)
    spacing = 2 * d  # Distance between centers of touching unit hexagons (side 1)

    # Relative vertices of a unit hexagon (pointy-top, centered at origin)
    # Angles: 0, 60, 120, 180, 240, 300
    verts_rel = np.array([
        [math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] for i in range(6)
    ])

    # Generate base centers: Center (0,0) + Ring 1 (6 hexagons)
    base_centers = [(0.0, 0.0)]
    for i in range(6):
        angle = i * math.pi / 3
        x = spacing * math.cos(angle)
        y = spacing * math.sin(angle)
        base_centers.append((x, y))

    # Generate Ring 2 centers (12 hexagons)
    ring2_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                x = spacing * (q + r / 2.0)
                y = spacing * (d / 2.0) * r
                ring2_centers.append((x, y))

    # Ensure we have 12 ring 2 centers
    if len(ring2_centers) != 12:
        for i in range(12):
            angle = i * math.pi / 6
            ring2_centers.append((2 * spacing * math.cos(angle), 2 * spacing * math.sin(angle)))

    def get_hex_verts(cx, cy, angle):
        c = math.cos(angle)
        s = math.sin(angle)
        rx = verts_rel[:, 0] * c - verts_rel[:, 1] * s
        ry = verts_rel[:, 0] * s + verts_rel[:, 1] * c
        return np.column_stack([rx + cx, ry + cy])

    def sat_overlap(v1, v2):
        """Return maximum overlap depth (positive if overlapping, negative if separated)."""
        max_overlap = 0.0
        for v in (v1, v2):
            for i in range(6):
                p1 = v[i]
                p2 = v[(i + 1) % 6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-10:
                    continue
                normal = normal / norm
                proj1 = np.dot(v1, normal)
                proj2 = np.dot(v2, normal)
                overlap = min(proj1.max(), proj2.max()) - max(proj1.min(), proj2.min())
                max_overlap = max(max_overlap, overlap)
        return max_overlap

    def compute_S_from_points(points, theta):
        """
        Compute minimal side length S of a regular hexagon with rotation theta
        that encloses the given points.
        """
        k_angles = theta + np.arange(6) * math.pi / 3
        normals = np.stack([np.cos(k_angles), np.sin(k_angles)], axis=-1)
        dots = np.dot(points, normals.T)
        extents = np.max(dots, axis=1) - np.min(dots, axis=1)
        max_extent = np.max(extents)
        return max_extent / math.sqrt(3)

    def compute_S_and_theta(params, theta):
        n_hex = len(params) // 3
        xs = params[0::3]
        ys = params[1::3]
        thetas = params[2::3]
        all_pts = []
        for i in range(n_hex):
            verts = get_hex_verts(xs[i], ys[i], thetas[i])
            all_pts.append(verts)
        all_pts = np.vstack(all_pts)
        return compute_S_from_points(all_pts, theta)

    def objective(params, theta):
        if time.time() > deadline - safety_margin:
            return 1e10
        n_hex = len(params) // 3
        xs = params[0::3]
        ys = params[1::3]
        thetas = params[2::3]
        verts_list = [get_hex_verts(xs[i], ys[i], thetas[i]) for i in range(n_hex)]
        penalty = 0.0
        for i in range(n_hex):
            for j in range(i + 1, n_hex):
                overlap = sat_overlap(verts_list[i], verts_list[j])
                if overlap > 0:
                    penalty += overlap * overlap * 1000.0
        S = compute_S_and_theta(params, theta)
        return S + penalty

    # Step 1: Enumerate combinations and find best rigid configuration
    best_S_rigid = float('inf')
    best_params_rigid = None
    best_theta_rigid = 0.0
    top_candidates = []

    all_combinations = list(__import__('itertools').combinations(range(12), 4))

    base_verts_list = [get_hex_verts(cx, cy, 0.0) for cx, cy in base_centers]
    base_pts = np.vstack(base_verts_list)

    for combo in all_combinations:
        if time.time() > deadline - safety_margin:
            break
        r2_centers = [ring2_centers[i] for i in combo]
        r2_verts_list = [get_hex_verts(cx, cy, 0.0) for cx, cy in r2_centers]
        r2_pts = np.vstack(r2_verts_list)
        all_pts = np.vstack([base_pts, r2_pts])

        thetas = np.linspace(0, math.pi / 3, 100)
        min_S = float('inf')
        best_theta = 0.0
        for theta in thetas:
            S_val = compute_S_from_points(all_pts, theta)
            if S_val < min_S:
                min_S = S_val
                best_theta = theta

        if min_S < best_S_rigid:
            if len(top_candidates) < 10:
                top_candidates.append((min_S, combo, best_theta))
            else:
                worst_idx = max(range(len(top_candidates)), key=lambda i: top_candidates[i][0])
                if min_S < top_candidates[worst_idx][0]:
                    top_candidates[worst_idx] = (min_S, combo, best_theta)
            best_S_rigid = min_S
            best_theta_rigid = best_theta
            final_centers = base_centers + r2_centers
            params = np.zeros(N * 3)
            for i, (cx, cy) in enumerate(final_centers):
                params[3 * i] = cx
                params[3 * i + 1] = cy
                params[3 * i + 2] = 0.0
            best_params_rigid = params

    # Step 2: Refine the top candidates using Differential Evolution
    best_S_final = best_S_rigid
    best_params_final = best_params_rigid
    best_theta_final = best_theta_rigid

    top_candidates.sort(key=lambda x: x[0])

    for S_cand, combo, theta_cand in top_candidates[:5]:
        if time.time() > deadline - safety_margin:
            break

        r2_centers = [ring2_centers[i] for i in combo]
        final_centers = base_centers + r2_centers
        init_params = np.zeros(N * 3)
        for i, (cx, cy) in enumerate(final_centers):
            init_params[3 * i] = cx
            init_params[3 * i + 1] = cy
            init_params[3 * i + 2] = 0.0

        # Define bounds for Differential Evolution
        # Positions: roughly within the outer hexagon
        # Outer side S ~ 3.5, so coordinates roughly in [-S, S]
        # Inner angles: [0, pi/3)
        bounds = []
        for i in range(N):
            bounds.append((-4.0, 4.0))  # x
            bounds.append((-4.0, 4.0))  # y
            bounds.append((0.0, math.pi / 3))  # angle
        bounds.append((0.0, math.pi / 3))  # outer theta

        def de_objective(x):
            p = x[:33]
            t = x[33]
            return objective(p, t)

        try:
            # Use Differential Evolution for global search
            de_res = differential_evolution(de_objective, bounds, maxiter=50, popsize=15, tol=1e-5,
                                            seed=42, polish=False, update_interval=10)

            if de_res.fun < best_S_final:
                p_res = de_res.x[:33]
                t_res = de_res.x[33]

                # Check validity
                n_hex = len(p_res) // 3
                xs = p_res[0::3]
                ys = p_res[1::3]
                thetas = p_res[2::3]
                verts_list = [get_hex_verts(xs[i], ys[i], thetas[i]) for i in range(n_hex)]

                max_overlap = 0.0
                for i in range(n_hex):
                    for j in range(i + 1, n_hex):
                        ov = sat_overlap(verts_list[i], verts_list[j])
                        if ov > max_overlap:
                            max_overlap = ov

                if max_overlap < 1e-6:
                    all_pts = np.vstack(verts_list)
                    S_actual = compute_S_from_points(all_pts, t_res)

                    if S_actual < best_S_final:
                        best_S_final = S_actual
                        best_params_final = p_res
                        best_theta_final = t_res

        except Exception:
            continue

    # Construct final output
    if best_params_final is None:
        best_params_final = np.zeros(N * 3)
        for i, (cx, cy) in enumerate(base_centers + [ring2_centers[i] for i in range(4)]):
            best_params_final[3 * i] = cx
            best_params_final[3 * i + 1] = cy
            best_params_final[3 * i + 2] = 0.0
        best_theta_final = 0.0
        best_S_final = compute_S_and_theta(best_params_final, best_theta_final)

    inner = []
    for i in range(N):
        x = best_params_final[3 * i]
        y = best_params_final[3 * i + 1]
        angle_deg = math.degrees(best_params_final[3 * i + 2])
        inner.append([x, y, angle_deg])

    return inner, (0.0, 0.0), best_S_final, math.degrees(best_theta_final)

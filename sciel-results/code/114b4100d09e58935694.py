import math
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin

    def get_hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return np.array(vertices)

    def sat_min_separation(v1, v2):
        """Compute minimum separation between two convex hexagons using SAT.
        Returns negative value if overlapping (penetration depth), positive if separated."""
        min_sep = float('inf')
        for verts in [v1, v2]:
            for i in range(6):
                p1 = verts[i]
                p2 = verts[(i + 1) % 6]
                edge = p2 - p1
                nx, ny = -edge[1], edge[0]
                # Normalize normal
                norm = math.sqrt(nx * nx + ny * ny)
                if norm < 1e-12:
                    continue
                nx /= norm
                ny /= norm
                proj1 = v1[:, 0] * nx + v1[:, 1] * ny
                proj2 = v2[:, 0] * nx + v2[:, 1] * ny
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                sep = min(max1 - min2, max2 - min1)
                if sep < min_sep:
                    min_sep = sep
                    if min_sep <= 0:
                        return min_sep
        return min_sep

    def compute_outer_side(centers, angles, otheta):
        """Compute minimum side length of regular hexagon containing all inner hexagons."""
        otheta_rad = math.radians(otheta)
        max_proj = 0.0
        for k in range(6):
            normal_angle = otheta_rad + math.pi / 6 + k * math.pi / 3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            for i in range(len(centers)):
                cx, cy = centers[i]
                angle_rad = math.radians(angles[i])
                for vi in range(6):
                    vert_angle = angle_rad + vi * math.pi / 3
                    vx = cx + math.cos(vert_angle)
                    vy = cy + math.sin(vert_angle)
                    proj = vx * nx + vy * ny
                    if proj > max_proj:
                        max_proj = proj
        return 2.0 * max_proj / math.sqrt(3)

    def get_penalty(centers, angles):
        """Compute smooth overlap penalty."""
        verts = []
        for i in range(12):
            v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            verts.append(v)
        penalty = 0.0
        for i in range(12):
            for j in range(i + 1, 12):
                sep = sat_min_separation(verts[i], verts[j])
                if sep < 0:
                    penalty += sep * sep
        return penalty

    def is_valid(centers, angles, tol=1e-9):
        """Check if all inner hexagons are disjoint."""
        verts = []
        for i in range(12):
            v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            verts.append(v)
        for i in range(12):
            for j in range(i + 1, 12):
                if sat_min_separation(verts[i], verts[j]) < -tol:
                    return False
        return True

    def objective(params):
        if time.time() > deadline:
            return 1e9
        centers = [[params[i * 2], params[i * 2 + 1]] for i in range(12)]
        angles = [params[24 + i] for i in range(12)]
        otheta = params[36]
        side = compute_outer_side(centers, angles, otheta)
        pen = get_penalty(centers, angles)
        return side + 200.0 * pen

    def generate_hexagonal_cluster():
        """Hexagonal cluster: center + ring of 6 + 5 from second ring."""
        d = math.sqrt(3) + 0.01
        u = np.array([d, 0.0])
        v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])
        centers = [np.array([0.0, 0.0])]
        ring1_vectors = [u, v, v - u, -u, -v, u - v]
        for vec in ring1_vectors:
            centers.append(vec.copy())
        # Add 5 from second ring
        basis = [u, v, v - u, -u, -v, u - v]
        ring2_candidates = []
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
        return centers, angles

    def generate_rectangular_grid():
        """3x4 grid."""
        dx = math.sqrt(3) + 0.01
        dy = 3.0 + 0.01
        centers = []
        angles = []
        for row in range(3):
            for col in range(4):
                cx = (col - 1.5) * dx
                cy = (row - 1.0) * dy
                centers.append([cx, cy])
                angles.append(30.0)
        return centers, angles

    def generate_two_rings():
        """Two concentric rings of 6."""
        r1 = 2.0
        r2 = 4.0
        centers = []
        angles = []
        for i in range(6):
            theta = i * math.pi / 3
            centers.append([r1 * math.cos(theta), r1 * math.sin(theta)])
            angles.append(30.0)
        for i in range(6):
            theta = i * math.pi / 3 + math.pi / 6
            centers.append([r2 * math.cos(theta), r2 * math.sin(theta)])
            angles.append(30.0)
        return centers, angles

    def params_from_config(centers, angles, otheta=0.0):
        params = []
        for cx, cy in centers:
            params.append(cx)
            params.append(cy)
        for a in angles:
            params.append(a)
        params.append(otheta)
        return np.array(params)

    # Generate initial configurations
    configs = []
    c1, a1 = generate_hexagonal_cluster()
    configs.append(params_from_config(c1, a1))
    c2, a2 = generate_rectangular_grid()
    configs.append(params_from_config(c2, a2))
    c3, a3 = generate_two_rings()
    configs.append(params_from_config(c3, a3))

    best_params = None
    best_val = float('inf')

    # Phase 1: Nelder-Mead from each initial config
    for init_params in configs:
        if time.time() > deadline - 20:
            break
        try:
            res = minimize(objective, init_params, method='Nelder-Mead',
                          options={'maxiter': 1500, 'xatol': 1e-7, 'fatol': 1e-9})
            centers = [[res.x[i * 2], res.x[i * 2 + 1]] for i in range(12)]
            angles = [res.x[24 + i] for i in range(12)]
            if is_valid(centers, angles):
                val = compute_outer_side(centers, angles, res.x[36])
                if val < best_val:
                    best_val = val
                    best_params = res.x.copy()
            else:
                init_centers = [[init_params[i * 2], init_params[i * 2 + 1]] for i in range(12)]
                init_angles = [init_params[24 + i] for i in range(12)]
                if is_valid(init_centers, init_angles):
                    val = compute_outer_side(init_centers, init_angles, init_params[36])
                    if val < best_val:
                        best_val = val
                        best_params = init_params.copy()
        except Exception:
            pass

    # Phase 2: Fine polish with Nelder-Mead from best solution
    if best_params is not None and time.time() < deadline - 15:
        try:
            polish_res = minimize(objective, best_params, method='Nelder-Mead',
                                 options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11})
            centers = [[polish_res.x[i * 2], polish_res.x[i * 2 + 1]] for i in range(12)]
            angles = [polish_res.x[24 + i] for i in range(12)]
            if is_valid(centers, angles):
                val = compute_outer_side(centers, angles, polish_res.x[36])
                if val < best_val:
                    best_val = val
                    best_params = polish_res.x.copy()
        except Exception:
            pass

    # Phase 3: If no valid solution, use fallback
    if best_params is None:
        best_params = configs[0].copy()
        best_val = compute_outer_side(
            [[best_params[i * 2], best_params[i * 2 + 1]] for i in range(12)],
            [best_params[24 + i] for i in range(12)],
            best_params[36]
        )

    # Phase 4: Multi-scale outer angle optimization
    centers = [[best_params[i * 2], best_params[i * 2 + 1]] for i in range(12)]
    angles = [best_params[24 + i] for i in range(12)]
    best_side = best_val
    best_otheta = best_params[36]

    for da in np.linspace(-30, 30, 121):
        s = compute_outer_side(centers, angles, best_otheta + da)
        if s < best_side:
            best_side = s
            best_otheta = best_otheta + da

    for da in np.linspace(-5, 5, 201):
        s = compute_outer_side(centers, angles, best_otheta + da)
        if s < best_side:
            best_side = s
            best_otheta = best_otheta + da

    for da in np.linspace(-1, 1, 201):
        s = compute_outer_side(centers, angles, best_otheta + da)
        if s < best_side:
            best_side = s
            best_otheta = best_otheta + da

    for da in np.linspace(-0.1, 0.1, 101):
        s = compute_outer_side(centers, angles, best_otheta + da)
        if s < best_side:
            best_side = s
            best_otheta = best_otheta + da

    # Phase 5: Simultaneous position + angle optimization with tighter constraints
    # Use a barrier-augmented objective that heavily penalizes overlaps
    def objective_tight(params):
        if time.time() > deadline:
            return 1e9
        centers = [[params[i * 2], params[i * 2 + 1]] for i in range(12)]
        angles = [params[24 + i] for i in range(12)]
        otheta = params[36]
        side = compute_outer_side(centers, angles, otheta)
        pen = get_penalty(centers, angles)
        return side + 500.0 * pen

    if time.time() < deadline - 10:
        try:
            tight_res = minimize(objective_tight, best_params, method='Nelder-Mead',
                                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            centers = [[tight_res.x[i * 2], tight_res.x[i * 2 + 1]] for i in range(12)]
            angles = [tight_res.x[24 + i] for i in range(12)]
            if is_valid(centers, angles):
                val = compute_outer_side(centers, angles, tight_res.x[36])
                if val < best_side:
                    best_side = val
                    best_otheta = tight_res.x[36]
                    best_params = tight_res.x.copy()
        except Exception:
            pass

    inner = [[c[0], c[1], a] for c, a in zip(centers, angles)]
    outer_center = (0.0, 0.0)
    outer_side = best_side
    outer_angle = best_otheta

    return inner, outer_center, outer_side, outer_angle

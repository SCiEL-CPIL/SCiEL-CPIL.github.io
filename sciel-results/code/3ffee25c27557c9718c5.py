import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE_MARGIN = 12.0


def construct():
    deadline = time.time() + 138.0
    random.seed(42)

    def get_hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return np.array(vertices)

    def get_hex_vertices_rad(cx, cy, side, angle_rad):
        angles = angle_rad + np.arange(6) * np.pi / 3
        return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

    def project_range(vertices, nx, ny):
        projs = np.dot(vertices, [nx, ny])
        return np.min(projs), np.max(projs)

    def sat_min_gap(v1, v2):
        """Return minimum signed gap between two hexagons (positive = separated, negative = overlap)."""
        min_sep = float('inf')
        for verts in (v1, v2):
            for i in range(6):
                p1 = verts[i]
                p2 = verts[(i + 1) % 6]
                edge = p2 - p1
                length = np.linalg.norm(edge)
                if length < 1e-12:
                    continue
                nx, ny = -edge[1] / length, edge[0] / length
                min1, max1 = project_range(v1, nx, ny)
                min2, max2 = project_range(v2, nx, ny)
                sep = min(max1 - min2, max2 - min1)
                if sep < min_sep:
                    min_sep = sep
        return min_sep

    def sat_separated(v1, v2, eps=1e-9):
        return sat_min_gap(v1, v2) > -eps

    def compute_outer_side(centers, angles, ocx, ocy, otheta):
        """Compute minimal outer hexagon side length containing all inner hexagons."""
        max_proj = 0.0
        otheta_rad = math.radians(otheta)
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
                    proj = (vx - ocx) * nx + (vy - ocy) * ny
                    if proj > max_proj:
                        max_proj = proj
        return 2.0 * max_proj / math.sqrt(3)

    def is_valid(centers, angles, eps=1e-6):
        verts = []
        for i in range(12):
            v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            verts.append(v)
        for i in range(12):
            for j in range(i + 1, 12):
                if not sat_separated(verts[i], verts[j], eps):
                    return False
        return True

    def generate_hexagonal_cluster():
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
        return centers, angles

    def generate_rectangular_grid():
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

    def generate_hex_ring_optimal():
        d = math.sqrt(3) + 0.005
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
        return centers, angles

    def slsqp_optimize(initial_params, max_iter=300):
        def objective(params):
            if time.time() > deadline - DEADLINE_MARGIN:
                return 1e9
            centers = [[params[i * 2], params[i * 2 + 1]] for i in range(12)]
            angles = [params[24 + i] for i in range(12)]
            otheta = params[36]
            return compute_outer_side(centers, angles, 0.0, 0.0, otheta)

        def constraints():
            cons = []
            for i in range(12):
                for j in range(i + 1, 12):
                    def sep_constraint(params, i=i, j=j):
                        if time.time() > deadline - DEADLINE_MARGIN:
                            return 1.0
                        ci = [params[i * 2], params[i * 2 + 1]]
                        cj = [params[j * 2], params[j * 2 + 1]]
                        ai = params[24 + i]
                        aj = params[24 + j]
                        vi = get_hex_vertices(ci[0], ci[1], 1.0, ai)
                        vj = get_hex_vertices(cj[0], cj[1], 1.0, aj)
                        return sat_min_gap(vi, vj) - 1e-6
                    cons.append({'type': 'ineq', 'fun': sep_constraint})
            return cons

        try:
            res = minimize(objective, initial_params, method='SLSQP',
                          constraints=constraints(),
                          options={'maxiter': max_iter, 'ftol': 1e-12, 'disp': False})
            return res.x, res.fun
        except Exception:
            return initial_params, objective(initial_params)

    def nelder_polish(params, max_iter=800, penalty_weight=100.0):
        def obj(params):
            if time.time() > deadline - DEADLINE_MARGIN:
                return 1e9
            centers = [[params[i * 2], params[i * 2 + 1]] for i in range(12)]
            angles = [params[24 + i] for i in range(12)]
            otheta = params[36]
            side = compute_outer_side(centers, angles, 0.0, 0.0, otheta)
            penalty = 0.0
            verts = []
            for i in range(12):
                v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
                verts.append(v)
            for i in range(12):
                for j in range(i + 1, 12):
                    gap = sat_min_gap(verts[i], verts[j])
                    if gap < 0:
                        penalty += gap ** 2
            return side + penalty_weight * penalty

        try:
            res = minimize(obj, params, method='Nelder-Mead',
                          options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12})
            return res.x, res.fun
        except Exception:
            return params, obj(params)

    def random_walk_optimize(params, max_iters=3000, step_init=0.1):
        """Local random walk that refines positions while maintaining disjointness."""
        best_params = params.copy()
        best_side = compute_outer_side(
            [[best_params[i*2], best_params[i*2+1]] for i in range(12)],
            [best_params[24+i] for i in range(12)],
            0.0, 0.0, best_params[36]
        )
        current_params = params.copy()
        current_side = best_side
        step = step_init

        for _ in range(max_iters):
            if time.time() > deadline - DEADLINE_MARGIN:
                break
            if step < 1e-5:
                break

            new_params = current_params.copy()
            r = random.random()
            if r < 0.65:
                idx = random.randint(0, 11)
                new_params[idx * 2] += random.uniform(-step, step)
                new_params[idx * 2 + 1] += random.uniform(-step, step)
            elif r < 0.8:
                idx = random.randint(0, 11)
                new_params[24 + idx] += random.uniform(-2.0 * step, 2.0 * step)
            else:
                new_params[36] += random.uniform(-3.0 * step, 3.0 * step)

            centers = [[new_params[i*2], new_params[i*2+1]] for i in range(12)]
            angles = [new_params[24+i] for i in range(12)]
            if not is_valid(centers, angles, eps=1e-5):
                step *= 0.95
                continue

            new_side = compute_outer_side(centers, angles, 0.0, 0.0, new_params[36])
            if new_side < current_side - 1e-9:
                current_params = new_params
                current_side = new_side
                step *= 0.98
                if new_side < best_side - 1e-10:
                    best_params = new_params.copy()
                    best_side = new_side
            else:
                step *= 0.97

        return best_params, best_side

    def high_precision_polish(params):
        """Final tight polish with very small step sizes and higher penalty."""
        def obj(params):
            if time.time() > deadline - DEADLINE_MARGIN:
                return 1e9
            centers = [[params[i * 2], params[i * 2 + 1]] for i in range(12)]
            angles = [params[24 + i] for i in range(12)]
            otheta = params[36]
            side = compute_outer_side(centers, angles, 0.0, 0.0, otheta)
            penalty = 0.0
            verts = []
            for i in range(12):
                v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
                verts.append(v)
            for i in range(12):
                for j in range(i + 1, 12):
                    gap = sat_min_gap(verts[i], verts[j])
                    if gap < 0:
                        penalty += gap ** 2
            return side + 500.0 * penalty

        try:
            res = minimize(obj, params, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
            return res.x, res.fun
        except Exception:
            return params, obj(params)

    def optimize_outer_angle(centers, angles, current_otheta):
        """Optimize the outer hexagon rotation angle for minimal side length."""
        def soft_max_projections(otheta):
            otheta_rad = math.radians(otheta)
            max_projs = []
            for k in range(6):
                normal_angle = otheta_rad + math.pi / 6 + k * math.pi / 3
                nx = math.cos(normal_angle)
                ny = math.sin(normal_angle)
                max_proj = 0.0
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
                max_projs.append(max_proj)
            T = 0.01
            max_m = max(max_projs)
            exps = [math.exp((mp - max_m) / T) for mp in max_projs]
            s = sum(exps)
            soft_max = max_m + T * math.log(s)
            return 2.0 * soft_max / math.sqrt(3)

        best_otheta = current_otheta
        best_val = soft_max_projections(current_otheta)

        # Coarse search
        for da in np.linspace(-30, 30, 121):
            s = soft_max_projections(current_otheta + da)
            if s < best_val:
                best_val = s
                best_otheta = current_otheta + da

        # Fine search
        for da in np.linspace(-5, 5, 201):
            s = soft_max_projections(best_otheta + da)
            if s < best_val:
                best_val = s
                best_otheta = best_otheta + da

        # Very fine search
        for da in np.linspace(-1, 1, 81):
            s = soft_max_projections(best_otheta + da)
            if s < best_val:
                best_val = s
                best_otheta = best_otheta + da

        # Ultra fine
        for da in np.linspace(-0.1, 0.1, 41):
            s = soft_max_projections(best_otheta + da)
            if s < best_val:
                best_val = s
                best_otheta = best_otheta + da

        return best_otheta, best_val

    # Generate initial configurations
    configs = []

    # Config 1: Standard hex lattice
    c1, a1 = generate_hexagonal_cluster()
    params1 = []
    for cx, cy in c1:
        params1.append(cx)
        params1.append(cy)
    for a in a1:
        params1.append(a)
    params1.append(0.0)
    configs.append(params1)

    # Config 2: Perturbed angles
    c2, a2 = c1, [30.0 + random.uniform(-3, 3) for _ in range(12)]
    params2 = []
    for cx, cy in c2:
        params2.append(cx)
        params2.append(cy)
    for a in a2:
        params2.append(a)
    params2.append(15.0)
    configs.append(params2)

    # Config 3: Rectangular 3x4 grid
    c3, a3 = generate_rectangular_grid()
    params3 = []
    for cx, cy in c3:
        params3.append(cx)
        params3.append(cy)
    for a in a3:
        params3.append(a)
    params3.append(0.0)
    configs.append(params3)

    # Config 4: Optimized hex ring
    c4, a4 = generate_hex_ring_optimal()
    params4 = []
    for cx, cy in c4:
        params4.append(cx)
        params4.append(cy)
    for a in a4:
        params4.append(a)
    params4.append(0.0)
    configs.append(params4)

    best_params = None
    best_val = float('inf')

    for init_params in configs:
        if time.time() > deadline - 60:
            break

        # SLSQP with hard constraints
        p_slsqp, val_slsqp = slsqp_optimize(init_params, max_iter=200)
        if val_slsqp < best_val and is_valid(
            [[p_slsqp[i*2], p_slsqp[i*2+1]] for i in range(12)],
            [p_slsqp[24+i] for i in range(12)]
        ):
            best_val = val_slsqp
            best_params = p_slsqp

        # Random walk refinement
        if time.time() > deadline - 30:
            break
        p_rw, val_rw = random_walk_optimize(p_slsqp, max_iters=2000, step_init=0.08)
        if val_rw < best_val and is_valid(
            [[p_rw[i*2], p_rw[i*2+1]] for i in range(12)],
            [p_rw[24+i] for i in range(12)]
        ):
            best_val = val_rw
            best_params = p_rw

        # Nelder-Mead polish
        if time.time() > deadline - 20:
            break
        p_nm, val_nm = nelder_polish(p_rw, max_iter=400)
        if val_nm < best_val and is_valid(
            [[p_nm[i*2], p_nm[i*2+1]] for i in range(12)],
            [p_nm[24+i] for i in range(12)]
        ):
            best_val = val_nm
            best_params = p_nm

    if best_params is None:
        best_params = configs[0]
        best_val = compute_outer_side(
            [[best_params[i*2], best_params[i*2+1]] for i in range(12)],
            [best_params[24+i] for i in range(12)],
            0.0, 0.0, best_params[36]
        )

    # High-precision polish pass
    if time.time() < deadline - DEADLINE_MARGIN - 15:
        p_hp, val_hp = high_precision_polish(best_params)
        if val_hp < best_val and is_valid(
            [[p_hp[i*2], p_hp[i*2+1]] for i in range(12)],
            [p_hp[24+i] for i in range(12)]
        ):
            best_val = val_hp
            best_params = p_hp

    # Final: optimize outer angle using smooth surrogate
    centers = [[best_params[i*2], best_params[i*2+1]] for i in range(12)]
    angles = [best_params[24+i] for i in range(12)]
    best_otheta, best_side = optimize_outer_angle(centers, angles, best_params[36])

    # Verify with exact computation
    exact_side = compute_outer_side(centers, angles, 0.0, 0.0, best_otheta)
    if exact_side < best_side:
        best_side = exact_side

    inner = [[c[0], c[1], a] for c, a in zip(centers, angles)]
    outer_center = (0.0, 0.0)
    outer_side = best_side
    outer_angle = best_otheta

    return inner, outer_center, outer_side, outer_angle

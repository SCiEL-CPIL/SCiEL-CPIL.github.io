import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    deadline = time.time() + 130.0
    
    def get_hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return np.array(vertices)

    def project_range(vertices, nx, ny):
        projs = np.dot(vertices, [nx, ny])
        return np.min(projs), np.max(projs)

    def sat_min_separation(v1, v2):
        """Return minimum separation (negative if overlapping) using SAT."""
        min_sep = float('inf')
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i + 1) % 6]
            edge = p2 - p1
            nx, ny = -edge[1], edge[0]
            norm = math.sqrt(nx*nx + ny*ny)
            if norm < 1e-12:
                continue
            nx, ny = nx/norm, ny/norm
            min1, max1 = project_range(v1, nx, ny)
            min2, max2 = project_range(v2, nx, ny)
            sep = min(max1 - min2, max2 - min1)
            if sep < min_sep:
                min_sep = sep
        for i in range(6):
            p1 = v2[i]
            p2 = v2[(i + 1) % 6]
            edge = p2 - p1
            nx, ny = -edge[1], edge[0]
            norm = math.sqrt(nx*nx + ny*ny)
            if norm < 1e-12:
                continue
            nx, ny = nx/norm, ny/norm
            min1, max1 = project_range(v1, nx, ny)
            min2, max2 = project_range(v2, nx, ny)
            sep = min(max1 - min2, max2 - min1)
            if sep < min_sep:
                min_sep = sep
        return min_sep

    def compute_outer_side(centers, angles, otheta):
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
                    proj = vx * nx + vy * ny
                    if proj > max_proj:
                        max_proj = proj
                    proj_neg = -vx * nx - vy * ny
                    if proj_neg > max_proj:
                        max_proj = proj_neg
        return 2.0 * max_proj / math.sqrt(3)

    def is_valid(centers, angles, tol=1e-7):
        verts = []
        for i in range(12):
            v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            verts.append(v)
        for i in range(12):
            for j in range(i + 1, 12):
                sep = sat_min_separation(verts[i], verts[j])
                if sep < -tol:
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

    def generate_rect3x4():
        d = math.sqrt(3) + 0.01
        dx = d
        dy = d * math.sqrt(3) / 2.0
        centers = []
        angles = []
        for row in range(4):
            for col in range(3):
                cx = (col - 1) * dx
                cy = (row - 1.5) * dy
                centers.append(np.array([cx, cy]))
                angles.append(30.0)
        return centers, angles

    def generate_hex6_plus6():
        d = math.sqrt(3) + 0.01
        u = np.array([d, 0.0])
        v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])
        ring1_vectors = [u, v, v - u, -u, -v, u - v]
        offset = 2.0
        centers = []
        angles = []
        for vec in [np.array([0.0, 0.0])] + ring1_vectors:
            centers.append(vec - np.array([offset, 0.0]))
            angles.append(30.0)
        for vec in [np.array([0.0, 0.0])] + ring1_vectors:
            centers.append(vec + np.array([offset, 0.0]))
            angles.append(30.0)
        return centers, angles

    def generate_triangular_grid():
        """Triangular lattice arrangement for 12 hexagons."""
        d = math.sqrt(3) + 0.01
        h = d * math.sqrt(3) / 2.0
        centers = []
        angles = []
        # 4 rows: 3, 4, 3, 2
        rows = [3, 4, 3, 2]
        y_offset = 0.0
        for r, count in enumerate(rows):
            x_start = -(count - 1) / 2.0 * d
            for c in range(count):
                cx = x_start + c * d
                cy = y_offset
                if r % 2 == 1:
                    cx += d / 2.0
                centers.append(np.array([cx, cy]))
                angles.append(30.0)
            y_offset += h
        return centers, angles

    def smooth_max(vals, alpha=30.0):
        """Soft max approximation."""
        m = max(vals)
        if m < -100:
            return m
        return m + math.log(sum(math.exp(alpha * (v - m)) for v in vals)) / alpha

    def objective_and_penalty(params, alpha=20.0):
        """Compute outer side length + soft overlap penalty."""
        centers = []
        for i in range(12):
            centers.append([params[i * 2], params[i * 2 + 1]])
        angles = [params[24 + i] for i in range(12)]
        otheta = params[36]
        
        side = compute_outer_side(centers, angles, otheta)
        
        # Compute overlap penalties
        verts = []
        for i in range(12):
            v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            verts.append(v)
        
        penalty = 0.0
        for i in range(12):
            for j in range(i + 1, 12):
                sep = sat_min_separation(verts[i], verts[j])
                if sep < 0:
                    penalty += math.exp(alpha * sep) / alpha
        
        return side + penalty

    def nelder_polish(params, max_iter=500, alpha=20.0):
        """Nelder-Mead with smooth overlap penalty."""
        def obj(p):
            if time.time() > deadline - 5:
                return 1e9
            return objective_and_penalty(p, alpha)
        
        try:
            res = minimize(obj, params, method='Nelder-Mead',
                          options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10,
                                   'adaptive': True})
            return res.x, res.fun
        except Exception:
            return params, obj(params)

    def slsqp_optimize(initial_params, max_iter=300):
        """SLSQP with hard separation constraints."""
        def objective(params):
            if time.time() > deadline - 5:
                return 1e9
            centers = []
            for i in range(12):
                centers.append([params[i * 2], params[i * 2 + 1]])
            angles = [params[24 + i] for i in range(12)]
            otheta = params[36]
            return compute_outer_side(centers, angles, otheta)

        def constraints():
            cons = []
            for i in range(12):
                for j in range(i + 1, 12):
                    def sep_constraint(params, i=i, j=j):
                        if time.time() > deadline - 5:
                            return 1.0
                        ci = [params[i * 2], params[i * 2 + 1]]
                        cj = [params[j * 2], params[j * 2 + 1]]
                        ai = params[24 + i]
                        aj = params[24 + j]
                        vi = get_hex_vertices(ci[0], ci[1], 1.0, ai)
                        vj = get_hex_vertices(cj[0], cj[1], 1.0, aj)
                        return sat_min_separation(vi, vj) + 1e-7
                    cons.append({'type': 'ineq', 'fun': sep_constraint})
            return cons

        try:
            res = minimize(objective, initial_params, method='SLSQP',
                          constraints=constraints(),
                          options={'maxiter': max_iter, 'ftol': 1e-10, 'disp': False})
            return res.x, objective(res.x)
        except Exception:
            return initial_params, objective(initial_params)

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
    
    # Config 2: Perturbed angles on hex lattice
    c2, a2 = generate_hexagonal_cluster()
    a2p = [30.0 + random.uniform(-5, 5) for _ in range(12)]
    params2 = []
    for cx, cy in c2:
        params2.append(cx + random.uniform(-0.2, 0.2))
        params2.append(cy + random.uniform(-0.2, 0.2))
    for a in a2p:
        params2.append(a)
    params2.append(15.0)
    configs.append(params2)

    # Config 3: 3x4 rectangular grid
    c3, a3 = generate_rect3x4()
    params3 = []
    for cx, cy in c3:
        params3.append(cx)
        params3.append(cy)
    for a in a3:
        params3.append(a)
    params3.append(0.0)
    configs.append(params3)

    # Config 4: Two hex-6 clusters
    c4, a4 = generate_hex6_plus6()
    params4 = []
    for cx, cy in c4:
        params4.append(cx)
        params4.append(cy)
    for a in a4:
        params4.append(a)
    params4.append(0.0)
    configs.append(params4)

    # Config 5: Triangular grid
    c5, a5 = generate_triangular_grid()
    params5 = []
    for cx, cy in c5:
        params5.append(cx)
        params5.append(cy)
    for a in a5:
        params5.append(a)
    params5.append(0.0)
    configs.append(params5)

    # Config 6: Perturbed rect grid
    c6, a6 = generate_rect3x4()
    a6p = [30.0 + random.uniform(-8, 8) for _ in range(12)]
    params6 = []
    for cx, cy in c6:
        params6.append(cx + random.uniform(-0.3, 0.3))
        params6.append(cy + random.uniform(-0.3, 0.3))
    for a in a6p:
        params6.append(a)
    params6.append(10.0)
    configs.append(params6)

    # Config 7: Random perturbation of hex cluster
    c7, a7 = generate_hexagonal_cluster()
    params7 = []
    for cx, cy in c7:
        params7.append(cx + random.uniform(-0.5, 0.5))
        params7.append(cy + random.uniform(-0.5, 0.5))
    for a in a7:
        params7.append(30.0 + random.uniform(-15, 15))
    params7.append(random.uniform(-30, 30))
    configs.append(params7)

    random.seed(42)
    best_params = None
    best_val = float('inf')

    # Phase 1: SLSQP from each config
    for idx, init_params in enumerate(configs):
        if time.time() > deadline - 50:
            break
        p_slsqp, val_slsqp = slsqp_optimize(init_params, max_iter=200)
        centers = [[p_slsqp[i*2], p_slsqp[i*2+1]] for i in range(12)]
        angles = [p_slsqp[24+i] for i in range(12)]
        if is_valid(centers, angles):
            side = compute_outer_side(centers, angles, p_slsqp[36])
            if side < best_val:
                best_val = side
                best_params = list(p_slsqp)
    
    # Phase 2: Nelder-Mead polish on best SLSQP result
    if best_params is None:
        best_params = configs[0]
        best_val = objective_and_penalty(best_params, alpha=20.0)
    
    if time.time() < deadline - 30:
        p_nm, val_nm = nelder_polish(best_params, max_iter=600, alpha=15.0)
        centers = [[p_nm[i*2], p_nm[i*2+1]] for i in range(12)]
        angles = [p_nm[24+i] for i in range(12)]
        if is_valid(centers, angles):
            side = compute_outer_side(centers, angles, p_nm[36])
            if side < best_val:
                best_val = side
                best_params = list(p_nm)
    
    # Phase 3: Try more configs with Nelder-Mead directly
    for idx in range(1, min(4, len(configs))):
        if time.time() > deadline - 25:
            break
        init = configs[idx]
        p_nm, val_nm = nelder_polish(init, max_iter=400, alpha=15.0)
        centers = [[p_nm[i*2], p_nm[i*2+1]] for i in range(12)]
        angles = [p_nm[24+i] for i in range(12)]
        if is_valid(centers, angles):
            side = compute_outer_side(centers, angles, p_nm[36])
            if side < best_val:
                best_val = side
                best_params = list(p_nm)
    
    # Phase 4: Outer angle optimization
    centers = [[best_params[i*2], best_params[i*2+1]] for i in range(12)]
    angles = [best_params[24+i] for i in range(12)]
    best_otheta = best_params[36]
    
    for da in np.linspace(-20, 20, 41):
        s = compute_outer_side(centers, angles, best_otheta + da)
        if s < best_val:
            best_val = s
            best_otheta = best_otheta + da
    
    # Phase 5: Fine angle refinement
    for da in np.linspace(-5, 5, 101):
        s = compute_outer_side(centers, angles, best_otheta + da)
        if s < best_val:
            best_val = s
            best_otheta = best_otheta + da
    
    # Phase 6: Final Nelder-Mead polish if time remains
    if time.time() < deadline - 15:
        final_params = list(best_params)
        final_params[36] = best_otheta
        p_final, val_final = nelder_polish(final_params, max_iter=400, alpha=25.0)
        centers_f = [[p_final[i*2], p_final[i*2+1]] for i in range(12)]
        angles_f = [p_final[24+i] for i in range(12)]
        if is_valid(centers_f, angles_f):
            side_f = compute_outer_side(centers_f, angles_f, p_final[36])
            if side_f < best_val:
                best_val = side_f
                best_params = list(p_final)
                centers = centers_f
                angles = angles_f
                best_otheta = p_final[36]
                # Final angle refinement
                for da in np.linspace(-5, 5, 101):
                    s = compute_outer_side(centers, angles, best_otheta + da)
                    if s < best_val:
                        best_val = s
                        best_otheta = best_otheta + da

    centers = [[best_params[i*2], best_params[i*2+1]] for i in range(12)]
    angles = [best_params[24+i] for i in range(12)]
    
    inner = [[c[0], c[1], a] for c, a in zip(centers, angles)]
    outer_center = (0.0, 0.0)
    outer_side = best_val
    outer_angle = best_otheta
    
    return inner, outer_center, outer_side, outer_angle

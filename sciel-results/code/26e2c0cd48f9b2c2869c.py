import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE_MARGIN = 10.0

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    
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

    def sat_separated(v1, v2, eps=1e-9):
        """Check if two hexagons are separated using SAT."""
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i + 1) % 6]
            edge = p2 - p1
            nx, ny = -edge[1], edge[0]
            min1, max1 = project_range(v1, nx, ny)
            min2, max2 = project_range(v2, nx, ny)
            if max1 < min2 + eps or max2 < min1 + eps:
                return True
        for i in range(6):
            p1 = v2[i]
            p2 = v2[(i + 1) % 6]
            edge = p2 - p1
            nx, ny = -edge[1], edge[0]
            min1, max1 = project_range(v1, nx, ny)
            min2, max2 = project_range(v2, nx, ny)
            if max1 < min2 + eps or max2 < min1 + eps:
                return True
        return False

    def compute_outer_side(centers, angles, ocx, ocy, otheta):
        """Compute minimal outer hexagon side length containing all inner hexagons.
        
        For a regular hexagon with side length s and inradius r = s*sqrt(3)/2,
        the apothem (distance from center to each edge) is r.
        So s = 2 * max_proj / sqrt(3) where max_proj is the max projection
        onto the 6 edge normals (apothem direction).
        """
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
        # apothem = s * sqrt(3) / 2, so s = 2 * apothem / sqrt(3)
        return 2.0 * max_proj / math.sqrt(3)

    def is_valid(centers, angles):
        verts = []
        for i in range(12):
            v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            verts.append(v)
        for i in range(12):
            for j in range(i + 1, 12):
                if not sat_separated(verts[i], verts[j]):
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
        """3x4 rectangular arrangement of hexagons with proper spacing."""
        dx = math.sqrt(3) + 0.01
        dy = 3.0 + 0.01  # height of hex = sqrt(3), so 2*height = 2*sqrt(3) ≈ 3.464
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
        """Optimized hexagonal ring: center + 6 ring1 + 5 from ring2, tuned."""
        d = math.sqrt(3) + 0.005
        u = np.array([d, 0.0])
        v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])
        centers = [np.array([0.0, 0.0])]
        ring1_vectors = [u, v, v - u, -u, -v, u - v]
        for vec in ring1_vectors:
            centers.append(vec.copy())
        # Add 5 more from ring2, choosing positions that minimize bounding hex
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

    def slsqp_optimize(initial_params, max_iter=500):
        """
        Use SLSQP with SAT-based non-overlap constraints.
        Parameters: 24 (centers) + 12 (angles) + 1 (outer angle) = 37
        We fix outer center at (0,0) by centering the configuration.
        """
        def objective(params):
            if time.time() > deadline - DEADLINE_MARGIN:
                return 1e9
            centers = []
            for i in range(12):
                centers.append([params[i * 2], params[i * 2 + 1]])
            angles = [params[24 + i] for i in range(12)]
            otheta = params[36]
            side = compute_outer_side(centers, angles, 0.0, 0.0, otheta)
            return side

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
                        # Compute minimum separation over all SAT axes
                        min_sep = float('inf')
                        for k in range(6):
                            p1 = vi[k]
                            p2 = vi[(k + 1) % 6]
                            edge = p2 - p1
                            nx, ny = -edge[1], edge[0]
                            min1, max1 = project_range(vi, nx, ny)
                            min2, max2 = project_range(vj, nx, ny)
                            sep = min(max1 - min2, max2 - min1)
                            if sep < min_sep:
                                min_sep = sep
                        for k in range(6):
                            p1 = vj[k]
                            p2 = vj[(k + 1) % 6]
                            edge = p2 - p1
                            nx, ny = -edge[1], edge[0]
                            min1, max1 = project_range(vi, nx, ny)
                            min2, max2 = project_range(vj, nx, ny)
                            sep = min(max1 - min2, max2 - min1)
                            if sep < min_sep:
                                min_sep = sep
                        return min_sep - 1e-6
                    cons.append({'type': 'ineq', 'fun': sep_constraint})
            return cons

        try:
            res = minimize(objective, initial_params, method='SLSQP',
                          constraints=constraints(),
                          options={'maxiter': max_iter, 'ftol': 1e-10, 'disp': False})
            return res.x, res.fun
        except Exception:
            return initial_params, objective(initial_params)

    def nelder_polish(params, max_iter=1000):
        """Nelder-Mead polish with soft overlap penalty."""
        def obj(params):
            if time.time() > deadline - DEADLINE_MARGIN:
                return 1e9
            centers = []
            for i in range(12):
                centers.append([params[i * 2], params[i * 2 + 1]])
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
                    min_sep = float('inf')
                    vi, vj = verts[i], verts[j]
                    for k in range(6):
                        p1 = vi[k]
                        p2 = vi[(k + 1) % 6]
                        edge = p2 - p1
                        nx, ny = -edge[1], edge[0]
                        min1, max1 = project_range(vi, nx, ny)
                        min2, max2 = project_range(vj, nx, ny)
                        sep = min(max1 - min2, max2 - min1)
                        if sep < min_sep:
                            min_sep = sep
                    for k in range(6):
                        p1 = vj[k]
                        p2 = vj[(k + 1) % 6]
                        edge = p2 - p1
                        nx, ny = -edge[1], edge[0]
                        min1, max1 = project_range(vi, nx, ny)
                        min2, max2 = project_range(vj, nx, ny)
                        sep = min(max1 - min2, max2 - min1)
                        if sep < min_sep:
                            min_sep = sep
                    if min_sep < 0:
                        penalty += min_sep ** 2
            return side + 100.0 * penalty

        try:
            res = minimize(obj, params, method='Nelder-Mead',
                          options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10})
            return res.x, res.fun
        except Exception:
            return params, obj(params)

    def high_precision_polish(params):
        """Final tight polish with very small step sizes and higher penalty."""
        def obj(params):
            if time.time() > deadline - DEADLINE_MARGIN:
                return 1e9
            centers = []
            for i in range(12):
                centers.append([params[i * 2], params[i * 2 + 1]])
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
                    min_sep = float('inf')
                    vi, vj = verts[i], verts[j]
                    for k in range(6):
                        p1 = vi[k]
                        p2 = vi[(k + 1) % 6]
                        edge = p2 - p1
                        nx, ny = -edge[1], edge[0]
                        min1, max1 = project_range(vi, nx, ny)
                        min2, max2 = project_range(vj, nx, ny)
                        sep = min(max1 - min2, max2 - min1)
                        if sep < min_sep:
                            min_sep = sep
                    for k in range(6):
                        p1 = vj[k]
                        p2 = vj[(k + 1) % 6]
                        edge = p2 - p1
                        nx, ny = -edge[1], edge[0]
                        min1, max1 = project_range(vi, nx, ny)
                        min2, max2 = project_range(vj, nx, ny)
                        sep = min(max1 - min2, max2 - min1)
                        if sep < min_sep:
                            min_sep = sep
                    if min_sep < 0:
                        penalty += min_sep ** 2
            return side + 500.0 * penalty

        try:
            res = minimize(obj, params, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12})
            return res.x, res.fun
        except Exception:
            return params, obj(params)

    def optimize_outer_angle(centers, angles, current_otheta):
        """Optimize the outer hexagon rotation angle for minimal side length.
        
        Uses a smooth surrogate: for each of the 6 normals, compute the max projection,
        then use a soft-max to combine them into a smooth objective.
        """
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
            # Soft max with temperature
            T = 0.01
            max_m = max(max_projs)
            exps = [math.exp((mp - max_m) / T) for mp in max_projs]
            s = sum(exps)
            soft_max = max_m + T * math.log(s)
            return 2.0 * soft_max / math.sqrt(3)

        # Golden section search on outer angle
        best_otheta = current_otheta
        best_val = soft_max_projections(current_otheta)
        
        # Coarse search
        for da in np.linspace(-30, 30, 61):
            s = soft_max_projections(current_otheta + da)
            if s < best_val:
                best_val = s
                best_otheta = current_otheta + da

        # Fine search
        for da in np.linspace(-5, 5, 101):
            s = soft_max_projections(best_otheta + da)
            if s < best_val:
                best_val = s
                best_otheta = best_otheta + da

        # Very fine search
        for da in np.linspace(-1, 1, 41):
            s = soft_max_projections(best_otheta + da)
            if s < best_val:
                best_val = s
                best_otheta = best_otheta + da

        # Ultra fine
        for da in np.linspace(-0.1, 0.1, 21):
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
    params1.append(0.0)  # outer angle
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

    random.seed(42)
    best_params = None
    best_val = float('inf')

    for init_params in configs:
        if time.time() > deadline - 50:
            break
        
        # SLSQP with hard constraints
        p_slsqp, val_slsqp = slsqp_optimize(init_params, max_iter=200)
        if val_slsqp < best_val and is_valid(
            [[p_slsqp[i*2], p_slsqp[i*2+1]] for i in range(12)],
            [p_slsqp[24+i] for i in range(12)]
        ):
            best_val = val_slsqp
            best_params = p_slsqp
        
        # Nelder-Mead polish
        if time.time() > deadline - 25:
            break
        p_nm, val_nm = nelder_polish(p_slsqp, max_iter=400)
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
    if time.time() < deadline - DEADLINE_MARGIN - 10:
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
# EVOLVE-BLOCK-END

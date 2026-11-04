import math
import time
import random
import numpy as np
from scipy.optimize import minimize, differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin

    def get_hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return np.array(vertices)

    def compute_outer_side(centers, angles, otheta):
        """Compute the minimum side length of a regular hexagon (centered at origin, rotated by otheta) containing all inner hexagons."""
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

    def hex_separation(c1, a1, c2, a2):
        """Compute minimum separation between two unit hexagons. Positive = separated."""
        # For two regular hexagons of side 1, the minimum center distance for
        # non-overlap depends on relative orientation. Use SAT for accuracy.
        v1 = get_hex_vertices(c1[0], c1[1], 1.0, a1)
        v2 = get_hex_vertices(c2[0], c2[1], 1.0, a2)
        min_sep = float('inf')
        for verts in [v1, v2]:
            for i in range(6):
                p1 = verts[i]
                p2 = verts[(i + 1) % 6]
                edge = p2 - p1
                nx, ny = -edge[1], edge[0]
                proj1 = np.dot(v1, [nx, ny])
                proj2 = np.dot(v2, [nx, ny])
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                sep = min(max1 - min2, max2 - min1)
                if sep < min_sep:
                    min_sep = sep
                if min_sep <= 0:
                    return min_sep
        return min_sep

    def is_valid(centers, angles, tol=-1e-9):
        """Check if all inner hexagons are disjoint."""
        n = len(centers)
        for i in range(n):
            for j in range(i + 1, n):
                if hex_separation(centers[i], angles[i], centers[j], angles[j]) < tol:
                    return False
        return True

    def params_from_config(centers, angles, otheta=0.0):
        params = []
        for cx, cy in centers:
            params.append(cx)
            params.append(cy)
        for a in angles:
            params.append(a)
        params.append(otheta)
        return np.array(params)

    def generate_hexagonal_lattice(rot_deg):
        """Generate a hexagonal cluster of 12 hexagons with lattice rotated by rot_deg."""
        d = math.sqrt(3) + 0.005  # minimum center distance for non-overlapping unit hexagons
        rot = math.radians(rot_deg)
        cos_r, sin_r = math.cos(rot), math.sin(rot)
        
        u = np.array([d, 0.0])
        v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])
        
        # Rotate basis vectors
        def rot_vec(vec):
            return np.array([vec[0]*cos_r - vec[1]*sin_r, vec[0]*sin_r + vec[1]*cos_r])
        
        u_r = rot_vec(u)
        v_r = rot_vec(v)
        
        basis = [u_r, v_r, v_r - u_r, -u_r, -v_r, u_r - v_r]
        ring1_vectors = [u_r, v_r, v_r - u_r, -u_r, -v_r, u_r - v_r]
        
        centers = [np.array([0.0, 0.0])]
        for vec in ring1_vectors:
            centers.append(vec.copy())
        
        # Generate ring 2 candidates
        ring2_candidates = []
        for i in range(6):
            p1 = ring1_vectors[i]
            for b in basis:
                p2 = p1 + b
                is_r0 = np.allclose(p2, np.array([0.0, 0.0]), atol=0.01)
                is_r1 = False
                for p1j in ring1_vectors:
                    if np.allclose(p2, p1j, atol=0.01):
                        is_r1 = True
                        break
                if not is_r0 and not is_r1:
                    duplicate = False
                    for pc in ring2_candidates:
                        if np.allclose(p2, pc, atol=0.01):
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
        """Generate a 3x4 rectangular grid of hexagons."""
        dx = math.sqrt(3) + 0.005
        dy = 3.0 + 0.005
        centers = []
        angles = []
        for row in range(3):
            for col in range(4):
                cx = (col - 1.5) * dx
                cy = (row - 1.0) * dy
                centers.append([cx, cy])
                angles.append(30.0)
        return centers, angles

    def slsqp_optimize(centers, angles, max_iters=80):
        """Use SLSQP with separation constraints to optimize the configuration."""
        n = len(centers)
        
        # Decision variables: x1,y1, a1, ..., xn,yn, an, otheta
        x0 = []
        for i in range(n):
            x0.extend([centers[i][0], centers[i][1], angles[i]])
        x0.append(0.0)  # otheta
        
        def objective(x):
            if time.time() > deadline - 20:
                return 1e10
            c = [[x[3*i], x[3*i+1]] for i in range(n)]
            a = [x[3*i+2] for i in range(n)]
            otheta = x[3*n]
            side = compute_outer_side(c, a, otheta)
            return side
        
        # Constraints: pairwise separation >= 0 (use small negative for feasibility)
        min_sep_req = -0.001
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                def sep_constraint(x, i=i, j=j):
                    if time.time() > deadline - 20:
                        return 1.0
                    ci = (x[3*i], x[3*i+1])
                    ai = x[3*i+2]
                    cj = (x[3*j], x[3*j+1])
                    aj = x[3*j+2]
                    return hex_separation(ci, ai, cj, aj) - min_sep_req
                constraints.append({'type': 'ineq', 'fun': sep_constraint})
        
        # Bounds
        bounds = []
        for i in range(n):
            bounds.extend([(-5, 5), (-5, 5), (0, 60)])
        bounds.extend([(-30, 30)])
        
        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                             constraints=constraints, options={'maxiter': max_iters, 'ftol': 1e-10})
            if result.fun < 1e9:
                c = [[result.x[3*i], result.x[3*i+1]] for i in range(n)]
                a = [result.x[3*i+2] for i in range(n)]
                otheta = result.x[3*n]
                if is_valid(c, a):
                    side = compute_outer_side(c, a, otheta)
                    return c, a, otheta, side
        except Exception:
            pass
        return None

    def penalty_objective(params):
        """Penalty-based objective for Nelder-Mead."""
        if time.time() > deadline - 20:
            return 1e9
        centers = [[params[i*2], params[i*2+1]] for i in range(12)]
        angles = [params[24+i] for i in range(12)]
        otheta = params[36]
        side = compute_outer_side(centers, angles, otheta)
        
        # Compute overlap penalty
        pen = 0.0
        for i in range(12):
            for j in range(i+1, 12):
                sep = hex_separation(centers[i], angles[i], centers[j], angles[j])
                if sep < 0:
                    pen += sep * sep
        return side + 200.0 * pen

    def polish_outer_angle(centers, angles, otheta, time_budget=5.0):
        """High-precision polish of outer hexagon angle."""
        best_side = compute_outer_side(centers, angles, otheta)
        best_otheta = otheta
        
        # Multi-scale search
        scales = [10.0, 1.0, 0.1, 0.01, 0.001]
        for scale in scales:
            if time.time() > start_time + 150.0 - time_budget:
                break
            improved = True
            while improved:
                improved = False
                for da in np.linspace(-scale, scale, 21):
                    s = compute_outer_side(centers, angles, best_otheta + da)
                    if s < best_side - 1e-12:
                        best_side = s
                        best_otheta = best_otheta + da
                        improved = True
        return best_otheta, best_side

    # Generate diverse initial configurations
    configs = []
    
    # Try multiple lattice rotations
    for rot_deg in [0, 5, 10, 15, 30]:
        c, a = generate_hexagonal_lattice(rot_deg)
        configs.append(params_from_config(c, a))
    
    # Rectangular grid
    c, a = generate_rectangular_grid()
    configs.append(params_from_config(c, a))

    best_params = None
    best_val = float('inf')

    # Phase 1: SLSQP from each initial config (most promising)
    for init_params in configs[:5]:
        if time.time() > deadline - 30:
            break
        
        centers = [[init_params[i*2], init_params[i*2+1]] for i in range(12)]
        angles = [init_params[24+i] for i in range(12)]
        
        # Check initial validity
        if is_valid(centers, angles):
            val = compute_outer_side(centers, angles, init_params[36])
            if val < best_val:
                best_val = val
                best_params = init_params.copy()
        
        # Try SLSQP
        slsqp_result = slsqp_optimize(centers, angles, max_iters=60)
        if slsqp_result is not None:
            c, a, otheta, side = slsqp_result
            if side < best_val:
                best_val = side
                best_params = params_from_config(c, a, otheta)

    # Phase 2: Nelder-Mead with penalty from best config
    if best_params is not None and time.time() < deadline - 25:
        try:
            res = minimize(penalty_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-9})
            centers = [[res.x[i*2], res.x[i*2+1]] for i in range(12)]
            angles = [res.x[24+i] for i in range(12)]
            if is_valid(centers, angles):
                val = compute_outer_side(centers, angles, res.x[36])
                if val < best_val:
                    best_val = val
                    best_params = res.x.copy()
        except Exception:
            pass

    # Phase 3: Try remaining configs with Nelder-Mead
    if time.time() < deadline - 20:
        for init_params in configs[5:]:
            if time.time() > deadline - 15:
                break
            try:
                res = minimize(penalty_objective, init_params, method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9})
                centers = [[res.x[i*2], res.x[i*2+1]] for i in range(12)]
                angles = [res.x[24+i] for i in range(12)]
                if is_valid(centers, angles):
                    val = compute_outer_side(centers, angles, res.x[36])
                    if val < best_val:
                        best_val = val
                        best_params = res.x.copy()
            except Exception:
                pass

    # Phase 4: Differential evolution for global search
    if time.time() < deadline - 35:
        try:
            bounds = [(-5, 5)] * 24 + [(0, 60)] * 12 + [(-30, 30)]
            de_res = differential_evolution(penalty_objective, bounds, maxiter=40, 
                                            popsize=12, tol=1e-5, 
                                            seed=42, polish=False,
                                            workers=-1)
            if de_res.fun < 1e9:
                centers = [[de_res.x[i*2], de_res.x[i*2+1]] for i in range(12)]
                angles = [de_res.x[24+i] for i in range(12)]
                if is_valid(centers, angles):
                    val = compute_outer_side(centers, angles, de_res.x[36])
                    if val < best_val:
                        best_val = val
                        best_params = de_res.x.copy()
        except Exception:
            pass

    # Fallback
    if best_params is None:
        best_params = configs[0].copy()
        best_val = compute_outer_side(
            [[best_params[i*2], best_params[i*2+1]] for i in range(12)],
            [best_params[24+i] for i in range(12)],
            best_params[36]
        )

    # Phase 5: SLSQP polish from best
    if time.time() < deadline - 15:
        centers = [[best_params[i*2], best_params[i*2+1]] for i in range(12)]
        angles = [best_params[24+i] for i in range(12)]
        slsqp_result = slsqp_optimize(centers, angles, max_iters=100)
        if slsqp_result is not None:
            c, a, otheta, side = slsqp_result
            if side < best_val:
                best_val = side
                best_params = params_from_config(c, a, otheta)

    # Phase 6: Final Nelder-Mead polish
    if time.time() < deadline - 10:
        try:
            polish_res = minimize(penalty_objective, best_params, method='Nelder-Mead',
                                 options={'maxiter': 1500, 'xatol': 1e-9, 'fatol': 1e-11})
            centers = [[polish_res.x[i*2], polish_res.x[i*2+1]] for i in range(12)]
            angles = [polish_res.x[24+i] for i in range(12)]
            if is_valid(centers, angles):
                val = compute_outer_side(centers, angles, polish_res.x[36])
                if val < best_val:
                    best_val = val
                    best_params = polish_res.x.copy()
        except Exception:
            pass

    # Phase 7: High-precision outer angle polish
    centers = [[best_params[i*2], best_params[i*2+1]] for i in range(12)]
    angles = [best_params[24+i] for i in range(12)]
    best_otheta, best_side = polish_outer_angle(centers, angles, best_params[36], time_budget=5.0)

    inner = [[c[0], c[1], a] for c, a in zip(centers, angles)]
    outer_center = (0.0, 0.0)
    outer_side = best_side
    outer_angle = best_otheta
    
    return inner, outer_center, outer_side, outer_angle

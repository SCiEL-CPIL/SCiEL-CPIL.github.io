import math
from scipy.optimize import minimize
import numpy as np
import random
import time
def construct():
    deadline = time.time() + 140.0  # 10s margin for 150s limit

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

    def min_separation(v1, v2):
        min_sep = float('inf')
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i + 1) % 6]
            edge = p2 - p1
            nx, ny = -edge[1], edge[0]
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
            min1, max1 = project_range(v1, nx, ny)
            min2, max2 = project_range(v2, nx, ny)
            sep = min(max1 - min2, max2 - min1)
            if sep < min_sep:
                min_sep = sep
        return min_sep

    def is_valid(centers, angles):
        verts = []
        for i in range(12):
            v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            verts.append(v)
        for i in range(12):
            for j in range(i + 1, 12):
                if min_separation(verts[i], verts[j]) < -1e-9:
                    return False
        return True

    def compute_outer_side(centers, angles, ocx, ocy, otheta):
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

    def get_penalty(centers, angles):
        verts = []
        for i in range(12):
            v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            verts.append(v)
        penalty = 0.0
        for i in range(12):
            for j in range(i + 1, 12):
                sep = min_separation(verts[i], verts[j])
                if sep < 0:
                    penalty += sep * sep
        return penalty

    def objective(params):
        if time.time() > deadline:
            return 1e9
        centers = []
        for i in range(12):
            centers.append([params[i * 2], params[i * 2 + 1]])
        angles = [params[24 + i] for i in range(12)]
        otheta = params[36]
        
        side = compute_outer_side(centers, angles, 0.0, 0.0, otheta)
        pen = get_penalty(centers, angles)
        return side + 100.0 * pen

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

    def generate_circular_cluster():
        # Place hexagons in a more compact circular arrangement
        centers = []
        angles = []
        # Center
        centers.append([0.0, 0.0])
        angles.append(30.0)
        # Ring 1: 6 hexagons
        d1 = math.sqrt(3) + 0.01
        for i in range(6):
            angle = i * math.pi / 3
            cx = d1 * math.cos(angle)
            cy = d1 * math.sin(angle)
            centers.append([cx, cy])
            angles.append(30.0)
        # Ring 2: 5 more hexagons (partial ring)
        d2 = 2 * (math.sqrt(3) + 0.01)
        for i in range(5):
            angle = i * 2 * math.pi / 5 + math.pi / 10
            cx = d2 * math.cos(angle)
            cy = d2 * math.sin(angle)
            centers.append([cx, cy])
            angles.append(30.0)
        return centers, angles

    # Generate initial configurations
    configs = []
    
    c1, a1 = generate_hexagonal_cluster()
    params1 = []
    for cx, cy in c1:
        params1.append(cx)
        params1.append(cy)
    for a in a1:
        params1.append(a)
    params1.append(0.0)
    configs.append(params1)
    
    c2, a2 = generate_rectangular_grid()
    params2 = []
    for cx, cy in c2:
        params2.append(cx)
        params2.append(cy)
    for a in a2:
        params2.append(a)
    params2.append(0.0)
    configs.append(params2)
    
    c3, a3 = generate_circular_cluster()
    params3 = []
    for cx, cy in c3:
        params3.append(cx)
        params3.append(cy)
    for a in a3:
        params3.append(a)
    params3.append(0.0)
    configs.append(params3)

    best_params = None
    best_val = float('inf')

    random.seed(42)
    
    for init_params in configs:
        if time.time() > deadline - 20:
            break
        
        # Try to optimize with Nelder-Mead first
        try:
            res = minimize(objective, init_params, method='Nelder-Mead',
                          options={'maxiter': 800, 'xatol': 1e-6, 'fatol': 1e-8})
            
            # Check if result is valid
            centers = [[res.x[i*2], res.x[i*2+1]] for i in range(12)]
            angles = [res.x[24+i] for i in range(12)]
            
            if is_valid(centers, angles):
                val = compute_outer_side(centers, angles, 0.0, 0.0, res.x[36])
                if val < best_val:
                    best_val = val
                    best_params = res.x.copy()
            else:
                # If invalid, try the initial params if they are valid
                if is_valid([[init_params[i*2], init_params[i*2+1]] for i in range(12)], [init_params[24+i] for i in range(12)]):
                    val = compute_outer_side([[init_params[i*2], init_params[i*2+1]] for i in range(12)], [init_params[24+i] for i in range(12)], 0.0, 0.0, init_params[36])
                    if val < best_val:
                        best_val = val
                        best_params = np.array(init_params)
        except Exception:
            pass

    if best_params is None:
        # Fallback to first config
        best_params = np.array(configs[0])
        best_val = compute_outer_side(
            [[best_params[i*2], best_params[i*2+1]] for i in range(12)],
            [best_params[24+i] for i in range(12)],
            0.0, 0.0, best_params[36]
        )

    # Final: fine grid search on outer angle
    centers = [[best_params[i*2], best_params[i*2+1]] for i in range(12)]
    angles = [best_params[24+i] for i in range(12)]
    best_side = best_val
    best_otheta = best_params[36]
    
    # Coarse search
    for da in np.linspace(-30, 30, 121):
        s = compute_outer_side(centers, angles, 0.0, 0.0, best_otheta + da)
        if s < best_side:
            best_side = s
            best_otheta = best_otheta + da

    # Medium search
    for da in np.linspace(-5, 5, 201):
        s = compute_outer_side(centers, angles, 0.0, 0.0, best_otheta + da)
        if s < best_side:
            best_side = s
            best_otheta = best_otheta + da

    # Fine search
    for da in np.linspace(-1, 1, 401):
        s = compute_outer_side(centers, angles, 0.0, 0.0, best_otheta + da)
        if s < best_side:
            best_side = s
            best_otheta = best_otheta + da

    # Very fine search
    for da in np.linspace(-0.1, 0.1, 401):
        s = compute_outer_side(centers, angles, 0.0, 0.0, best_otheta + da)
        if s < best_side:
            best_side = s
            best_otheta = best_otheta + da

    # High-precision polish pass: use BFGS to refine positions and angles
    if time.time() < deadline - 10:
        try:
            def polish_objective(params):
                if time.time() > deadline:
                    return 1e9
                centers_p = []
                for i in range(12):
                    centers_p.append([params[i * 2], params[i * 2 + 1]])
                angles_p = [params[24 + i] for i in range(12)]
                otheta_p = params[36]
                
                side_p = compute_outer_side(centers_p, angles_p, 0.0, 0.0, otheta_p)
                pen_p = get_penalty(centers_p, angles_p)
                return side_p + 100.0 * pen_p

            polish_res = minimize(polish_objective, best_params, method='BFGS',
                                options={'maxiter': 200, 'gtol': 1e-10})
            
            centers_polish = [[polish_res.x[i*2], polish_res.x[i*2+1]] for i in range(12)]
            angles_polish = [polish_res.x[24+i] for i in range(12)]
            
            if is_valid(centers_polish, angles_polish):
                val_polish = compute_outer_side(centers_polish, angles_polish, 0.0, 0.0, polish_res.x[36])
                if val_polish < best_side:
                    best_side = val_polish
                    best_otheta = polish_res.x[36]
                    centers = centers_polish
                    angles = angles_polish
        except Exception:
            pass

    inner = [[c[0], c[1], a] for c, a in zip(centers, angles)]
    outer_center = (0.0, 0.0)
    outer_side = best_side
    outer_angle = best_otheta
    
    return inner, outer_center, outer_side, outer_angle

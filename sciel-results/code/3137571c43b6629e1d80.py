import math
import time
import random
import numpy as np
from scipy.optimize import minimize, differential_evolution, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin for 150s limit

    def get_hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return np.array(vertices)

    def sat_separated(v1, v2):
        """Check if two convex polygons are separated using SAT. Returns min separation (positive = separated)."""
        min_sep = float('inf')
        normals = []
        for verts in [v1, v2]:
            for i in range(6):
                p1 = verts[i]
                p2 = verts[(i + 1) % 6]
                edge = p2 - p1
                nx, ny = -edge[1], edge[0]
                norm = math.sqrt(nx*nx + ny*ny)
                if norm > 1e-12:
                    nx /= norm
                    ny /= norm
                normals.append((nx, ny))
        
        for nx, ny in normals:
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

    def compute_outer_side(centers, angles, ocx, ocy, otheta):
        """Compute the minimum side length of a regular hexagon (centered at ocx,ocy, rotated by otheta) containing all inner hexagons."""
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
                    proj = (vx - ocx) * nx + (vy - ocy) * ny
                    if proj > max_proj:
                        max_proj = proj
        return 2.0 * max_proj / math.sqrt(3)

    def get_penalty(centers, angles):
        """Compute overlap penalty between all pairs of inner hexagons."""
        verts = []
        for i in range(len(centers)):
            v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            verts.append(v)
        penalty = 0.0
        n = len(centers)
        for i in range(n):
            for j in range(i + 1, n):
                sep = sat_separated(verts[i], verts[j])
                if sep < 0:
                    penalty += sep * sep
        return penalty

    def is_valid(centers, angles):
        """Check if all inner hexagons are disjoint."""
        verts = []
        for i in range(len(centers)):
            v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            verts.append(v)
        n = len(centers)
        for i in range(n):
            for j in range(i + 1, n):
                if sat_separated(verts[i], verts[j]) < -1e-9:
                    return False
        return True

    def check_disjoint_simple(centers, angles, min_dist=math.sqrt(3) - 0.0001):
        """Quick check using center distances (approximate, for initial configs)."""
        n = len(centers)
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < min_dist:
                    return False
        return True

    def objective(params, ocx=0.0, ocy=0.0):
        if time.time() > deadline:
            return 1e9
        n = 12
        centers = [[params[i * 2], params[i * 2 + 1]] for i in range(n)]
        angles = [params[24 + i] for i in range(n)]
        otheta = params[36]
        side = compute_outer_side(centers, angles, ocx, ocy, otheta)
        pen = get_penalty(centers, angles)
        return side + 500.0 * pen

    def generate_hexagonal_cluster():
        """Generate a hexagonal cluster of 12 hexagons."""
        d = math.sqrt(3) + 0.01
        u = np.array([d, 0.0])
        v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])
        centers = [np.array([0.0, 0.0])]
        ring1_vectors = [u, v, v - u, -u, -v, u - v]
        for vec in ring1_vectors:
            centers.append(vec.copy())
        basis = [u, v, v - u, -u, -v, u - v]
        ring2_candidates = []
        for i in range(6):
            p1 = ring1_vectors[i]
            for b in basis:
                p2 = p1 + b
                is_r0 = np.allclose(p2, np.array([0.0, 0.0]), atol=1e-6)
                is_r1 = False
                for p1j in ring1_vectors:
                    if np.allclose(p2, p1j, atol=1e-6):
                        is_r1 = True
                        break
                if not is_r0 and not is_r1:
                    duplicate = False
                    for pc in ring2_candidates:
                        if np.allclose(p2, pc, atol=1e-6):
                            duplicate = True
                            break
                    if not duplicate:
                        ring2_candidates.append(p2)
        ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
        for i in range(5):
            centers.append(ring2_candidates[i].copy())
        angles = [30.0] * 12
        return [[c[0], c[1]] for c in centers], angles

    def generate_rectangular_grid():
        """Generate a 3x4 rectangular grid of hexagons."""
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
        """Generate two concentric rings of 6 hexagons each."""
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

    def slsqp_optimize(centers, angles, max_iters=200):
        """SLSQP with distance constraints for tighter packing."""
        n = len(centers)
        x0 = []
        for i in range(n):
            x0.extend([centers[i][0], centers[i][1], angles[i]])
        x0.extend([0.0, 0.0, 0.0])  # outer center x, y, angle
        
        def obj(x):
            if time.time() > deadline - 20:
                return 1e10
            c = [[x[3*i], x[3*i+1]] for i in range(n)]
            a = [x[3*i+2] for i in range(n)]
            ocx, ocy, otheta = x[3*n], x[3*n+1], x[3*n+2]
            side = compute_outer_side(c, a, ocx, ocy, otheta)
            # Add penalty for overlaps
            pen = get_penalty(c, a)
            return side + 1000.0 * pen
        
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                def dist_con(x, i=i, j=j):
                    xi = x[3*i]
                    yi = x[3*i+1]
                    xj = x[3*j]
                    yj = x[3*j+1]
                    return (xi - xj)**2 + (yi - yj)**2 - (math.sqrt(3) - 0.001)**2
                constraints.append({'type': 'ineq', 'fun': dist_con})
        
        bounds = []
        for i in range(n):
            bounds.extend([(-5, 5), (-5, 5), (0, 60)])
        bounds.extend([(-2, 2), (-2, 2), (0, 60)])
        
        try:
            result = minimize(obj, x0, method='SLSQP', bounds=bounds,
                             constraints=constraints, 
                             options={'maxiter': max_iters, 'ftol': 1e-12})
            if result.fun < 1e8:
                c = [[result.x[3*i], result.x[3*i+1]] for i in range(n)]
                a = [result.x[3*i+2] for i in range(n)]
                ocx, ocy, otheta = result.x[3*n], result.x[3*n+1], result.x[3*n+2]
                if is_valid(c, a):
                    side = compute_outer_side(c, a, ocx, ocy, otheta)
                    return c, a, ocx, ocy, otheta, side
        except Exception:
            pass
        return None

    def polish_outer(centers, angles, ocx, ocy, otheta):
        """High-precision polish of outer hexagon parameters."""
        best_side = compute_outer_side(centers, angles, ocx, ocy, otheta)
        best_ocx, best_ocy, best_otheta = ocx, ocy, otheta
        
        scales = [1.0, 0.1, 0.01, 0.001, 0.0001]
        for scale in scales:
            if time.time() > deadline - 10:
                break
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                for dx in [-scale, 0, scale]:
                    for dy in [-scale, 0, scale]:
                        for da in [-scale*10, 0, scale*10]:
                            otheta_new = (best_otheta + da) % 60
                            side = compute_outer_side(centers, angles, best_ocx + dx, best_ocy + dy, otheta_new)
                            if side < best_side - 1e-14:
                                best_side = side
                                best_ocx += dx
                                best_ocy += dy
                                best_otheta = otheta_new
                                improved = True
        return best_ocx, best_ocy, best_otheta, best_side

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
    best_centers = None
    best_angles = None
    best_ocx = 0.0
    best_ocy = 0.0
    best_otheta = 0.0

    # Phase 1: Nelder-Mead from each initial config
    for init_params in configs:
        if time.time() > deadline - 30:
            break
        try:
            res = minimize(objective, init_params, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            centers = [[res.x[i*2], res.x[i*2+1]] for i in range(12)]
            angles = [res.x[24+i] for i in range(12)]
            if is_valid(centers, angles):
                val = compute_outer_side(centers, angles, 0.0, 0.0, res.x[36])
                if val < best_val:
                    best_val = val
                    best_params = res.x.copy()
                    best_centers = centers
                    best_angles = angles
                    best_otheta = res.x[36]
        except Exception:
            pass

    # Phase 2: SLSQP optimization from best NM result
    if time.time() < deadline - 40 and best_centers is not None:
        try:
            slsqp_res = slsqp_optimize(best_centers, best_angles, max_iters=100)
            if slsqp_res is not None:
                c, a, ocx, ocy, otheta, side = slsqp_res
                if side < best_val:
                    best_val = side
                    best_centers = c
                    best_angles = a
                    best_ocx = ocx
                    best_ocy = ocy
                    best_otheta = otheta
                    best_params = params_from_config(c, a, otheta)
        except Exception:
            pass

    # Phase 3: Differential evolution for global search
    if time.time() < deadline - 50:
        try:
            bounds = [(-4, 4)] * 24 + [(0, 60)] * 12 + [(0, 60)]
            de_res = differential_evolution(objective, bounds, maxiter=60, 
                                            popsize=15, tol=1e-8, 
                                            seed=42, polish=False)
            if de_res.fun < 1e8:
                centers = [[de_res.x[i*2], de_res.x[i*2+1]] for i in range(12)]
                angles = [de_res.x[24+i] for i in range(12)]
                if is_valid(centers, angles):
                    val = compute_outer_side(centers, angles, 0.0, 0.0, de_res.x[36])
                    if val < best_val:
                        best_val = val
                        best_params = de_res.x.copy()
                        best_centers = centers
                        best_angles = angles
                        best_otheta = de_res.x[36]
        except Exception:
            pass

    # Phase 4: Multiple SLSQP restarts from perturbed best solution
    if time.time() < deadline - 30 and best_centers is not None:
        for restart in range(3):
            if time.time() > deadline - 20:
                break
            try:
                perturbed_centers = []
                for c in best_centers:
                    perturbed_centers.append([c[0] + random.uniform(-0.1, 0.1), 
                                             c[1] + random.uniform(-0.1, 0.1)])
                perturbed_angles = [a + random.uniform(-5, 5) for a in best_angles]
                
                slsqp_res = slsqp_optimize(perturbed_centers, perturbed_angles, max_iters=50)
                if slsqp_res is not None:
                    c, a, ocx, ocy, otheta, side = slsqp_res
                    if side < best_val:
                        best_val = side
                        best_centers = c
                        best_angles = a
                        best_ocx = ocx
                        best_ocy = ocy
                        best_otheta = otheta
                        best_params = params_from_config(c, a, otheta)
            except Exception:
                pass

    # Phase 5: Fine polish with Nelder-Mead
    if time.time() < deadline - 15 and best_params is not None:
        try:
            polish_res = minimize(objective, best_params, method='Nelder-Mead',
                                 options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            centers = [[polish_res.x[i*2], polish_res.x[i*2+1]] for i in range(12)]
            angles = [polish_res.x[24+i] for i in range(12)]
            if is_valid(centers, angles):
                val = compute_outer_side(centers, angles, 0.0, 0.0, polish_res.x[36])
                if val < best_val:
                    best_val = val
                    best_params = polish_res.x.copy()
                    best_centers = centers
                    best_angles = angles
                    best_otheta = polish_res.x[36]
        except Exception:
            pass

    # Phase 6: Polish outer hexagon parameters
    if best_centers is not None:
        best_ocx, best_ocy, best_otheta, best_val = polish_outer(
            best_centers, best_angles, best_ocx, best_ocy, best_otheta)

    # Fallback if no valid solution found
    if best_centers is None:
        best_centers, best_angles = generate_hexagonal_cluster()
        best_val = compute_outer_side(best_centers, best_angles, 0.0, 0.0, 0.0)
        best_otheta = 0.0

    inner = [[c[0], c[1], a] for c, a in zip(best_centers, best_angles)]
    outer_center = (best_ocx, best_ocy)
    outer_side = best_val
    outer_angle = best_otheta
    
    return inner, outer_center, outer_side, outer_angle

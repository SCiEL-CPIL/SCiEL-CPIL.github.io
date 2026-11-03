import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    def softplus(z, eps):
        return np.logaddexp(0, z / eps) * eps

    def get_initial_layout_hex():
        r0 = 0.12
        centers = []
        for row in range(10):
            for col in range(10):
                x = col * 2 * r0
                y = row * np.sqrt(3) * r0
                if row % 2 == 1:
                    x += r0
                centers.append((x, y))
        
        centers.sort(key=lambda p: (p[0]-0.5)**2 + (p[1]-0.5)**2)
        centers = centers[:N]
        
        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max_x - min_x
        height = max_y - min_y
        
        scale = min(0.8 / width, 0.8 / height)
        
        scaled_centers = []
        for x, y in centers:
            sx = (x - min_x) * scale + 0.1
            sy = (y - min_y) * scale + 0.1
            scaled_centers.append((sx, sy))
            
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((scaled_centers[i][0]-scaled_centers[j][0])**2 + (scaled_centers[i][1]-scaled_centers[j][1])**2)
                if d < min_dist:
                    min_dist = d
        min_boundary = np.inf
        for x, y in scaled_centers:
            min_boundary = min(min_boundary, x, 1-x, y, 1-y)
            
        r_init = min(min_dist / 2, min_boundary) * 0.95
        
        v = []
        for x, y in scaled_centers:
            v.extend([x, y, r_init])
        return np.array(v)

    def get_initial_layout_corner():
        positions = []
        corners = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92],
                   [0.5, 0.08], [0.5, 0.92], [0.08, 0.5], [0.92, 0.5]]
        for cx, cy in corners:
            positions.append([cx, cy, 0.08])
        rng = np.random.default_rng(42)
        for _ in range(N - len(positions)):
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            positions.append([x, y, rng.uniform(0.06, 0.10)])
        
        v = []
        for x, y, r in positions[:N]:
            v.extend([x, y, r])
        return np.array(v)

    def get_initial_layout_grid():
        side = int(np.ceil(np.sqrt(N)))
        positions = []
        r = 0.5/side * 0.9
        for i in range(side):
            for j in range(side):
                x = (i+0.5)/side
                y = (j+0.5)/side
                positions.append([x, y, r])
                if len(positions) >= N:
                    break
            if len(positions) >= N:
                break
        while len(positions) < N:
            positions.append([0.5, 0.5, r*0.5])
        
        v = []
        for x, y, r in positions[:N]:
            v.extend([x, y, r])
        return np.array(v)

    def objective(v, penalty=1000.0, eps=1e-4):
        x = v[0::3]
        y = v[1::3]
        r = np.maximum(v[2::3], 1e-6)
        
        score = -np.sum(r)
        
        lx = r - x
        ux = x + r - 1.0
        ly = r - y
        uy = y + r - 1.0
        wall_pen = np.sum(softplus(lx, eps)**2) + np.sum(softplus(ux, eps)**2) + \
                   np.sum(softplus(ly, eps)**2) + np.sum(softplus(uy, eps)**2)
        
        iu0, iu1 = np.triu_indices(N, k=1)
        dx = x[iu0] - x[iu1]
        dy = y[iu0] - y[iu1]
        dist = np.sqrt(dx*dx + dy*dy + 1e-12)
        overlap = (r[iu0] + r[iu1]) - dist
        pair_pen = np.sum(softplus(overlap, eps)**2)
        
        return score + penalty * (wall_pen + pair_pen)

    def gradient(v, penalty=1000.0, eps=1e-4):
        x = v[0::3]
        y = v[1::3]
        r = np.maximum(v[2::3], 1e-6)
        
        grad = np.zeros_like(v)
        grad[2::3] -= 1.0
        
        def d_softplus_sq(z, eps):
            sp = softplus(z, eps)
            sig = 1.0 / (1.0 + np.exp(-z / eps))
            return 2.0 * sp * sig

        for i in range(N):
            z = r[i] - x[i]
            dz = d_softplus_sq(z, eps)
            grad[i*3] += -penalty * dz
            grad[i*3+2] += penalty * dz
            
            z = x[i] + r[i] - 1.0
            dz = d_softplus_sq(z, eps)
            grad[i*3] += penalty * dz
            grad[i*3+2] += penalty * dz
            
            z = r[i] - y[i]
            dz = d_softplus_sq(z, eps)
            grad[i*3+1] += -penalty * dz
            grad[i*3+2] += penalty * dz
            
            z = y[i] + r[i] - 1.0
            dz = d_softplus_sq(z, eps)
            grad[i*3+1] += penalty * dz
            grad[i*3+2] += penalty * dz

        iu0, iu1 = np.triu_indices(N, k=1)
        dx = x[iu0] - x[iu1]
        dy = y[iu0] - y[iu1]
        dist = np.sqrt(dx*dx + dy*dy + 1e-12)
        overlap = (r[iu0] + r[iu1]) - dist
        
        dz = d_softplus_sq(overlap, eps)
        ux = dx / dist
        uy = dy / dist
        
        np.add.at(grad[0::3], iu0, -penalty * dz * ux)
        np.add.at(grad[0::3], iu1, penalty * dz * ux)
        np.add.at(grad[1::3], iu0, -penalty * dz * uy)
        np.add.at(grad[1::3], iu1, penalty * dz * uy)
        np.add.at(grad[2::3], iu0, penalty * dz)
        np.add.at(grad[2::3], iu1, penalty * dz)
        
        return grad

    def repair(v):
        p = v.reshape(N, 3).copy()
        p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
        p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
        p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)
        
        wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                          np.minimum(p[:, 1], 1.0 - p[:, 1]))
        p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))
        
        for _ in range(50):
            iu = np.triu_indices(N, k=1)
            dx = p[iu[0], 0] - p[iu[1], 0]
            dy = p[iu[0], 1] - p[iu[1], 1]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = (p[iu[0], 2] + p[iu[1], 2]) - dist
            active = overlap > 0
            if not np.any(active):
                break
            for k in range(len(active)):
                if not active[k]:
                    continue
                i = iu[0][k]
                j = iu[1][k]
                ov = overlap[k]
                if p[i, 2] >= p[j, 2]:
                    shrink = min(ov, p[i, 2] - 1e-6)
                    p[i, 2] -= shrink
                    rem = ov - shrink
                    if rem > 0:
                        shrink2 = min(rem, p[j, 2] - 1e-6)
                        p[j, 2] -= shrink2
                else:
                    shrink = min(ov, p[j, 2] - 1e-6)
                    p[j, 2] -= shrink
                    rem = ov - shrink
                    if rem > 0:
                        shrink2 = min(rem, p[i, 2] - 1e-6)
                        p[i, 2] -= shrink2
            wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                              np.minimum(p[:, 1], 1.0 - p[:, 1]))
            p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))
            
        return p.flatten()

    def is_valid(v, tol=1e-7):
        p = v.reshape(N, 3)
        x, y, r = p[:, 0], p[:, 1], p[:, 2]
        if np.any(r < 0) or np.any(x < -tol) or np.any(x > 1+tol) or np.any(y < -tol) or np.any(y > 1+tol):
            return False
        if np.any(r > x + tol) or np.any(r > 1-x + tol) or np.any(r > y + tol) or np.any(r > 1-y + tol):
            return False
        iu = np.triu_indices(N, k=1)
        dx = x[iu[0]] - x[iu[1]]
        dy = y[iu[0]] - y[iu[1]]
        dist = np.sqrt(dx*dx + dy*dy)
        if np.any(r[iu[0]] + r[iu[1]] > dist + tol):
            return False
        return True

    def physics_relax(v, time_budget=5.0):
        """
        Physics-based relaxation: treat circles as repelling particles with
        wall repulsion. This can escape local optima.
        """
        if time.time() > deadline - time_budget:
            return v
        
        p = v.reshape(N, 3).copy()
        dt = 0.01
        damping = 0.95
        
        for step in range(200):
            if time.time() > deadline - time_budget:
                break
            
            forces = np.zeros_like(p)
            
            # Pair repulsion (inverse square)
            for i in range(N):
                for j in range(i+1, N):
                    dx = p[i, 0] - p[j, 0]
                    dy = p[i, 1] - p[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                    min_dist = p[i, 2] + p[j, 2]
                    if dist < min_dist:
                        # Strong repulsion when overlapping
                        f = 100.0 * (min_dist - dist) / dist
                        forces[i, 0] += f * dx / dist
                        forces[i, 1] += f * dy / dist
                        forces[j, 0] -= f * dx / dist
                        forces[j, 1] -= f * dy / dist
            
            # Wall repulsion
            for i in range(N):
                if p[i, 0] < p[i, 2]:
                    forces[i, 0] += 100.0 * (p[i, 2] - p[i, 0])
                if 1.0 - p[i, 0] < p[i, 2]:
                    forces[i, 0] -= 100.0 * (p[i, 2] - (1.0 - p[i, 0]))
                if p[i, 1] < p[i, 2]:
                    forces[i, 1] += 100.0 * (p[i, 2] - p[i, 1])
                if 1.0 - p[i, 1] < p[i, 2]:
                    forces[i, 1] -= 100.0 * (p[i, 2] - (1.0 - p[i, 1]))
            
            # Update positions (keep radii fixed)
            p[:, 0] += dt * forces[:, 0] * damping
            p[:, 1] += dt * forces[:, 1] * damping
            
            # Clip to bounds
            p[:, 0] = np.clip(p[:, 0], 0, 1)
            p[:, 1] = np.clip(p[:, 1], 0, 1)
            
            # Repair if needed
            p_repaired = repair(p.flatten()).reshape(N, 3)
            if is_valid(p_repaired.flatten()):
                p = p_repaired
        
        return p.flatten()

    def analytic_polish(v_best, time_budget=10.0):
        """
        High-precision polish using active set detection and iterative refinement.
        """
        if time.time() > deadline - time_budget:
            return v_best
        
        p = v_best.reshape(N, 3).copy()
        
        # Identify active constraints (touching pairs and walls)
        active_pairs = []
        active_walls = []
        tol = 1e-5
        
        iu = np.triu_indices(N, k=1)
        for k in range(len(iu[0])):
            i, j = iu[0][k], iu[1][k]
            dx = p[i, 0] - p[j, 0]
            dy = p[i, 1] - p[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if abs(dist - (p[i, 2] + p[j, 2])) < tol:
                active_pairs.append((i, j))
        
        for i in range(N):
            if abs(p[i, 0] - p[i, 2]) < tol: active_walls.append((i, 0))
            if abs(1.0 - p[i, 0] - p[i, 2]) < tol: active_walls.append((i, 1))
            if abs(p[i, 1] - p[i, 2]) < tol: active_walls.append((i, 2))
            if abs(1.0 - p[i, 1] - p[i, 2]) < tol: active_walls.append((i, 3))
        
        # Iterative refinement:
        # 1. For each active pair, adjust positions to maintain contact
        # 2. For each active wall, adjust position to maintain wall contact
        # 3. Try to increase radii slightly while maintaining constraints
        
        for iteration in range(50):
            if time.time() > deadline - time_budget:
                break
            
            p_new = p.copy()
            
            # Enforce wall constraints
            for i, wall in active_walls:
                if wall == 0:  # left wall
                    p_new[i, 0] = p_new[i, 2]
                elif wall == 1:  # right wall
                    p_new[i, 0] = 1.0 - p_new[i, 2]
                elif wall == 2:  # bottom wall
                    p_new[i, 1] = p_new[i, 2]
                elif wall == 3:  # top wall
                    p_new[i, 1] = 1.0 - p_new[i, 2]
            
            # Enforce pair constraints (iterative)
            for _ in range(10):
                for i, j in active_pairs:
                    dx = p_new[i, 0] - p_new[j, 0]
                    dy = p_new[i, 1] - p_new[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                    target = p_new[i, 2] + p_new[j, 2]
                    if dist > 0:
                        scale = target / dist
                        # Move both circles toward each other
                        mid_x = (p_new[i, 0] + p_new[j, 0]) / 2
                        mid_y = (p_new[i, 1] + p_new[j, 1]) / 2
                        new_dx = dx * scale
                        new_dy = dy * scale
                        p_new[i, 0] = mid_x + new_dx / 2
                        p_new[i, 1] = mid_y + new_dy / 2
                        p_new[j, 0] = mid_x - new_dx / 2
                        p_new[j, 1] = mid_y - new_dy / 2
            
            # Check if improved
            if np.sum(p_new[:, 2]) >= np.sum(p[:, 2]):
                p = p_new
            
            # Try to increase radii slightly
            for i in range(N):
                # Check if circle can be enlarged
                min_margin = np.inf
                for j in range(N):
                    if i == j:
                        continue
                    dx = p[i, 0] - p[j, 0]
                    dy = p[i, 1] - p[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    margin = dist - (p[i, 2] + p[j, 2])
                    min_margin = min(min_margin, margin)
                
                # Check walls
                min_margin = min(min_margin, p[i, 0] - p[i, 2],
                                 1.0 - p[i, 0] - p[i, 2],
                                 p[i, 1] - p[i, 2],
                                 1.0 - p[i, 1] - p[i, 2])
                
                if min_margin > 1e-6:
                    p[i, 2] += min(min_margin / 2, 1e-4)
            
            p = repair(p.flatten()).reshape(N, 3)
        
        return p.flatten()

    def contact_polish(v_best, time_budget=10.0):
        """
        Combined polish: physics relaxation + analytic refinement + L-BFGS-B.
        """
        if time.time() > deadline - time_budget:
            return v_best
        
        # Step 1: Physics relaxation to escape local optima
        v_phys = physics_relax(v_best, time_budget=time_budget/3)
        
        # Step 2: Analytic polish
        v_analytic = analytic_polish(v_phys, time_budget=time_budget/3)
        
        # Step 3: Final L-BFGS-B with high precision
        v_final = v_analytic.copy()
        
        try:
            res = minimize(objective, v_final, jac=gradient, method='L-BFGS-B', 
                           bounds=[(0, 1), (0, 1), (1e-6, 0.5)] * N,
                           args=(10000.0, 1e-6),
                           options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50})
            v_res = res.x
        except Exception:
            v_res = v_final
        
        v_res = repair(v_res)
        
        # Return best of all candidates
        candidates = [v_best, v_phys, v_analytic, v_res]
        best = v_best
        best_score = np.sum(v_best[2::3])
        
        for c in candidates:
            if is_valid(c):
                s = np.sum(c[2::3])
                if s > best_score:
                    best_score = s
                    best = c
        
        return best

    bounds = [(0, 1), (0, 1), (1e-6, 0.5)] * N
    
    best_score = -1
    best_v = None
    
    initial_generators = [get_initial_layout_hex, get_initial_layout_corner, get_initial_layout_grid]
    gen_idx = 0
    
    # Multi-start loop
    while time.time() < deadline - 15:  # Leave 15s for final polish
        if gen_idx == 0:
            v_init = get_initial_layout_hex()
        elif gen_idx == 1:
            v_init = get_initial_layout_corner()
        else:
            v_init = get_initial_layout_grid()
        
        # Add randomness
        v_init = v_init + np.random.normal(0, 0.03, size=3*N)
        v_init[0::3] = np.clip(v_init[0::3], 0, 1)
        v_init[1::3] = np.clip(v_init[1::3], 0, 1)
        v_init[2::3] = np.clip(v_init[2::3], 1e-6, 0.5)
        
        try:
            res = minimize(objective, v_init, jac=gradient, method='L-BFGS-B', bounds=bounds, 
                           args=(1000.0, 1e-4),
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            v_res = res.x
        except Exception:
            v_res = v_init
        
        v_repaired = repair(v_res)
        
        if is_valid(v_repaired):
            score = np.sum(v_repaired[2::3])
            if score > best_score:
                best_score = score
                best_v = v_repaired.copy()
        
        # Physics relaxation on current best to escape local optima
        if best_v is not None and time.time() < deadline - 20:
            v_phys = physics_relax(best_v, time_budget=3.0)
            if is_valid(v_phys):
                phys_score = np.sum(v_phys[2::3])
                if phys_score > best_score:
                    best_score = phys_score
                    best_v = v_phys.copy()
        
        # Polish current best
        if best_v is not None and time.time() < deadline - 15:
            best_v = contact_polish(best_v, time_budget=5.0)
            if is_valid(best_v):
                new_score = np.sum(best_v[2::3])
                if new_score > best_score:
                    best_score = new_score
        
        gen_idx = (gen_idx + 1) % len(initial_generators)
        
        # Perturb around best
        if best_v is not None and np.random.rand() < 0.5:
            v_init = best_v + np.random.normal(0, 0.02, size=3*N)
            v_init[0::3] = np.clip(v_init[0::3], 0, 1)
            v_init[1::3] = np.clip(v_init[1::3], 0, 1)
            v_init[2::3] = np.clip(v_init[2::3], 1e-6, 0.5)
    
    # Final high-precision polish
    if best_v is not None:
        # Multiple polish passes for better convergence
        for _ in range(3):
            if time.time() > deadline - 2:
                break
            best_v = contact_polish(best_v, time_budget=5.0)
            if is_valid(best_v):
                final_score = np.sum(best_v[2::3])
                if final_score > best_score:
                    best_score = final_score
    
    if best_v is None:
        v_fallback = get_initial_layout_hex()
        v_fallback = repair(v_fallback)
        return v_fallback.reshape(N, 3)
    
    return best_v.reshape(N, 3)

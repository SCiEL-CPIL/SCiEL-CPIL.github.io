"""Circle packing: multi-start local optimization with smooth penalty + trust-constr polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    best_sum = 0.0
    best_result = None

    def check_validity(x, y, r, tol=1e-9):
        """Check if packing is valid (all constraints satisfied)."""
        if np.any(x < r - tol) or np.any(x > 1 - r + tol) or np.any(y < r - tol) or np.any(y > 1 - r + tol):
            return False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if d < r[i] + r[j] - tol:
                    return False
        return True

    def objective(z, penalty_weight=5000.0):
        """Objective: minimize -sum(r) + penalty for violations."""
        x = z[:N]
        y = z[N:2*N]
        r = z[2*N:]

        r_safe = np.maximum(r, 0.001)
        sum_r = np.sum(r_safe)

        # Boundary violations
        v_bound = 0.0
        v_bound += np.sum(np.maximum(0, r_safe - x)**2)
        v_bound += np.sum(np.maximum(0, r_safe - (1 - x))**2)
        v_bound += np.sum(np.maximum(0, r_safe - y)**2)
        v_bound += np.sum(np.maximum(0, r_safe - (1 - y))**2)

        # Collision violations - vectorized
        x_diff = x[:, None] - x[None, :]
        y_diff = y[:, None] - y[None, :]
        dist_sq = x_diff**2 + y_diff**2
        sum_r_mat = r_safe[:, None] + r_safe[None, :]
        overlap_terms = np.maximum(0, sum_r_mat**2 - dist_sq)
        np.fill_diagonal(overlap_terms, 0)
        v_coll = np.sum(overlap_terms**2)

        return -sum_r + penalty_weight * (v_bound + v_coll)

    def gradient(z):
        """Gradient of objective for L-BFGS-B."""
        x = z[:N]
        y = z[N:2*N]
        r = z[2*N:]
        r_safe = np.maximum(r, 0.001)
        px = np.zeros(N)
        py = np.zeros(N)
        pr = np.zeros(N)

        # Boundary gradient
        for i in range(N):
            ri = r_safe[i]
            if ri - x[i] > 0:
                px[i] -= 2 * penalty_weight * (ri - x[i])
            if ri - (1 - x[i]) > 0:
                px[i] += 2 * penalty_weight * (ri - (1 - x[i]))
            if ri - y[i] > 0:
                py[i] -= 2 * penalty_weight * (ri - y[i])
            if ri - (1 - y[i]) > 0:
                py[i] += 2 * penalty_weight * (ri - (1 - y[i]))

        # Collision gradient
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d_sq = dx*dx + dy*dy
                s = r_safe[i] + r_safe[j]
                if s*s - d_sq > 0:
                    overlap = s*s - d_sq
                    if d_sq > 1e-12:
                        d = np.sqrt(d_sq)
                        # d(overlap)/d(x[i]) = -2*(x[i]-x[j])
                        # d(overlap^2)/d(x[i]) = 2*overlap * (-2*dx)
                        gx = 4 * penalty_weight * overlap * dx
                        gy = 4 * penalty_weight * overlap * dy
                        px[i] += gx
                        py[i] += gy
                        px[j] -= gx
                        py[j] -= gy
                    # Gradient w.r.t. radii
                    pr[i] += 4 * penalty_weight * overlap * s
                    pr[j] += 4 * penalty_weight * overlap * s

        # Objective gradient
        pr -= np.ones(N) * 1.0  # from -sum(r)

        return np.concatenate([px, py, pr])

    def local_optimize(x0, y0, r0, max_iter=400, use_gradient=True):
        """Run local optimization from initial guess."""
        z0 = np.concatenate([x0, y0, r0])
        try:
            if use_gradient:
                res = minimize(objective, z0, jac=gradient, method='L-BFGS-B',
                              options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
            else:
                res = minimize(objective, z0, method='L-BFGS-B',
                              options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:]
            r_opt = np.maximum(r_opt, 0.001)

            if check_validity(x_opt, y_opt, r_opt):
                return x_opt, y_opt, r_opt
            else:
                for scale in [0.98, 0.95, 0.90, 0.85]:
                    r_adj = r_opt * scale
                    if check_validity(x_opt, y_opt, r_adj):
                        return x_opt, y_opt, r_adj
                return None
        except Exception:
            return None

    def trust_constr_polish(x0, y0, r0, max_iter=200):
        """Use trust-constr with explicit constraints for high-precision polish."""
        z0 = np.concatenate([x0, y0, r0])
        
        def obj(z):
            return -np.sum(np.maximum(z[2*N:], 0.001))

        # Constraints: g(z) >= 0
        cons = []
        # Boundary constraints
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[i] - z[2*N+i], 'jac': lambda z, i=i: np.array([1.0]*N + [0.0]*N + [-1.0]*N) if False else None})
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: 1 - z[i] - z[2*N+i]})
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[N+i] - z[2*N+i]})
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: 1 - z[N+i] - z[2*N+i]})
        # Pairwise constraints
        for i in range(N):
            for j in range(i+1, N):
                def pair_con(z, i=i, j=j):
                    dx = z[i] - z[j]
                    dy = z[N+i] - z[N+j]
                    d = np.sqrt(dx*dx + dy*dy)
                    return d - z[2*N+i] - z[2*N+j]
                cons.append({'type': 'ineq', 'fun': pair_con})

        # Bounds
        bounds = []
        for i in range(N):
            bounds.append((0.001, 0.5))
            bounds.append((0.001, 0.5))
            bounds.append((0.001, 0.5))

        try:
            res = minimize(obj, z0, method='trust-constr', bounds=bounds, 
                          constraints=cons, options={'maxiter': max_iter, 'gtol': 1e-8})
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:]
            r_opt = np.maximum(r_opt, 0.001)

            if check_validity(x_opt, y_opt, r_opt):
                return x_opt, y_opt, r_opt
            else:
                for scale in [0.98, 0.95, 0.90]:
                    r_adj = r_opt * scale
                    if check_validity(x_opt, y_opt, r_adj):
                        return x_opt, y_opt, r_adj
                return None
        except Exception:
            return None

    def force_directed_anneal(x0, y0, r0, n_steps=2000, T0=0.05, T1=0.001):
        """Force-directed system with simulated annealing to find dense packings."""
        x = x0.copy().astype(float)
        y = y0.copy().astype(float)
        r = r0.copy().astype(float)

        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        best_sum_local = np.sum(r)

        r_mean = np.mean(r) if np.mean(r) > 0 else 0.05

        for step in range(n_steps):
            if time.time() > deadline:
                break

            # Temperature schedule (exponential cooling)
            t = step / max(n_steps - 1, 1)
            T = T0 * (T1 / T0) ** t

            # Compute forces
            fx = np.zeros(N)
            fy = np.zeros(N)

            # Pairwise repulsion for overlapping circles
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    req = r[i] + r[j]
                    if d < req and d > 1e-10:
                        overlap = req - d
                        force = 10.0 * overlap / max(r_mean, 0.01)
                        fx[i] += force * dx / d
                        fy[i] += force * dy / d
                        fx[j] -= force * dx / d
                        fy[j] -= force * dy / d

            # Boundary forces
            for i in range(N):
                if x[i] < r[i]:
                    fx[i] += 5.0 * (r[i] - x[i]) / max(r_mean, 0.01)
                if x[i] > 1 - r[i]:
                    fx[i] -= 5.0 * (x[i] - (1 - r[i])) / max(r_mean, 0.01)
                if y[i] < r[i]:
                    fy[i] += 5.0 * (r[i] - y[i]) / max(r_mean, 0.01)
                if y[i] > 1 - r[i]:
                    fy[i] -= 5.0 * (y[i] - (1 - r[i])) / max(r_mean, 0.01)

            # Apply forces with annealing noise
            dx = fx * 0.01 + np.random.normal(0, T, N)
            dy = fy * 0.01 + np.random.normal(0, T, N)

            x += dx
            y += dy

            # Project back to valid region
            for i in range(N):
                x[i] = np.clip(x[i], r[i], 1 - r[i])
                y[i] = np.clip(y[i], r[i], 1 - r[i])

            # Try to grow radii if configuration is valid
            if t > 0.3:
                valid = True
                for i in range(N):
                    for j in range(i + 1, N):
                        d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                        if d < r[i] + r[j] - 1e-9:
                            valid = False
                            break
                    if not valid:
                        break

                if valid:
                    growth = 0.0005 * (T / T1)
                    new_r = r + growth
                    valid_growth = True
                    for i in range(N):
                        max_r_i = min(x[i], 1-x[i], y[i], 1-y[i])
                        if new_r[i] > max_r_i:
                            valid_growth = False
                            break
                        for j in range(i + 1, N):
                            d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                            if d < new_r[i] + new_r[j] - 1e-9:
                                valid_growth = False
                                break
                        if not valid_growth:
                            break

                    if valid_growth:
                        r = new_r

            current_sum = np.sum(r)
            if current_sum > best_sum_local:
                best_sum_local = current_sum
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()

        return best_x, best_y, best_r

    def hexagonal_init(n_circles):
        """Create hexagonal packing initialization with proper centering."""
        x = np.zeros(n_circles)
        y = np.zeros(n_circles)
        r = np.zeros(n_circles)

        cols = 6
        rows = 5
        n_in_grid = cols * rows

        spacing = 0.18
        row_h = spacing * 0.866

        total_w = (cols - 1) * spacing
        total_h = (rows - 1) * row_h
        offset_x = (1 - total_w) / 2
        offset_y = (1 - total_h) / 2

        for i in range(min(n_circles, n_in_grid)):
            row = i // cols
            col = i % cols
            x[i] = offset_x + col * spacing + (row % 2) * spacing * 0.5
            y[i] = offset_y + row * row_h
            r[i] = spacing * 0.48

        for i in range(n_in_grid, n_circles):
            x[i] = 0.5 + np.random.uniform(-0.3, 0.3)
            y[i] = 0.5 + np.random.uniform(-0.3, 0.3)
            r[i] = 0.08

        for i in range(n_circles):
            r[i] = min(r[i], min(x[i], 1-x[i], y[i], 1-y[i]) * 0.99)

        return x, y, r

    def random_init():
        """Random initialization."""
        r = np.random.uniform(0.03, 0.1, N)
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        return x, y, r

    def grid_init():
        """Grid initialization."""
        side = int(np.ceil(np.sqrt(N)))
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            row = i // side
            col = i % side
            x[i] = (col + 0.5) / side
            y[i] = (row + 0.5) / side
            r[i] = 0.5 / side * 0.95
        return x, y, r

    def corner_heavy_init():
        """Corner-heavy initialization."""
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        # Corners
        corners = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
        for i, (cx, cy) in enumerate(corners):
            x[i] = cx
            y[i] = cy
            r[i] = 0.12
        # Rest distributed
        for i in range(4, N):
            x[i] = np.random.uniform(0.15, 0.85)
            y[i] = np.random.uniform(0.15, 0.85)
            r[i] = 0.08
        for i in range(N):
            r[i] = min(r[i], min(x[i], 1-x[i], y[i], 1-y[i]) * 0.99)
        return x, y, r

    def center_ring_init():
        """Center-ring initialization."""
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        x[0] = 0.5; y[0] = 0.5; r[0] = 0.12
        n_ring = N - 1
        ring_radius = 0.35
        for i in range(1, N):
            angle = 2 * np.pi * (i - 1) / n_ring
            x[i] = 0.5 + ring_radius * np.cos(angle)
            y[i] = 0.5 + ring_radius * np.sin(angle)
            r[i] = 0.08
        for i in range(N):
            r[i] = min(r[i], min(x[i], 1-x[i], y[i], 1-y[i]) * 0.99)
        return x, y, r

    def two_ring_init():
        """Two-ring initialization."""
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        x[0] = 0.5; y[0] = 0.5; r[0] = 0.10
        n_inner = 8
        inner_radius = 0.25
        for i in range(1, 1 + n_inner):
            angle = 2 * np.pi * (i - 1) / n_inner
            x[i] = 0.5 + inner_radius * np.cos(angle)
            y[i] = 0.5 + inner_radius * np.sin(angle)
            r[i] = 0.07
        n_outer = N - 1 - n_inner
        outer_radius = 0.42
        for i in range(1 + n_inner, N):
            idx = i - 1 - n_inner
            angle = 2 * np.pi * idx / n_outer + np.pi / n_outer
            x[i] = 0.5 + outer_radius * np.cos(angle)
            y[i] = 0.5 + outer_radius * np.sin(angle)
            r[i] = 0.06
        for i in range(N):
            r[i] = min(r[i], min(x[i], 1-x[i], y[i], 1-y[i]) * 0.99)
        return x, y, r

    def try_optimize(x0, y0, r0):
        """Try to optimize from given initial state, return result if better."""
        nonlocal best_sum, best_result
        result = local_optimize(x0, y0, r0)
        if result is not None:
            x_opt, y_opt, r_opt = result
            current_sum = np.sum(r_opt)
            if current_sum > best_sum:
                best_sum = current_sum
                best_result = np.column_stack((x_opt, y_opt, r_opt))

    def try_anneal_then_optimize(x0, y0, r0, n_anneal_steps=1500):
        """Run force-directed annealing then local optimization."""
        nonlocal best_sum, best_result
        if time.time() > deadline:
            return
        x_a, y_a, r_a = force_directed_anneal(x0, y0, r0, n_steps=n_anneal_steps)
        result = local_optimize(x_a, y_a, r_a, max_iter=200)
        if result is not None:
            x_opt, y_opt, r_opt = result
            current_sum = np.sum(r_opt)
            if current_sum > best_sum:
                best_sum = current_sum
                best_result = np.column_stack((x_opt, y_opt, r_opt))

    # Phase 1: Multi-start local optimization with diverse initializations
    # Hexagonal starts
    for k in range(15):
        if time.time() > deadline - 30:
            break
        x0, y0, r0 = hexagonal_init(N)
        x0 += np.random.uniform(-0.015, 0.015, N)
        y0 += np.random.uniform(-0.015, 0.015, N)
        r0 *= np.random.uniform(0.92, 1.08, N)
        try_optimize(x0, y0, r0)

    # Random starts
    for k in range(20):
        if time.time() > deadline - 30:
            break
        x0, y0, r0 = random_init()
        try_optimize(x0, y0, r0)

    # Grid starts
    for k in range(5):
        if time.time() > deadline - 30:
            break
        x0, y0, r0 = grid_init()
        x0 += np.random.uniform(-0.01, 0.01, N)
        y0 += np.random.uniform(-0.01, 0.01, N)
        try_optimize(x0, y0, r0)

    # Corner-heavy starts
    for k in range(8):
        if time.time() > deadline - 30:
            break
        x0, y0, r0 = corner_heavy_init()
        x0 += np.random.uniform(-0.01, 0.01, N)
        y0 += np.random.uniform(-0.01, 0.01, N)
        try_optimize(x0, y0, r0)

    # Center-ring starts
    for k in range(8):
        if time.time() > deadline - 30:
            break
        x0, y0, r0 = center_ring_init()
        x0 += np.random.uniform(-0.01, 0.01, N)
        y0 += np.random.uniform(-0.01, 0.01, N)
        r0 *= np.random.uniform(0.9, 1.1, N)
        try_optimize(x0, y0, r0)

    # Two-ring starts
    for k in range(8):
        if time.time() > deadline - 30:
            break
        x0, y0, r0 = two_ring_init()
        x0 += np.random.uniform(-0.01, 0.01, N)
        y0 += np.random.uniform(-0.01, 0.01, N)
        r0 *= np.random.uniform(0.9, 1.1, N)
        try_optimize(x0, y0, r0)

    # Phase 2: Force-directed annealing + local optimization
    if best_result is not None:
        for k in range(5):
            if time.time() > deadline - 20:
                break
            x0 = best_result[:, 0].copy()
            y0 = best_result[:, 1].copy()
            r0 = best_result[:, 2].copy()
            x0 += np.random.uniform(-0.02, 0.02, N)
            y0 += np.random.uniform(-0.02, 0.02, N)
            r0 *= 0.85
            try_anneal_then_optimize(x0, y0, r0, n_anneal_steps=1200)

    for k in range(10):
        if time.time() > deadline - 20:
            break
        x0, y0, r0 = hexagonal_init(N)
        x0 += np.random.uniform(-0.03, 0.03, N)
        y0 += np.random.uniform(-0.03, 0.03, N)
        r0 *= np.random.uniform(0.8, 1.0, N)
        try_anneal_then_optimize(x0, y0, r0, n_anneal_steps=1000)

    for k in range(6):
        if time.time() > deadline - 20:
            break
        x0, y0, r0 = random_init()
        try_anneal_then_optimize(x0, y0, r0, n_anneal_steps=800)

    # Phase 3: Final high-precision polish on best result
    if best_result is not None and time.time() < deadline - 15:
        x_best = best_result[:, 0].copy()
        y_best = best_result[:, 1].copy()
        r_best = best_result[:, 2].copy()

        # L-BFGS-B polish with gradient
        z0 = np.concatenate([x_best, y_best, r_best])
        try:
            res = minimize(objective, z0, jac=gradient, method='L-BFGS-B',
                          options={'maxiter': 600, 'ftol': 1e-12, 'gtol': 1e-10})
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:]
            r_opt = np.maximum(r_opt, 0.001)
            if check_validity(x_opt, y_opt, r_opt):
                current_sum = np.sum(r_opt)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = np.column_stack((x_opt, y_opt, r_opt))
        except Exception:
            pass

        # Trust-constr polish
        if time.time() < deadline - 5:
            x_polish = best_result[:, 0].copy()
            y_polish = best_result[:, 1].copy()
            r_polish = best_result[:, 2].copy()
            result_tc = trust_constr_polish(x_polish, y_polish, r_polish, max_iter=150)
            if result_tc is not None:
                x_opt, y_opt, r_opt = result_tc
                current_sum = np.sum(r_opt)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = np.column_stack((x_opt, y_opt, r_opt))

    # Fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best_result
# EVOLVE-BLOCK-END

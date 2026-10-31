"""Circle packing in unit square - optimized with radial expansion and multi-start."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin for 150s limit

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

    def objective(z, penalty_weight=5000.0, smooth_eps=0.005):
        """Objective: minimize -sum(r) + smooth penalty using softplus for better gradient flow."""
        x = z[:N]
        y = z[N:2*N]
        r = z[2*N:]

        r_safe = np.maximum(r, 0.001)
        sum_r = np.sum(r_safe)

        # Smooth softplus: max(0, v) ≈ eps * log(1 + exp(v/eps))
        def softplus(v, eps=smooth_eps):
            return eps * np.log1p(np.exp(np.clip(v / eps, -50, 50)))

        # Boundary violations (smooth)
        v_bound = 0.0
        v_bound += np.sum(softplus(r_safe - x))
        v_bound += np.sum(softplus(r_safe - (1 - x)))
        v_bound += np.sum(softplus(r_safe - y))
        v_bound += np.sum(softplus(r_safe - (1 - y)))

        # Collision violations - vectorized, smooth
        x_diff = x[:, None] - x[None, :]
        y_diff = y[:, None] - y[None, :]
        dist_sq = x_diff**2 + y_diff**2
        sum_r_mat = r_safe[:, None] + r_safe[None, :]
        # overlap = (r_i + r_j)^2 - d^2, smooth max(0, overlap)
        overlap_terms = softplus(sum_r_mat**2 - dist_sq)
        np.fill_diagonal(overlap_terms, 0)
        v_coll = np.sum(overlap_terms)

        return -sum_r + penalty_weight * (v_bound + v_coll)

    def grow_and_push(x, y, r, max_iters=300):
        """Hard-constraint grow-and-push: enforce feasibility strictly, then grow radii."""
        x = x.copy().astype(float)
        y = y.copy().astype(float)
        r = r.copy().astype(float)
        r = np.maximum(r, 0.001)

        for it in range(max_iters):
            if time.time() > deadline:
                break

            # First: resolve overlaps by pushing apart (strict)
            resolved = True
            for _push in range(80):
                any_overlap = False
                for i in range(N):
                    for j in range(i + 1, N):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        d = np.sqrt(dx*dx + dy*dy)
                        req = r[i] + r[j]
                        if d < req - 1e-12:
                            any_overlap = True
                            if d < 1e-10:
                                dx = np.random.uniform(-0.01, 0.01)
                                dy = np.random.uniform(-0.01, 0.01)
                                d = np.sqrt(dx*dx + dy*dy)
                            push = (req - d) / 2.0 + 1e-6
                            fx = push * dx / d
                            fy = push * dy / d
                            x[i] += fx
                            y[i] += fy
                            x[j] -= fx
                            y[j] -= fy

                # Project to boundary
                for i in range(N):
                    x[i] = np.clip(x[i], r[i], 1 - r[i])
                    y[i] = np.clip(y[i], r[i], 1 - r[i])

                if not any_overlap:
                    break

            if not any_overlap:
                # Configuration is valid; try to grow all radii
                growth = 0.003
                new_r = r + growth
                # Check boundary
                for i in range(N):
                    max_bi = min(x[i], 1-x[i], y[i], 1-y[i])
                    if new_r[i] > max_bi:
                        new_r[i] = max_bi
                # Check collisions after growth
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                        if d < new_r[i] + new_r[j] - 1e-12:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    r = new_r
                else:
                    # Try smaller growth
                    for frac in [0.5, 0.25, 0.1, 0.05]:
                        new_r = r + growth * frac
                        for i in range(N):
                            max_bi = min(x[i], 1-x[i], y[i], 1-y[i])
                            if new_r[i] > max_bi:
                                new_r[i] = max_bi
                        valid = True
                        for i in range(N):
                            for j in range(i+1, N):
                                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                                if d < new_r[i] + new_r[j] - 1e-12:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            r = new_r
                            break
                    break  # No more growth possible at this position

            # Reduce growth rate over time
            growth *= 0.97

        return x, y, r

    def radial_expansion(x, y, r, max_rounds=50):
        """Iteratively expand each circle's radius as much as possible."""
        x = x.copy().astype(float)
        y = y.copy().astype(float)
        r = r.copy().astype(float)

        for _round in range(max_rounds):
            if time.time() > deadline:
                break
            improved = False
            for i in range(N):
                # Max radius for circle i given others fixed
                max_r = min(x[i], 1-x[i], y[i], 1-y[i])
                for j in range(N):
                    if i == j:
                        continue
                    d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    max_r = min(max_r, d - r[j])
                if max_r > r[i] + 1e-8:
                    r[i] = max_r - 1e-6
                    improved = True
            if not improved:
                break

        return x, y, r

    def local_optimize(x0, y0, r0, max_iter=500):
        """Run local optimization from initial guess with grow-and-push pre-processing."""
        z0 = np.concatenate([x0, y0, r0])
        try:
            # Pre-process with grow-and-push to get a valid starting point
            x_gp, y_gp, r_gp = grow_and_push(x0, y0, r0, max_iters=150)
            z0 = np.concatenate([x_gp, y_gp, r_gp])

            # Two-stage optimization: coarse then fine
            res = minimize(objective, z0, method='L-BFGS-B',
                          options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:]
            r_opt = np.maximum(r_opt, 0.001)

            if check_validity(x_opt, y_opt, r_opt):
                # Post-process with radial expansion
                x_re, y_re, r_re = radial_expansion(x_opt, y_opt, r_opt, max_rounds=30)
                if check_validity(x_re, y_re, r_re):
                    return x_re, y_re, r_re
                return x_opt, y_opt, r_opt
            else:
                # Try grow-and-push on the optimizer result
                x_gp2, y_gp2, r_gp2 = grow_and_push(x_opt, y_opt, r_opt, max_iters=80)
                if check_validity(x_gp2, y_gp2, r_gp2):
                    x_re, y_re, r_re = radial_expansion(x_gp2, y_gp2, r_gp2, max_rounds=20)
                    if check_validity(x_re, y_re, r_re):
                        return x_re, y_re, r_re
                    return x_gp2, y_gp2, r_gp2
                else:
                    for scale in [0.95, 0.90, 0.85]:
                        r_adj = r_opt * scale
                        if check_validity(x_opt, y_opt, r_adj):
                            return x_opt, y_opt, r_adj
                    # Fallback to grow-and-push result
                    if check_validity(x_gp, y_gp, r_gp):
                        return x_gp, y_gp, r_gp
                    return None
        except Exception:
            try:
                x_gp, y_gp, r_gp = grow_and_push(x0, y0, r0, max_iters=150)
                if check_validity(x_gp, y_gp, r_gp):
                    return x_gp, y_gp, r_gp
            except Exception:
                pass
            return None

    def force_directed_anneal(x0, y0, r0, n_steps=2000, T0=0.04, T1=0.0005):
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

            # Temperature schedule
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
                        force = 15.0 * overlap / max(r_mean, 0.01)
                        fx[i] += force * dx / d
                        fy[i] += force * dy / d
                        fx[j] -= force * dx / d
                        fy[j] -= force * dy / d

            # Boundary forces
            for i in range(N):
                if x[i] < r[i]:
                    fx[i] += 8.0 * (r[i] - x[i]) / max(r_mean, 0.01)
                if x[i] > 1 - r[i]:
                    fx[i] -= 8.0 * (x[i] - (1 - r[i])) / max(r_mean, 0.01)
                if y[i] < r[i]:
                    fy[i] += 8.0 * (r[i] - y[i]) / max(r_mean, 0.01)
                if y[i] > 1 - r[i]:
                    fy[i] -= 8.0 * (y[i] - (1 - r[i])) / max(r_mean, 0.01)

            # Apply forces with annealing noise
            dx = fx * 0.008 + np.random.normal(0, T, N)
            dy = fy * 0.008 + np.random.normal(0, T, N)

            x += dx
            y += dy

            # Project back to valid region
            for i in range(N):
                x[i] = np.clip(x[i], r[i], 1 - r[i])
                y[i] = np.clip(y[i], r[i], 1 - r[i])

            # Try to grow radii if configuration is valid
            if t > 0.4:
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
                    growth = 0.0004 * (T / T1)
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
        """Create hexagonal packing initialization."""
        x = np.zeros(n_circles)
        y = np.zeros(n_circles)
        r = np.zeros(n_circles)

        cols = 6
        rows = 5
        n_in_grid = cols * rows

        spacing = 0.17
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
            r[i] = spacing * 0.47

        for i in range(n_in_grid, n_circles):
            x[i] = 0.5 + np.random.uniform(-0.3, 0.3)
            y[i] = 0.5 + np.random.uniform(-0.3, 0.3)
            r[i] = 0.08

        for i in range(n_circles):
            r[i] = min(r[i], min(x[i], 1-x[i], y[i], 1-y[i]) * 0.99)

        return x, y, r

    def random_init():
        """Random initialization."""
        r = np.random.uniform(0.03, 0.11, N)
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

    def center_ring_init():
        """Center-ring initialization."""
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        x[0] = 0.5; y[0] = 0.5; r[0] = 0.13
        n_ring = N - 1
        ring_radius = 0.33
        for i in range(1, N):
            angle = 2 * np.pi * (i - 1) / n_ring
            x[i] = 0.5 + ring_radius * np.cos(angle)
            y[i] = 0.5 + ring_radius * np.sin(angle)
            r[i] = 0.085
        for i in range(N):
            r[i] = min(r[i], min(x[i], 1-x[i], y[i], 1-y[i]) * 0.99)
        return x, y, r

    def two_ring_init():
        """Two-ring initialization."""
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        x[0] = 0.5; y[0] = 0.5; r[0] = 0.11
        n_inner = 8
        inner_radius = 0.24
        for i in range(1, 1 + n_inner):
            angle = 2 * np.pi * (i - 1) / n_inner
            x[i] = 0.5 + inner_radius * np.cos(angle)
            y[i] = 0.5 + inner_radius * np.sin(angle)
            r[i] = 0.075
        n_outer = N - 1 - n_inner
        outer_radius = 0.40
        for i in range(1 + n_inner, N):
            idx = i - 1 - n_inner
            angle = 2 * np.pi * idx / n_outer + np.pi / n_outer
            x[i] = 0.5 + outer_radius * np.cos(angle)
            y[i] = 0.5 + outer_radius * np.sin(angle)
            r[i] = 0.065
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
        result = local_optimize(x_a, y_a, r_a, max_iter=300)
        if result is not None:
            x_opt, y_opt, r_opt = result
            current_sum = np.sum(r_opt)
            if current_sum > best_sum:
                best_sum = current_sum
                best_result = np.column_stack((x_opt, y_opt, r_opt))

    # Phase 1: Multi-start local optimization with diverse initializations
    # Hexagonal starts
    for k in range(20):
        if time.time() > deadline - 20:
            break
        x0, y0, r0 = hexagonal_init(N)
        x0 += np.random.uniform(-0.015, 0.015, N)
        y0 += np.random.uniform(-0.015, 0.015, N)
        r0 *= np.random.uniform(0.92, 1.08, N)
        try_optimize(x0, y0, r0)

    # Random starts
    for k in range(20):
        if time.time() > deadline - 20:
            break
        x0, y0, r0 = random_init()
        try_optimize(x0, y0, r0)

    # Grid starts
    for k in range(8):
        if time.time() > deadline - 20:
            break
        x0, y0, r0 = grid_init()
        x0 += np.random.uniform(-0.01, 0.01, N)
        y0 += np.random.uniform(-0.01, 0.01, N)
        try_optimize(x0, y0, r0)

    # Center-ring starts
    for k in range(12):
        if time.time() > deadline - 20:
            break
        x0, y0, r0 = center_ring_init()
        x0 += np.random.uniform(-0.01, 0.01, N)
        y0 += np.random.uniform(-0.01, 0.01, N)
        r0 *= np.random.uniform(0.9, 1.1, N)
        try_optimize(x0, y0, r0)

    # Two-ring starts
    for k in range(12):
        if time.time() > deadline - 20:
            break
        x0, y0, r0 = two_ring_init()
        x0 += np.random.uniform(-0.01, 0.01, N)
        y0 += np.random.uniform(-0.01, 0.01, N)
        r0 *= np.random.uniform(0.9, 1.1, N)
        try_optimize(x0, y0, r0)

    # Phase 2: Force-directed annealing + local optimization
    if best_result is not None:
        for k in range(8):
            if time.time() > deadline - 15:
                break
            x0 = best_result[:, 0].copy()
            y0 = best_result[:, 1].copy()
            r0 = best_result[:, 2].copy()
            x0 += np.random.uniform(-0.02, 0.02, N)
            y0 += np.random.uniform(-0.02, 0.02, N)
            r0 *= 0.85
            try_anneal_then_optimize(x0, y0, r0, n_anneal_steps=1500)

    # Also try annealing from fresh inits
    for k in range(15):
        if time.time() > deadline - 15:
            break
        x0, y0, r0 = hexagonal_init(N)
        x0 += np.random.uniform(-0.03, 0.03, N)
        y0 += np.random.uniform(-0.03, 0.03, N)
        r0 *= np.random.uniform(0.8, 1.0, N)
        try_anneal_then_optimize(x0, y0, r0, n_anneal_steps=1200)

    for k in range(10):
        if time.time() > deadline - 15:
            break
        x0, y0, r0 = random_init()
        try_anneal_then_optimize(x0, y0, r0, n_anneal_steps=1000)

    # Phase 3: Final high-precision polish on best result
    if best_result is not None and time.time() < deadline - 10:
        x_best = best_result[:, 0].copy()
        y_best = best_result[:, 1].copy()
        r_best = best_result[:, 2].copy()

        # Grow-and-push to maximize radii under strict constraints
        x_gp, y_gp, r_gp = grow_and_push(x_best, y_best, r_best, max_iters=200)
        if check_validity(x_gp, y_gp, r_gp):
            current_sum = np.sum(r_gp)
            if current_sum > best_sum:
                best_sum = current_sum
                best_result = np.column_stack((x_gp, y_gp, r_gp))
                x_best, y_best, r_best = x_gp, y_gp, r_gp

        # High-precision L-BFGS polish with tighter tolerances
        z0 = np.concatenate([x_best, y_best, r_best])
        try:
            res = minimize(objective, z0, method='L-BFGS-B',
                          options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:]
            r_opt = np.maximum(r_opt, 0.001)
            if check_validity(x_opt, y_opt, r_opt):
                x_re, y_re, r_re = radial_expansion(x_opt, y_opt, r_opt, max_rounds=50)
                if check_validity(x_re, y_re, r_re):
                    current_sum = np.sum(r_re)
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_result = np.column_stack((x_re, y_re, r_re))
                else:
                    current_sum = np.sum(r_opt)
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_result = np.column_stack((x_opt, y_opt, r_opt))
            else:
                x_gp2, y_gp2, r_gp2 = grow_and_push(x_opt, y_opt, r_opt, max_iters=100)
                if check_validity(x_gp2, y_gp2, r_gp2):
                    x_re, y_re, r_re = radial_expansion(x_gp2, y_gp2, r_gp2, max_rounds=30)
                    if check_validity(x_re, y_re, r_re):
                        current_sum = np.sum(r_re)
                        if current_sum > best_sum:
                            best_sum = current_sum
                            best_result = np.column_stack((x_re, y_re, r_re))
                    else:
                        current_sum = np.sum(r_gp2)
                        if current_sum > best_sum:
                            best_sum = current_sum
                            best_result = np.column_stack((x_gp2, y_gp2, r_gp2))
        except Exception:
            pass

        # Final micro-polish: iterative single-circle radius maximization with position adjustment
        if best_result is not None and time.time() < deadline - 3:
            x_f = best_result[:, 0].copy()
            y_f = best_result[:, 1].copy()
            r_f = best_result[:, 2].copy()
            for _round in range(8):
                if time.time() > deadline - 1:
                    break
                improved = False
                for i in range(N):
                    # Compute max radius given current positions
                    max_r = min(x_f[i], 1-x_f[i], y_f[i], 1-y_f[i])
                    for j in range(N):
                        if i == j:
                            continue
                        d = np.sqrt((x_f[i]-x_f[j])**2 + (y_f[i]-y_f[j])**2)
                        max_r = min(max_r, d - r_f[j])
                    if max_r > r_f[i] + 1e-9:
                        r_f[i] = max_r - 1e-7
                        improved = True
                if not improved:
                    break
            
            # Try small position perturbations to unlock more growth
            if time.time() < deadline - 1:
                for _attempt in range(10):
                    x_try = x_f.copy()
                    y_try = y_f.copy()
                    # Small random perturbation
                    x_try += np.random.normal(0, 0.002, N)
                    y_try += np.random.normal(0, 0.002, N)
                    # Project to valid region
                    for i in range(N):
                        x_try[i] = np.clip(x_try[i], r_f[i], 1-r_f[i])
                        y_try[i] = np.clip(y_try[i], r_f[i], 1-r_f[i])
                    # Check if this allows more growth
                    r_try = r_f.copy()
                    valid = True
                    for i in range(N):
                        max_r = min(x_try[i], 1-x_try[i], y_try[i], 1-y_try[i])
                        for j in range(N):
                            if i == j:
                                continue
                            d = np.sqrt((x_try[i]-x_try[j])**2 + (y_try[i]-y_try[j])**2)
                            max_r = min(max_r, d - r_try[j])
                        if max_r < r_f[i] - 1e-8:
                            valid = False
                            break
                        r_try[i] = max_r - 1e-7
                    if valid and np.sum(r_try) > np.sum(r_f) + 1e-8:
                        x_f, y_f, r_f = x_try, y_try, r_try
            if check_validity(x_f, y_f, r_f):
                current_sum = np.sum(r_f)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = np.column_stack((x_f, y_f, r_f))

    # Fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best_result

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    best_value = -np.inf
    best_result = None
    
    def evaluate(circles, w, h):
        """Return sum of radii if valid, else -inf."""
        n = len(circles)
        for i in range(n):
            x, y, r = circles[i]
            if x - r < -1e-6 or x + r > w + 1e-6 or y - r < -1e-6 or y + r > h + 1e-6:
                return -np.inf
        for i in range(n):
            for j in range(i + 1, n):
                xi, yi, ri = circles[i]
                xj, yj, rj = circles[j]
                d = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                if d < ri + rj - 1e-6:
                    return -np.inf
        return float(np.sum(circles[:, 2]))
    
    def check_and_update(x, w, h):
        nonlocal best_value, best_result
        if time.time() > deadline:
            return
        xs = x[0::3]
        ys = x[1::3]
        rs = x[2::3]
        circles = np.column_stack([xs, ys, rs])
        val = evaluate(circles, w, h)
        if val > best_value:
            best_value = val
            best_result = x.copy()
    
    def softplus(x, beta=100.0):
        """Numerically stable softplus."""
        x_beta = beta * x
        result = np.where(x_beta > 0,
                         x + np.log1p(np.exp(-x_beta)) / beta,
                         np.log1p(np.exp(x_beta)) / beta)
        return result
    
    def softplus_grad(x, beta=100.0):
        """Gradient of softplus: sigmoid(beta*x)."""
        return 1.0 / (1.0 + np.exp(-beta * x))
    
    def objective_and_grad(x, w, h, beta=100.0):
        """
        Force-directed penalty objective with analytical gradients (vectorized).
        x: [x0, y0, r0, x1, y1, r1, ...]
        """
        c = x.reshape(-1, 3)
        n = len(c)
        xs = c[:, 0]
        ys = c[:, 1]
        rs = c[:, 2]
        
        obj = -np.sum(rs)
        grad = np.zeros_like(x)
        
        # Boundary penalties (vectorized)
        # x - r < 0
        v1 = rs - xs
        obj += np.sum(softplus(v1, beta))
        g1 = softplus_grad(v1, beta)
        grad[0::3] -= g1
        grad[2::3] += g1
        
        # x + r > w
        v2 = xs + rs - w
        obj += np.sum(softplus(v2, beta))
        g2 = softplus_grad(v2, beta)
        grad[0::3] += g2
        grad[2::3] += g2
        
        # y - r < 0
        v3 = rs - ys
        obj += np.sum(softplus(v3, beta))
        g3 = softplus_grad(v3, beta)
        grad[1::3] -= g3
        grad[2::3] += g3
        
        # y + r > h
        v4 = ys + rs - h
        obj += np.sum(softplus(v4, beta))
        g4 = softplus_grad(v4, beta)
        grad[1::3] += g4
        grad[2::3] += g4
        
        # r < 0
        v5 = -rs
        obj += np.sum(softplus(v5, beta))
        g5 = softplus_grad(v5, beta)
        grad[2::3] -= g5
        
        # Pairwise overlap penalties (vectorized)
        for i in range(n):
            dx = xs[i] - xs[i+1:]
            dy = ys[i] - ys[i+1:]
            d = np.sqrt(dx * dx + dy * dy + 1e-12)
            
            v = rs[i] + rs[i+1:] - d
            obj += np.sum(softplus(v, beta))
            g = softplus_grad(v, beta)
            
            # Gradient contribution to circle i
            grad[3*i] += -np.sum(g * dx / d)
            grad[3*i+1] += -np.sum(g * dy / d)
            grad[3*i+2] += np.sum(g)
            
            # Gradient contribution to circles j > i
            idx = np.arange(i+1, n)
            grad[3*idx] += g * dx / d
            grad[3*idx+1] += g * dy / d
            grad[3*idx+2] += g
        
        # Gradient of -sum(r)
        grad[2::3] += -1.0
        
        return obj, grad
    
    def optimize_from(circles0, w, h, max_iter=400, beta=100.0, ftol=1e-12, gtol=1e-8):
        """L-BFGS-B optimization with softplus penalties."""
        x0 = circles0.flatten()
        
        bounds = []
        r_max = min(w, h) / 2.0
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((1e-6, r_max))
        
        try:
            res = minimize(
                lambda p: objective_and_grad(p, w, h, beta),
                x0,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
            )
            return res.x
        except Exception:
            return x0

    def equalization_polish(x_best, w, h, max_iter=300, beta=150.0):
        """
        Equalization polish: add a soft-max penalty on radii to encourage
        more uniform circle sizes, which can find tighter packings.
        """
        x0 = x_best.copy()
        n = N
        
        def eq_obj_and_grad(p):
            c = p.reshape(-1, 3)
            rs = c[:, 2]
            
            base_obj, base_grad = objective_and_grad(p, w, h, beta)
            
            r_mean = np.mean(rs)
            margin = 0.01
            beta_eq = 80.0
            violations = rs - r_mean - margin
            eq_penalty = np.sum(softplus(violations, beta_eq))
            eq_grad = softplus_grad(violations, beta_eq)
            
            grad = base_grad.copy()
            grad[2::3] += eq_grad
            
            obj = base_obj + eq_penalty
            return obj, grad
        
        bounds = []
        r_max = min(w, h) / 2.0
        for i in range(n):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((1e-6, r_max))
        
        try:
            res = minimize(
                eq_obj_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-13, 'gtol': 1e-10}
            )
            return res.x
        except Exception:
            return x0
    
    def contact_graph_polish(x_best, w, h, max_iter=500, beta=2000.0):
        """
        High-beta polish focused on contact constraints. Uses very high beta
        to nearly enforce exact non-overlap, then polishes positions.
        """
        x0 = x_best.copy()
        n = N
        
        def cg_obj_and_grad(p):
            c = p.reshape(-1, 3)
            xs = c[:, 0]
            ys = c[:, 1]
            rs = c[:, 2]
            
            obj = -np.sum(rs)
            grad = np.zeros_like(p)
            
            # Boundary penalties with high beta
            v1 = rs - xs
            obj += np.sum(softplus(v1, beta))
            g1 = softplus_grad(v1, beta)
            grad[0::3] -= g1
            grad[2::3] += g1
            
            v2 = xs + rs - w
            obj += np.sum(softplus(v2, beta))
            g2 = softplus_grad(v2, beta)
            grad[0::3] += g2
            grad[2::3] += g2
            
            v3 = rs - ys
            obj += np.sum(softplus(v3, beta))
            g3 = softplus_grad(v3, beta)
            grad[1::3] -= g3
            grad[2::3] += g3
            
            v4 = ys + rs - h
            obj += np.sum(softplus(v4, beta))
            g4 = softplus_grad(v4, beta)
            grad[1::3] += g4
            grad[2::3] += g4
            
            v5 = -rs
            obj += np.sum(softplus(v5, beta))
            g5 = softplus_grad(v5, beta)
            grad[2::3] -= g5
            
            # Pairwise overlap with very high beta
            for i in range(n):
                dx = xs[i] - xs[i+1:]
                dy = ys[i] - ys[i+1:]
                d = np.sqrt(dx * dx + dy * dy + 1e-12)
                
                v = rs[i] + rs[i+1:] - d
                obj += np.sum(softplus(v, beta))
                g = softplus_grad(v, beta)
                
                grad[3*i] += -np.sum(g * dx / d)
                grad[3*i+1] += -np.sum(g * dy / d)
                grad[3*i+2] += np.sum(g)
                
                idx = np.arange(i+1, n)
                grad[3*idx] += g * dx / d
                grad[3*idx+1] += g * dy / d
                grad[3*idx+2] += g
            
            grad[2::3] += -1.0
            return obj, grad
        
        bounds = []
        r_max = min(w, h) / 2.0
        for i in range(n):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((1e-6, r_max))
        
        try:
            res = minimize(
                cg_obj_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-11}
            )
            return res.x
        except Exception:
            return x0
    
    def force_directed_refine(x_best, w, h, n_steps=200, dt=0.002):
        """
        Physics-based force-directed refinement. Applies repulsive forces
        between overlapping circles and attractive forces toward larger radii.
        Can escape local minima that gradient methods get stuck in.
        """
        x = x_best.copy()
        n = N
        xs = x[0::3].copy()
        ys = x[1::3].copy()
        rs = x[2::3].copy()
        
        for step in range(n_steps):
            fx = np.zeros(n)
            fy = np.zeros(n)
            fr = np.zeros(n)
            
            # Repulsive forces between overlapping/near circles
            for i in range(n):
                for j in range(i + 1, n):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    d = np.sqrt(dx * dx + dy * dy + 1e-12)
                    overlap = rs[i] + rs[j] - d
                    if overlap > -0.05:  # Apply force if close
                        force = overlap * 50.0  # Spring constant
                        if d > 1e-8:
                            fx[i] += force * dx / d
                            fy[i] += force * dy / d
                            fx[j] -= force * dx / d
                            fy[j] -= force * dy / d
                        fr[i] += force * 0.5
                        fr[j] += force * 0.5
            
            # Boundary repulsion
            for i in range(n):
                if xs[i] - rs[i] < 0.02:
                    fx[i] += (0.02 - (xs[i] - rs[i])) * 100.0
                if xs[i] + rs[i] > w - 0.02:
                    fx[i] -= (xs[i] + rs[i] - (w - 0.02)) * 100.0
                if ys[i] - rs[i] < 0.02:
                    fy[i] += (0.02 - (ys[i] - rs[i])) * 100.0
                if ys[i] + rs[i] > h - 0.02:
                    fy[i] -= (ys[i] + rs[i] - (h - 0.02)) * 100.0
            
            # Grow radii (attractive force)
            for i in range(n):
                fr[i] += 0.1  # Gentle growth pressure
            
            # Update positions with damping
            damping = 0.95
            for i in range(n):
                xs[i] += dt * fx[i] * damping
                ys[i] += dt * fy[i] * damping
                rs[i] += dt * fr[i] * 0.5 * damping
                
                # Clamp
                rs[i] = max(rs[i], 1e-6)
                rs[i] = min(rs[i], min(w, h) / 2)
                xs[i] = np.clip(xs[i], rs[i], w - rs[i])
                ys[i] = np.clip(ys[i], rs[i], h - rs[i])
            
            x[0::3] = xs
            x[1::3] = ys
            x[2::3] = rs
        
        return x
    
    def aspect_refine(x_best, w, h, n_trials=6, max_iter=300):
        """
        Refine aspect ratio around the current best by trying small perturbations.
        """
        best_x = x_best.copy()
        best_val = evaluate(x_best.reshape(-1, 3), w, h)
        
        for trial in range(n_trials):
            dw = (trial - n_trials // 2) * 0.02
            new_w = w + dw
            new_h = 2.0 - new_w  # Perimeter constraint: 2*(w+h)=4 => h=2-w
            
            if new_h <= 0.1 or new_w <= 0.1:
                continue
            
            # Scale positions to new aspect
            scale_x = new_w / w
            scale_y = new_h / h
            x_new = best_x.copy()
            x_new[0::3] = x_new[0::3] * scale_x
            x_new[1::3] = x_new[1::3] * scale_y
            # Scale radii by min of scales
            scale_r = min(scale_x, scale_y)
            x_new[2::3] = x_new[2::3] * scale_r
            
            # Polish
            x_polished = optimize_from(x_new, new_w, new_h, max_iter=max_iter, beta=100.0)
            val = evaluate(x_polished.reshape(-1, 3), new_w, new_h)
            if val > best_val:
                best_val = val
                best_x = x_polished.copy()
        
        return best_x, best_val
    
    def polish(circles0, w, h, max_iter=500, beta=200.0):
        """High-precision polish pass with tighter tolerance and higher beta."""
        return optimize_from(circles0, w, h, max_iter=max_iter, beta=beta, ftol=1e-14, gtol=1e-10)
    
    def hex_lattice_init(w, h):
        """
        Generate hexagonal lattice initialization for N=21 circles.
        Tries multiple row configurations and returns best initial config.
        """
        best_circles = None
        best_val = -np.inf
        
        # Row configurations summing to 21
        row_configs = [
            [3, 4, 4, 4, 3, 3],
            [4, 4, 4, 4, 3, 2],
            [2, 4, 4, 4, 4, 3],
            [3, 4, 4, 4, 4, 2],
            [4, 3, 4, 4, 4, 2],
            [2, 3, 4, 4, 4, 4],
            [3, 3, 4, 4, 4, 3],
            [4, 4, 3, 4, 4, 2],
            [2, 4, 3, 4, 4, 4],
            [3, 4, 3, 4, 4, 3],
            [5, 4, 4, 4, 4],
            [4, 5, 4, 4, 4],
            [3, 5, 4, 4, 5],
            [4, 4, 5, 4, 4],
            [2, 5, 4, 5, 5],
            [5, 5, 5, 5, 1],
            [1, 5, 5, 5, 5],
            [3, 4, 5, 4, 5],
            [5, 4, 5, 4, 3],
        ]
        
        for row_counts in row_configs:
            if sum(row_counts) != N:
                continue
            
            n_rows = len(row_counts)
            max_cols = max(row_counts)
            
            # Estimate radius from available space
            est_r_h = w / (max_cols + 0.5)
            est_r_v = h / (n_rows * np.sqrt(3) / 2 + 1.0)
            r_est = min(est_r_h, est_r_v) * 0.95
            
            if r_est < 0.01:
                continue
            
            circles = []
            for row_idx, n_cols in enumerate(row_counts):
                y = (row_idx + 0.5) * r_est * np.sqrt(3)
                x_offset = r_est if row_idx % 2 == 1 else 0.0
                row_width = n_cols * 2 * r_est
                center_x = w / 2
                for col_idx in range(n_cols):
                    x = center_x - row_width / 2 + x_offset + (col_idx + 0.5) * 2 * r_est
                    circles.append([x, y, r_est])
            
            circles = np.array(circles)
            val = evaluate(circles, w, h)
            
            if val > best_val:
                best_val = val
                best_circles = circles.copy()
        
        if best_circles is None:
            a = int(np.ceil(np.sqrt(N * w / h)))
            b = int(np.ceil(N / a))
            while a * b < N:
                b += 1
            circles = []
            for i in range(N):
                col = i % a
                row = i // a
                r = min(w / a, h / b) * 0.45
                x = (col + 0.5) * w / a
                y = (row + 0.5) * h / b
                circles.append([x, y, r])
            best_circles = np.array(circles)
        
        return best_circles
    
    def grid_init(w, h):
        """Simple grid initialization."""
        a = int(np.ceil(np.sqrt(N * w / h)))
        b = int(np.ceil(N / a))
        while a * b < N:
            b += 1
        circles = []
        for i in range(N):
            col = i % a
            row = i // a
            r = min(w / (a + 0.5), h / (b + 0.5)) * 0.95
            x = (col + 0.5) * w / a
            y = (row + 0.5) * h / b
            circles.append([x, y, r])
        return np.array(circles)
    
    def random_init(w, h, rng, n_attempts=20):
        """Generate random initial configurations and return the best."""
        best_circles = None
        best_val = -np.inf
        
        for _ in range(n_attempts):
            circles = []
            for _ in range(N):
                r = rng.uniform(0.02, 0.15)
                x = rng.uniform(r, w - r)
                y = rng.uniform(r, h - r)
                circles.append([x, y, r])
            circles = np.array(circles)
            val = evaluate(circles, w, h)
            if val > best_val:
                best_val = val
                best_circles = circles.copy()
        
        return best_circles if best_circles is not None else grid_init(w, h)
    
    # Try multiple aspect ratios
    aspects = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0,
               1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5]
    
    rng = np.random.default_rng(42)
    
    # Track best aspect for final refinement
    best_aspect_w = 1.0
    best_aspect_h = 1.0
    
    for aspect in aspects:
        if time.time() > deadline - 5:
            break
        
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)
        
        # Try multiple initializations
        inits = [
            hex_lattice_init(w, h),
            grid_init(w, h),
            random_init(w, h, rng, n_attempts=10)
        ]
        
        best_x_for_aspect = None
        best_val_for_aspect = -np.inf
        
        for circles0 in inits:
            if time.time() > deadline - 8:
                break
            
            # Check initial config
            val0 = evaluate(circles0, w, h)
            if val0 > best_val_for_aspect:
                best_val_for_aspect = val0
                best_x_for_aspect = circles0.flatten().copy()
            
            # Phase 1: Coarse optimization
            if time.time() < deadline - 10:
                x1 = optimize_from(circles0, w, h, max_iter=250, beta=50.0)
                val1 = evaluate(x1.reshape(-1, 3), w, h)
                if val1 > best_val_for_aspect:
                    best_val_for_aspect = val1
                    best_x_for_aspect = x1.copy()
            
            # Phase 2: Medium optimization
            if time.time() < deadline - 8:
                x2 = optimize_from(x1, w, h, max_iter=250, beta=100.0)
                val2 = evaluate(x2.reshape(-1, 3), w, h)
                if val2 > best_val_for_aspect:
                    best_val_for_aspect = val2
                    best_x_for_aspect = x2.copy()
        
        if best_x_for_aspect is None:
            continue
        
        # Phase 3: Polish the best for this aspect
        if time.time() < deadline - 5:
            x3 = polish(best_x_for_aspect, w, h, max_iter=400, beta=200.0)
            check_and_update(x3, w, h)
        
        # Phase 3b: Equalization polish
        if time.time() < deadline - 4:
            x3b = equalization_polish(x3, w, h, max_iter=250, beta=150.0)
            val3b = evaluate(x3b.reshape(-1, 3), w, h)
            if val3b > best_value:
                best_value = val3b
                best_result = x3b.copy()
                best_aspect_w = w
                best_aspect_h = h
            if time.time() < deadline - 3.5:
                x3b2 = equalization_polish(best_x_for_aspect, w, h, max_iter=200, beta=120.0)
                val3b2 = evaluate(x3b2.reshape(-1, 3), w, h)
                if val3b2 > best_value:
                    best_value = val3b2
                    best_result = x3b2.copy()
                    best_aspect_w = w
                    best_aspect_h = h
        
        # Phase 3c: Force-directed refinement (new)
        if time.time() < deadline - 4:
            x3c = force_directed_refine(x3, w, h, n_steps=150, dt=0.003)
            val3c = evaluate(x3c.reshape(-1, 3), w, h)
            if val3c > best_value:
                best_value = val3c
                best_result = x3c.copy()
                best_aspect_w = w
                best_aspect_h = h
            # Polish after force-directed
            if time.time() < deadline - 3.5:
                x3c_pol = polish(x3c, w, h, max_iter=300, beta=200.0)
                check_and_update(x3c_pol, w, h)
        
        # Phase 4: Perturbed restart
        if time.time() < deadline - 3:
            x4 = best_x_for_aspect.copy()
            for i in range(N):
                x4[3 * i] += rng.normal(0, 0.008)
                x4[3 * i + 1] += rng.normal(0, 0.008)
                x4[3 * i + 2] += rng.normal(0, 0.004)
            for i in range(N):
                r = max(x4[3 * i + 2], 1e-6)
                x4[3 * i] = np.clip(x4[3 * i], r, w - r)
                x4[3 * i + 1] = np.clip(x4[3 * i + 1], r, h - r)
                x4[3 * i + 2] = np.clip(x4[3 * i + 2], 1e-6, min(w, h) / 2)
            
            x4 = polish(x4, w, h, max_iter=300, beta=200.0)
            check_and_update(x4, w, h)
            
            if time.time() < deadline - 2.5:
                x4b = equalization_polish(x4, w, h, max_iter=200, beta=150.0)
                val4b = evaluate(x4b.reshape(-1, 3), w, h)
                if val4b > best_value:
                    best_value = val4b
                    best_result = x4b.copy()
                    best_aspect_w = w
                    best_aspect_h = h
        
        # Phase 5: Ultra-high-precision final polish
        if time.time() < deadline - 2:
            x5 = optimize_from(best_x_for_aspect, w, h, max_iter=400, beta=500.0, ftol=1e-15, gtol=1e-12)
            check_and_update(x5, w, h)
    
    # Fallback if no valid result found
    if best_result is None:
        a, b = 7, 3
        w, h = 1.4, 0.6
        r = 0.099
        best_result = np.zeros(3 * N)
        for i in range(a):
            for j in range(b):
                idx = i * b + j
                best_result[3 * idx] = (i + 0.5) * w / a
                best_result[3 * idx + 1] = (j + 0.5) * h / b
                best_result[3 * idx + 2] = r
    
    # Final high-precision contact-graph polish on the best result
    if best_result is not None and time.time() < deadline - 2:
        x_cg = contact_graph_polish(best_result, best_aspect_w, best_aspect_h, max_iter=400, beta=3000.0)
        val_cg = evaluate(x_cg.reshape(-1, 3), best_aspect_w, best_aspect_h)
        if val_cg > best_value:
            best_value = val_cg
            best_result = x_cg.copy()
    
    # Aspect ratio refinement around best
    if best_result is not None and time.time() < deadline - 3:
        x_ar, val_ar = aspect_refine(best_result, best_aspect_w, best_aspect_h, n_trials=5, max_iter=250)
        if val_ar > best_value:
            best_value = val_ar
            best_result = x_ar.copy()
    
    # Final ultra-polish
    if best_result is not None and time.time() < deadline - 1:
        x_final = polish(best_result, best_aspect_w, best_aspect_h, max_iter=300, beta=500.0)
        val_final = evaluate(x_final.reshape(-1, 3), best_aspect_w, best_aspect_h)
        if val_final > best_value:
            best_value = val_final
            best_result = x_final.copy()
    
    # Format output: (x, y, r) for each circle
    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = best_result[3 * i:3 * i + 3]
    
    return result
# EVOLVE-BLOCK-END

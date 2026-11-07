"""Rewritten program: Powell's method with Nelder-Mead polish for BBOB f22."""
import numpy as np


class Optimizer:
    """Deterministic gradient-free trust-region method using Powell's conjugate
    direction search with Nelder-Mead polish for rapid local exploitation."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        calls_made = 0

        def wrapped_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            return float(func(np.asarray(x, dtype=float)))

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = np.array(x, dtype=float)

        # ---- Phase 1: Coarse grid scan to quickly find promising region ----
        # Use a stratified grid to cover the domain efficiently
        # 3 levels per dimension = 3^5 = 243 points, but we'll use a subset
        # Actually, let's use a smarter approach: evaluate at key points first
        
        # Start with origin and axis-aligned points
        start_points = [np.zeros(dim)]
        
        # Add points along each axis at several distances
        for i in range(dim):
            for dist in [1.0, 2.0, 3.0, -1.0, -2.0, -3.0]:
                p = np.zeros(dim)
                p[i] = dist
                start_points.append(p)
        
        # Add some random points for diversity
        rng = np.random.default_rng(12345)
        for _ in range(10):
            start_points.append(rng.uniform(-4, 4, dim))
        
        # Evaluate all start points
        for sp in start_points:
            if calls_made >= budget:
                break
            f_sp = wrapped_func(sp)
            update_best(sp, f_sp)
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped_func(best_x)
        
        # ---- Phase 2: Nelder-Mead on the best point found so far ----
        nm_budget = max(50, (budget - calls_made) // 3)
        
        simplex = np.zeros((dim + 1, dim))
        simplex[0] = best_x.copy()
        # Adaptive initial simplex size based on function landscape
        init_size = 0.5
        for i in range(dim):
            simplex[i + 1] = best_x.copy()
            simplex[i + 1][i] += init_size
        
        f_vals = np.zeros(dim + 1)
        f_vals[0] = best_f
        for i in range(1, dim + 1):
            f_vals[i] = wrapped_func(simplex[i])
            update_best(simplex[i], f_vals[i])
            if calls_made >= budget:
                break
        
        if calls_made < budget:
            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            for _ in range(nm_budget):
                if calls_made >= budget:
                    break

                order = np.argsort(f_vals)
                simplex = simplex[order]
                f_vals = f_vals[order]

                if f_vals[-1] - f_vals[0] < 1e-12:
                    break

                centroid = np.mean(simplex[:-1], axis=0)
                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = wrapped_func(x_r)
                update_best(x_r, f_r)
                if calls_made >= budget:
                    break

                if f_vals[0] <= f_r < f_vals[-2]:
                    simplex[-1] = x_r
                    f_vals[-1] = f_r
                elif f_r < f_vals[0]:
                    x_e = centroid + gamma * (x_r - centroid)
                    f_e = wrapped_func(x_e)
                    update_best(x_e, f_e)
                    if calls_made >= budget:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_vals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_vals[-1] = f_r
                else:
                    if f_r < f_vals[-1]:
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        x_c = centroid - rho * (simplex[-1] - centroid)
                    f_c = wrapped_func(x_c)
                    update_best(x_c, f_c)
                    if calls_made >= budget:
                        break
                    if f_c < min(f_r, f_vals[-1]):
                        simplex[-1] = x_c
                        f_vals[-1] = f_c
                    else:
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_vals[i] = wrapped_func(simplex[i])
                            update_best(simplex[i], f_vals[i])
                            if calls_made >= budget:
                                break
        
        # ---- Phase 3: Powell's conjugate direction method ----
        def powell_method(x0, f0, max_calls):
            """Powell's method with adaptive direction updates and efficient line search."""
            x = np.array(x0, dtype=float)
            f = f0
            # Initial directions: standard basis
            dirs = [np.eye(dim, dtype=float)[i].copy() for i in range(dim)]
            n_powell = 0

            while n_powell < max_calls:
                if calls_made >= budget:
                    break

                x_old = x.copy()
                f_old = f
                min_decrease = 0.0
                min_dir_idx = -1

                # Line search along each direction
                for i in range(dim):
                    if calls_made >= budget:
                        break
                    d = dirs[i]
                    
                    # Efficient line search: try a few steps and pick the best
                    # Use adaptive step size based on previous improvements
                    steps = [0.1, 0.2, 0.5, 1.0, -0.1, -0.2, -0.5, -1.0]
                    best_t = 0.0
                    best_f_local = f
                    
                    for t in steps:
                        if calls_made >= budget:
                            break
                        x_trial = x + t * d
                        f_trial = wrapped_func(x_trial)
                        if f_trial < best_f_local:
                            best_f_local = f_trial
                            best_t = t
                    
                    # If we found improvement, do a finer search around best_t
                    if best_t != 0.0 and best_f_local < f:
                        # Fine search in a small window around best_t
                        t_center = best_t
                        t_width = abs(best_t) * 0.5 if best_t != 0 else 0.1
                        fine_steps = np.linspace(t_center - t_width, t_center + t_width, 5)
                        for t in fine_steps:
                            if calls_made >= budget:
                                break
                            x_trial = x + t * d
                            f_trial = wrapped_func(x_trial)
                            if f_trial < best_f_local:
                                best_f_local = f_trial
                                best_t = t
                    
                    if best_f_local < f:
                        x = x + best_t * d
                        f = best_f_local
                        decrease = f_old - f
                        if decrease > min_decrease:
                            min_decrease = decrease
                            min_dir_idx = i

                if calls_made >= budget:
                    break

                # Update directions
                if min_dir_idx >= 0:
                    # Replace the direction that gave the most decrease with the
                    # direction from x_old to x
                    new_dir = x - x_old
                    norm_new = np.linalg.norm(new_dir)
                    if norm_new > 1e-10:
                        new_dir = new_dir / norm_new
                    # Shift directions: move the best direction to the end
                    dirs = dirs[:min_dir_idx] + dirs[min_dir_idx + 1:] + [new_dir]
                else:
                    # No improvement, stop
                    break

                # Check convergence
                if min_decrease < 1e-12:
                    break

                n_powell += 1

            return x, f

        remaining = budget - calls_made
        if remaining > 2 * (dim + 1):
            x_pw, f_pw = powell_method(best_x.copy(), best_f, remaining)
            if f_pw < best_f:
                best_f = f_pw
                best_x = x_pw.copy()

        # ---- Phase 4: Final Nelder-Mead polish ----
        if calls_made < budget and best_x is not None:
            polish_budget = budget - calls_made
            if polish_budget >= 2 * (dim + 1):
                simplex = np.zeros((dim + 1, dim))
                simplex[0] = best_x.copy()
                # Smaller initial simplex for polishing
                polish_size = 0.1
                for i in range(dim):
                    simplex[i + 1] = best_x.copy()
                    simplex[i + 1][i] += polish_size

                f_vals = np.zeros(dim + 1)
                f_vals[0] = best_f
                for i in range(1, dim + 1):
                    f_vals[i] = wrapped_func(simplex[i])
                    update_best(simplex[i], f_vals[i])
                    if calls_made >= budget:
                        break

                alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
                for _ in range(polish_budget):
                    if calls_made >= budget:
                        break

                    order = np.argsort(f_vals)
                    simplex = simplex[order]
                    f_vals = f_vals[order]

                    if f_vals[-1] - f_vals[0] < 1e-14:
                        break

                    centroid = np.mean(simplex[:-1], axis=0)
                    x_r = centroid + alpha * (centroid - simplex[-1])
                    f_r = wrapped_func(x_r)
                    update_best(x_r, f_r)
                    if calls_made >= budget:
                        break

                    if f_vals[0] <= f_r < f_vals[-2]:
                        simplex[-1] = x_r
                        f_vals[-1] = f_r
                    elif f_r < f_vals[0]:
                        x_e = centroid + gamma * (x_r - centroid)
                        f_e = wrapped_func(x_e)
                        update_best(x_e, f_e)
                        if calls_made >= budget:
                            break
                        if f_e < f_r:
                            simplex[-1] = x_e
                            f_vals[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            f_vals[-1] = f_r
                    else:
                        if f_r < f_vals[-1]:
                            x_c = centroid + rho * (x_r - centroid)
                        else:
                            x_c = centroid - rho * (simplex[-1] - centroid)
                        f_c = wrapped_func(x_c)
                        update_best(x_c, f_c)
                        if calls_made >= budget:
                            break
                        if f_c < min(f_r, f_vals[-1]):
                            simplex[-1] = x_c
                            f_vals[-1] = f_c
                        else:
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                f_vals[i] = wrapped_func(simplex[i])
                                update_best(simplex[i], f_vals[i])
                                if calls_made >= budget:
                                    break

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped_func(best_x)

        return best_x, best_f

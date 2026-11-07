import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0  # Safety margin before 300s limit
        calls = 0
        best_x = None
        best_f = float('inf')
        rng = np.random.default_rng(42)

        def safe_call(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return None
            calls += 1
            try:
                x_clipped = np.clip(x, -5.0, 5.0)
                f_val = func(x_clipped)
                if f_val is None or np.isnan(f_val) or np.isinf(f_val):
                    return None
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_clipped.copy()
                return f_val
            except Exception:
                return None

        # Phase 1: Global Exploration
        n_global = 15
        for _ in range(n_global):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, size=dim)
            safe_call(x)

        if best_x is None:
            x0 = np.zeros(dim)
            safe_call(x0)
            return best_x, best_f

        # Nelder-Mead implementation for refinement
        def nelder_mead(x0, f0, max_iter=30):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return

            # Initialize simplex
            simplex = [x0.copy()]
            f_values = [f0]

            # Create other vertices by perturbing each coordinate
            for i in range(dim):
                x_new = x0.copy()
                x_new[i] += 0.05
                simplex.append(x_new)
                f_val = safe_call(x_new)
                if f_val is None:
                    return
                f_values.append(f_val)

            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            for iteration in range(max_iter):
                if calls >= budget or time.time() > deadline:
                    return

                # Sort simplex by function value
                order = np.argsort(f_values)
                simplex = [simplex[i] for i in order]
                f_values = [f_values[i] for i in order]

                # Check for convergence
                if abs(f_values[0] - f_values[-1]) < 1e-12:
                    break

                # Compute centroid of all but the worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = safe_call(x_r)
                if f_r is None:
                    return

                if f_values[0] <= f_r < f_values[-2]:
                    simplex[-1] = x_r
                    f_values[-1] = f_r
                elif f_r < f_values[0]:
                    x_e = centroid + gamma * (x_r - centroid)
                    f_e = safe_call(x_e)
                    if f_e is None:
                        return
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                else:
                    if f_r < f_values[-1]:
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        x_c = centroid - rho * (simplex[-1] - centroid)
                    f_c = safe_call(x_c)
                    if f_c is None:
                        return

                    if f_c < min(f_r, f_values[-1]):
                        simplex[-1] = x_c
                        f_values[-1] = f_c
                    else:
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_val = safe_call(simplex[i])
                            if f_val is None:
                                return
                            f_values[i] = f_val

                if f_values[0] < best_f:
                    best_f = f_values[0]
                    best_x = simplex[0].copy()

        # Adaptive Coordinate Descent
        def coord_descent(x_start, max_rounds=30):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return

            x = x_start.copy()
            f_curr = safe_call(x)
            if f_curr is None:
                return

            # Per-axis step sizes, initialized adaptively
            steps = np.ones(dim) * 0.5
            min_step = 1e-6
            max_step = 10.0

            for rnd in range(max_rounds):
                if calls >= budget or time.time() > deadline:
                    return

                improved = False
                for i in range(dim):
                    if calls >= budget or time.time() > deadline:
                        return

                    # 1D line search along axis i
                    for direction in [1, -1]:
                        step = steps[i]
                        x_trial = x.copy()
                        x_trial[i] += direction * step
                        x_trial = np.clip(x_trial, -5.0, 5.0)
                        f_trial = safe_call(x_trial)
                        if f_trial is None:
                            return

                        if f_trial < f_curr:
                            # Move in this direction
                            x = x_trial
                            f_curr = f_trial
                            improved = True
                            # Grow step
                            steps[i] = min(steps[i] * 1.5, max_step)
                            # Continue searching in same direction
                            while calls < budget and time.time() < deadline:
                                x_trial = x.copy()
                                x_trial[i] += direction * step
                                x_trial = np.clip(x_trial, -5.0, 5.0)
                                f_trial = safe_call(x_trial)
                                if f_trial is None:
                                    break
                                if f_trial < f_curr:
                                    x = x_trial
                                    f_curr = f_trial
                                    improved = True
                                    steps[i] = min(steps[i] * 1.3, max_step)
                                else:
                                    break
                        else:
                            # Shrink step if no improvement
                            steps[i] = max(steps[i] * 0.5, min_step)

                # Update global best
                if f_curr < best_f:
                    best_f = f_curr
                    best_x = x.copy()

                # If no improvement in this round, check convergence
                if not improved:
                    # Try smaller steps
                    steps *= 0.5
                    if np.min(steps) < min_step:
                        break

        # Phase 2: Local Refinement with Adaptive Coordinate Descent
        # This exploits the anisotropic ridge geometry of f13
        coord_descent(best_x.copy(), max_rounds=30)

        # Phase 3: Nelder-Mead refinement to polish the solution
        nelder_mead(best_x.copy(), best_f, max_iter=30)

        # Phase 4: If budget remains, do a few more coordinate descent passes
        remaining_budget = budget - calls
        if remaining_budget > 50 and time.time() < deadline:
            coord_descent(best_x.copy(), max_rounds=20)
            nelder_mead(best_x.copy(), best_f, max_iter=20)

        # Phase 5: Final Nelder-Mead polish
        if calls < budget and time.time() < deadline:
            nelder_mead(best_x.copy(), best_f, max_iter=20)

        return best_x, best_f

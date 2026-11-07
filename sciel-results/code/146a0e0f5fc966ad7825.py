"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead with adaptive restarts and deadline guard for sharp ridge f13."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # Safety margin before 300s limit
        rng = np.random.default_rng(42)

        # Safe wrapper for function calls
        call_count = [0]

        def safe_call(x):
            call_count[0] += 1
            if call_count[0] > budget:
                return float('inf')
            try:
                val = func(np.clip(x, -5, 5))
                return float(val)
            except Exception:
                return float('inf')

        def nelder_mead(x0, fx0, max_iter, sigma_init=1.0):
            """Nelder-Mead simplex method."""
            dim = len(x0)
            # Build initial simplex
            simplex = [x0.copy() + fx0 * 0]
            fx_vals = [fx0]
            for i in range(dim):
                dx = x0.copy()
                dx[i] += sigma_init
                simplex.append(np.clip(dx, -5, 5))
                fx_vals.append(safe_call(simplex[-1]))

            # Sort by function value
            order = np.argsort(fx_vals)
            simplex = [simplex[i] for i in order]
            fx_vals = [fx_vals[i] for i in order]

            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink

            for _ in range(max_iter):
                if time.time() > deadline - 5:
                    break

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                fx_r = safe_call(x_r)

                if fx_r < fx_vals[0]:
                    # Try expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    fx_e = safe_call(x_e)
                    if fx_e < fx_r:
                        simplex[-1] = x_e
                        fx_vals[-1] = fx_e
                    else:
                        simplex[-1] = x_r
                        fx_vals[-1] = fx_r
                elif fx_r < fx_vals[-2]:
                    simplex[-1] = x_r
                    fx_vals[-1] = fx_r
                else:
                    # Contraction
                    if fx_r < fx_vals[-1]:
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        x_c = centroid - rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    fx_c = safe_call(x_c)
                    if fx_c < fx_vals[-1]:
                        simplex[-1] = x_c
                        fx_vals[-1] = fx_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fx_vals[i] = safe_call(simplex[i])

                # Re-sort
                order = np.argsort(fx_vals)
                simplex = [simplex[i] for i in order]
                fx_vals = [fx_vals[i] for i in order]

            return simplex[0], fx_vals[0]

        # Phase 1: Global search with multiple restarts
        best_x = None
        best_fx = float('inf')
        num_restarts = 5
        calls_per_restart = max(10, budget // (num_restarts + 2))

        for r in range(num_restarts):
            if time.time() > deadline - 30:
                break
            if call_count[0] >= budget:
                break

            # Random starting point, biased toward center for ridge
            x0 = rng.uniform(-3, 3, dim)
            fx0 = safe_call(x0)
            if fx0 < best_fx:
                best_x = x0.copy()
                best_fx = fx0

            # Run Nelder-Mead from this point
            x_opt, fx_opt = nelder_mead(x0, fx0, max_iter=80, sigma_init=1.5)
            if fx_opt < best_fx:
                best_x = x_opt.copy()
                best_fx = fx_opt

        # Phase 2: Local refinement around best point
        if best_x is not None and time.time() < deadline - 20 and call_count[0] < budget - 10:
            # Adaptive Gaussian local search
            sigma = 0.5
            for _ in range(30):
                if time.time() > deadline - 5:
                    break
                if call_count[0] >= budget - 5:
                    break

                x_trial = best_x + sigma * rng.standard_normal(dim)
                x_trial = np.clip(x_trial, -5, 5)
                fx_trial = safe_call(x_trial)

                if fx_trial < best_fx:
                    best_x = x_trial
                    best_fx = fx_trial
                    sigma *= 1.2
                else:
                    sigma *= 0.7

        # Phase 3: Final fine-tuning
        if time.time() < deadline - 5 and call_count[0] < budget - 5:
            sigma = 0.1
            for _ in range(20):
                if time.time() > deadline - 2:
                    break
                if call_count[0] >= budget:
                    break

                x_trial = best_x + sigma * rng.standard_normal(dim)
                x_trial = np.clip(x_trial, -5, 5)
                fx_trial = safe_call(x_trial)

                if fx_trial < best_fx:
                    best_x = x_trial
                    best_fx = fx_trial

        return best_x, best_fx
# EVOLVE-BLOCK-END

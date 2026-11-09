import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        calls = 0

        def try_call(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def run_asa(x_start, n_steps, rng_asa):
            nonlocal best_x, best_f, calls

            x = x_start.copy()
            fx = try_call(x)
            if fx is None:
                return x, fx

            # Temperature schedule: start high, decay geometrically
            T = 2.5
            alpha = 0.998
            T_min = 1e-12

            prev_x = x.copy()

            for step in range(n_steps):
                if calls >= budget or time.time() >= deadline:
                    break

                # Classic ASA direction: unit vector in direction of (x - prev_x)
                # scaled by norm of a Gaussian vector
                direction = x - prev_x
                norm_dir = np.linalg.norm(direction)

                if norm_dir < 1e-14:
                    # Random direction if stuck
                    direction = rng_asa.standard_normal(dim)
                    norm_dir = np.linalg.norm(direction)
                    if norm_dir < 1e-14:
                        direction = rng_asa.standard_normal(dim)
                        norm_dir = np.linalg.norm(direction)

                # Scale direction by norm of a new Gaussian (ASA rule)
                g = rng_asa.standard_normal(dim)
                scale = np.linalg.norm(g) / max(norm_dir, 1e-14)

                # Temperature-scaled step
                step_vec = T * direction * scale

                x_new = x + step_vec
                x_new = np.clip(x_new, -5.0, 5.0)

                fx_new = try_call(x_new)
                if fx_new is None:
                    break

                prev_x = x.copy()
                x = x_new
                fx = fx_new

                # Decay temperature
                T *= alpha
                if T < T_min:
                    T = T_min

            return x, fx

        # Phase 1: Initialization
        x0 = np.zeros(dim)
        f0 = try_call(x0)

        # A few random points to get a sense of the landscape
        n_init = 6
        for i in range(n_init):
            if calls >= budget or time.time() >= deadline:
                break
            xr = rng.uniform(-4.0, 4.0, size=dim)
            fr = try_call(xr)

        # Phase 2: Multi-start ASA
        # Use multiple restarts, with budget weighted toward later restarts
        # (later restarts start closer to the optimum, so they refine more)
        n_restarts = 5

        # Budget allocation: geometric progression, more to later restarts
        # Weights: 0.5, 0.75, 1.0, 1.25, 1.5 → normalized
        weights = np.array([0.5, 0.75, 1.0, 1.25, 1.5])
        weights /= np.sum(weights)

        remaining_budget = budget - calls
        asa_budgets = (weights * remaining_budget * 0.85).astype(int)
        asa_budgets = np.maximum(asa_budgets, 10)

        for i in range(n_restarts):
            if calls >= budget or time.time() >= deadline:
                break

            # Start point: mix of best found and random perturbation
            if best_x is not None:
                if i == 0:
                    x_start = best_x.copy()
                elif i == 1:
                    # Perturb best point
                    x_start = best_x + 1.5 * rng.standard_normal(dim)
                    x_start = np.clip(x_start, -5.0, 5.0)
                elif i == 2:
                    # Larger perturbation
                    x_start = best_x + 2.5 * rng.standard_normal(dim)
                    x_start = np.clip(x_start, -5.0, 5.0)
                elif i == 3:
                    # Random start
                    x_start = rng.uniform(-4.0, 4.0, size=dim)
                else:
                    # Perturb best with moderate noise
                    x_start = best_x + 1.0 * rng.standard_normal(dim)
                    x_start = np.clip(x_start, -5.0, 5.0)
            else:
                x_start = rng.uniform(-4.0, 4.0, size=dim)

            n_steps = min(asa_budgets[i], budget - calls)
            if n_steps < 5:
                break

            x_end, f_end = run_asa(x_start, n_steps, rng)

            # If we found a new best, do a short refinement from there
            if best_x is not None and calls < budget and time.time() < deadline:
                # Quick local polish with a short ASA run
                n_polish = min(30, budget - calls)
                if n_polish >= 5:
                    x_ref, f_ref = run_asa(best_x, n_polish, rng)

        # Phase 3: Nelder-Mead polish on the best point
        if calls < budget and time.time() < deadline and best_x is not None:
            try:
                from scipy.optimize import minimize

                remaining = budget - calls
                if remaining > 10:
                    # Use a small initial simplex
                    result = minimize(
                        func,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': min(remaining, 200),
                            'xatol': 1e-10,
                            'fatol': 1e-12,
                            'adaptive': True
                        }
                    )
                    if result.fun < best_f:
                        best_x = np.clip(result.x, -5.0, 5.0)
                        best_f = result.fun
            except Exception:
                pass

        # Phase 4: Final short ASA refinement if budget remains
        if calls < budget and time.time() < deadline and best_x is not None:
            n_final = min(budget - calls, 50)
            if n_final >= 5:
                x_final, f_final = run_asa(best_x, n_final, rng)

        return best_x, best_f

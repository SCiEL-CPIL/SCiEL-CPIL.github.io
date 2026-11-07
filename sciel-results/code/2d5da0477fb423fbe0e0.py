"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(mu+lambda)-ES with adaptive sigma, multi-start, and (1+1)-ES polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Deadline guard: evaluator kills after 330s. Allow 290s for safety.
        start_time = time.monotonic()
        max_time = 290.0

        def time_ok():
            return time.monotonic() - start_time < max_time

        rng = np.random.default_rng(42)

        # Call tracking
        calls = {"n": 0}
        best_x = np.zeros(dim)
        best_f = np.inf

        def call(x):
            nonlocal best_x, best_f
            if calls["n"] >= budget or not time_ok():
                return None
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            calls["n"] += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # ---- (mu+lambda)-ES core ----
        def es_search(x_init, fx_init, sigma_init, max_calls):
            nonlocal best_x, best_f
            mu = 4
            lam = 12
            sigma = sigma_init
            sigma_min = 1e-5
            sigma_max = 10.0
            x = x_init.copy()
            fx = fx_init

            # Success rate tracking
            success_count = 0
            window = 20
            window_count = 0

            while calls["n"] < budget and time_ok():
                # Generate offspring
                offspring = []
                for _ in range(lam):
                    if calls["n"] >= budget or not time_ok():
                        break
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = call(y)
                    if fy is None:
                        break
                    offspring.append((y, fy))

                if len(offspring) < mu:
                    break

                # Sort offspring by fitness
                offspring.sort(key=lambda t: t[1])
                selected = offspring[:mu]

                # Adapt sigma based on success rate
                improved = sum(1 for (_, fy) in selected if fy < fx)
                success_rate = improved / mu

                if success_rate > 0.2:
                    sigma *= 1.3
                elif success_rate < 0.2:
                    sigma *= 0.8
                sigma = np.clip(sigma, sigma_min, sigma_max)

                # Update parent as mean of selected offspring
                x_new = np.mean([pt for (pt, _) in selected], axis=0)
                fx_new = min(fy for (_, fy) in selected)

                x, fx = x_new, fx_new

        # ---- (1+1)-ES polish ----
        def es_polish(x_init, sigma_init):
            nonlocal best_x, best_f
            x = x_init.copy()
            fx = best_f
            sigma = sigma_init
            sigma_min = 1e-6

            while calls["n"] < budget and time_ok():
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = call(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.85
                    if sigma < sigma_min:
                        break

        # ---- Multi-start strategy ----
        # Phase 1: Sample diverse starting points
        n_starts = 8
        starts = [np.zeros(dim)]
        for _ in range(n_starts - 1):
            starts.append(rng.uniform(-4, 4, dim))

        # Evaluate all starts
        start_vals = []
        for s in starts:
            if not time_ok() or calls["n"] >= budget:
                break
            fs = call(s)
            if fs is not None:
                start_vals.append((s, fs))

        if not start_vals:
            return best_x, best_f

        # Sort starts by function value
        start_vals.sort(key=lambda t: t[1])

        # Phase 2: Run (mu+lambda)-ES from the best few starts
        n_es_runs = 3
        for i in range(min(n_es_runs, len(start_vals))):
            if not time_ok() or calls["n"] >= budget:
                break
            x0, f0 = start_vals[i]
            sigma0 = 1.5 if i == 0 else 1.0
            es_search(x0, f0, sigma0, budget)

        # Phase 3: Final (1+1)-ES polish on the global best
        if time_ok() and calls["n"] < budget and best_x is not None:
            es_polish(best_x, 0.3)

        return best_x, best_f
# EVOLVE-BLOCK-END

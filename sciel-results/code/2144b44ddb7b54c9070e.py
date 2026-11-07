"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio of (1+1)-ES runs with Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        deadline = time.time() + 285  # Safety margin

        rng = np.random.default_rng()

        def run_es(x_init, budget):
            """Run (1+1)-ES for a specific budget."""
            x = x_init.copy()
            fx = func(x)
            calls = 1
            sigma = 2.5
            min_sigma = 1e-8
            max_sigma = 8.0

            c_inc = 1.3
            c_dec = 0.7

            for _ in range(budget - 1):
                if time.time() > deadline:
                    break

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)

                fy = func(y)
                calls += 1

                if fy < fx:
                    x = y
                    fx = fy
                    sigma = min(sigma * c_inc, max_sigma)
                else:
                    sigma = max(sigma * c_dec, min_sigma)

            return x, fx

        def nelder_mead(x0, budget, initial_step=0.5):
            """Nelder-Mead simplex method."""
            n = dim
            # Initialize simplex
            simplex = [x0.copy()]
            for i in range(n):
                v = x0.copy()
                v[i] += initial_step
                simplex.append(np.clip(v, -5, 5))

            fvals = [func(s) for s in simplex]
            calls = n + 1

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 0.5

            for _ in range(budget - calls):
                if time.time() > deadline:
                    break

                # Sort
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls += 1

                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    calls += 1
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < fvals[-1]:
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        x_c = centroid - rho * (x_r - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls += 1

                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                            calls += 1

                if calls >= budget:
                    break

            best_idx = np.argmin(fvals)
            return simplex[best_idx], fvals[best_idx]

        # Strategy:
        # 1. Explore with 3 random (1+1)-ES starts, using 100 calls each.
        # 2. Polish the best with Nelder-Mead using remaining 200 calls.

        if total_budget < 20:
            x_init = rng.uniform(-5, 5, dim)
            return run_es(x_init, total_budget)

        n_explore = 3
        budget_explore = min(100, total_budget // (n_explore + 1))
        budget_nm = total_budget - n_explore * budget_explore

        best_x = None
        best_f = float('inf')

        for i in range(n_explore):
            if time.time() > deadline:
                break

            x_init = rng.uniform(-5, 5, dim)
            x_run, f_run = run_es(x_init, budget_explore)

            if f_run < best_f:
                best_f = f_run
                best_x = x_run

        # Polish with Nelder-Mead
        if time.time() < deadline and budget_nm > 5:
            x_final, f_final = nelder_mead(best_x, budget_nm, initial_step=0.3)
            if f_final < best_f:
                best_f = f_final
                best_x = x_final

        return best_x, best_f
# EVOLVE-BLOCK-END

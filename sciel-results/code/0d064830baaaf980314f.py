import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        calls = 0
        best_f = np.inf
        best_x = np.zeros(dim)

        def eval_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or not time_ok():
                return np.inf
            calls += 1
            f = float(func(x))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Random sampling to find a good starting point
        n_random = 30
        for i in range(n_random):
            x = rng.uniform(-4.0, 4.0, size=dim)
            eval_func(x)

        # Phase 2: Multi-start Nelder-Mead
        # Each run gets the best of the remaining budget
        n_restarts = 5
        nm_budget_per_run = max(10, (budget - calls - 1) // n_restarts)

        for r in range(n_restarts):
            if calls >= budget - 1 or not time_ok():
                break

            # Start Nelder-Mead from the current best point with small perturbation
            x0 = best_x + rng.normal(0, 0.3, size=dim)
            x0 = np.clip(x0, -5.0, 5.0)

            # Nelder-Mead simplex
            alpha, gamma, rho, sigma = 2.0, 2.0, 0.5, 0.5
            simplex = np.zeros((dim + 1, dim))
            fvals = np.zeros(dim + 1)

            simplex[0] = x0
            fvals[0] = eval_func(simplex[0])
            for i in range(1, dim + 1):
                simplex[i] = x0.copy()
                if fvals[i - 1] == np.inf:
                    break
                simplex[i][i - 1] += 0.1 * (5.0 - (-5.0)) / dim
                simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                fvals[i] = eval_func(simplex[i])

            # Nelder-Mead iterations
            nm_calls_remaining = nm_budget_per_run
            max_iters = nm_calls_remaining

            for _ in range(max_iters):
                if calls >= budget - 1 or not time_ok():
                    break

                # Sort by fitness
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflect
                x_r = centroid + gamma * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = eval_func(x_r)
                if f_r == np.inf:
                    break

                if fvals[0] <= f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                elif f_r < fvals[0]:
                    # Expand
                    x_e = centroid + alpha * (x_r - centroid)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = eval_func(x_e)
                    if f_e == np.inf:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                else:
                    # Contract
                    x_c = centroid + sigma * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = eval_func(x_c)
                    if f_c == np.inf:
                        break
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            fvals[i] = eval_func(simplex[i])
                        if fvals[i] == np.inf:
                            break

        return best_x, best_f

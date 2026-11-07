"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with (1+1)-ES polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        rng = np.random.default_rng(0)

        def time_ok():
            return time.time() < deadline

        calls_made = 0

        # Phase 1: Multi-start exploration
        n_starts = min(10, self.budget)
        best_x = None
        best_f = np.inf
        for i in range(n_starts):
            if not time_ok() or calls_made >= self.budget:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Nelder-Mead
        x = best_x
        fx = best_f
        d = self.dim

        simplex = [x.copy()]
        for i in range(d):
            xi = x.copy()
            xi[i] += 0.5 if xi[i] < 4.5 else -0.5
            simplex.append(xi)

        simplex_f = [fx]
        for i in range(1, d + 1):
            if not time_ok() or calls_made >= self.budget:
                break
            simplex_f.append(func(simplex[i]))
            calls_made += 1

        alpha, gamma, rho = 1.0, 2.0, 0.5
        max_nm_iter = 150
        es_reserve = 50

        for _ in range(max_nm_iter):
            if not time_ok() or calls_made >= self.budget - es_reserve:
                break

            order = np.argsort(simplex_f)
            simplex = [simplex[i].copy() for i in order]
            simplex_f = [simplex_f[i] for i in order]

            centroid = np.mean(simplex[:-1], axis=0)

            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls_made += 1
            if calls_made >= self.budget - es_reserve:
                break

            if f_r < simplex_f[0]:
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls_made += 1
                if f_e < f_r:
                    simplex[-1] = x_e
                    simplex_f[-1] = f_e
                else:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
            elif f_r < simplex_f[-2]:
                simplex[-1] = x_r
                simplex_f[-1] = f_r
            else:
                x_c = centroid + rho * (simplex[-1] - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                calls_made += 1
                if f_c < simplex_f[-1]:
                    simplex[-1] = x_c
                    simplex_f[-1] = f_c
                else:
                    for i in range(1, d + 1):
                        if not time_ok() or calls_made >= self.budget - es_reserve:
                            break
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        simplex_f[i] = func(simplex[i])
                        calls_made += 1

            min_idx = np.argmin(simplex_f)
            if simplex_f[min_idx] < best_f:
                best_f = simplex_f[min_idx]
                best_x = simplex[min_idx].copy()

        # Phase 3: (1+1)-ES polish
        x = best_x
        fx = best_f
        sigma = 0.5
        while calls_made < self.budget:
            if not time_ok():
                break
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls_made += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.1
            else:
                sigma *= 0.9
            if sigma < 1e-6:
                break

        return x, fx
# EVOLVE-BLOCK-END

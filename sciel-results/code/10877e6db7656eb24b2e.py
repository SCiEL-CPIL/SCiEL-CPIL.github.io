"""Optimizer for rotated Rosenbrock (BBOB f9) with deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Adaptive population search exploiting Rosenbrock valley structure."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(777)
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        # Phase 1: broad exploration to locate the basin
        n_explore = min(30, budget // 3)
        best_x = rng.uniform(-4, 4, dim)
        best_f = func(best_x)
        calls = 1

        for _ in range(n_explore - 1):
            if not time_ok() or calls >= budget:
                break
            x = rng.uniform(-4.5, 4.5, dim)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: adaptive local search with valley-aware scaling
        # Use a population that maintains the Rosenbrock valley relationship
        lam = 10
        mu = 4

        # Initialize population around best, with valley bias
        pop = []
        for i in range(lam):
            if i == 0:
                pop.append((best_f, best_x.copy()))
            else:
                # Perturb with valley-aware steps: make x[i+1] close to x[i]^2
                x = best_x + 0.8 * rng.standard_normal(dim)
                # Enforce valley relationship softly
                for j in range(dim - 1):
                    target = x[j] ** 2
                    x[j + 1] = 0.5 * x[j + 1] + 0.5 * target
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls += 1
                if not time_ok():
                    return best_x, best_f
                pop.append((fx, x.copy()))
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        sigma = 1.0
        sigma_min = 0.02
        sigma_max = 4.0

        while calls < budget and time_ok():
            pop.sort(key=lambda t: t[0])
            parents = np.array([p[1] for p in pop[:mu]])
            mean = np.mean(parents, axis=0)
            # Per-dimension step from parent spread
            spread = np.std(parents, axis=0) + 1e-6

            new_pop = [(pop[0][0], pop[0][1].copy())]
            improved = False

            for k in range(lam - 1):
                if not time_ok() or calls >= budget:
                    break
                step = sigma * spread
                y = mean + step * rng.standard_normal(dim)
                # Valley bias: pull consecutive coords toward x[i+1] ~ x[i]^2
                for j in range(dim - 1):
                    target = y[j] ** 2
                    y[j + 1] = 0.7 * y[j + 1] + 0.3 * target
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                new_pop.append((fy, y.copy()))
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    improved = True

            new_pop.sort(key=lambda t: t[0])
            pop = new_pop[:lam]

            # Adapt sigma
            if improved:
                sigma = min(sigma * 1.1, sigma_max)
            else:
                sigma = max(sigma * 0.9, sigma_min)

        return best_x, best_f
# EVOLVE-BLOCK-END

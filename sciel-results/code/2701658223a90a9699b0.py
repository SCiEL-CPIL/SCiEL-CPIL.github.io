"""Optimized solver for BBOB f9 (rotated Rosenbrock) using DE global search + L-BFGS-B + dense BFGS polish."""
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """DE-based global search followed by high-precision L-BFGS-B and dense BFGS polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        call_count = [0]
        best_f = [np.inf]
        best_x = [None]

        def counted_func(x):
            call_count[0] += 1
            if call_count[0] > budget:
                return best_f[0]
            x_clipped = np.clip(x, -5.0, 5.0)
            f = func(x_clipped)
            if f < best_f[0]:
                best_f[0] = f
                best_x[0] = x_clipped.copy()
            return f

        def remaining_time():
            return deadline - time.time()

        def has_budget():
            return call_count[0] < budget and remaining_time() > 3.0

        bounds = [(-5.0, 5.0)] * dim
        rng = np.random.default_rng(12345)

        # Phase 1: Differential Evolution for robust global search
        # DE population size = 10 * dim = 50, but we'll use a smaller one to conserve budget
        pop_size = min(10 * dim, 30)
        de_maxiter = 30  # limited to conserve budget for polish

        # Initialize population
        population = []
        # Include origin and some structured points
        population.append(np.zeros(dim))
        for _ in range(1):
            population.append(np.ones(dim) * 2.0)
        for _ in range(1):
            population.append(np.ones(dim) * -2.0)
        while len(population) < pop_size:
            population.append(rng.uniform(-4.5, 4.5, dim))

        # Evaluate initial population
        pop_f = []
        for x in population:
            if not has_budget():
                break
            f = counted_func(x)
            pop_f.append(f)

        # If we didn't evaluate all, pad with inf
        while len(pop_f) < len(population):
            pop_f.append(np.inf)

        # DE loop
        F = 0.8  # mutation factor
        CR = 0.9  # crossover rate

        for gen in range(de_maxiter):
            if not has_budget() or remaining_time() < 30.0:
                break

            # Sort population by fitness
            indices = np.argsort(pop_f)
            population = [population[i] for i in indices]
            pop_f = [pop_f[i] for i in indices]

            new_population = list(population)
            new_pop_f = list(pop_f)

            for i in range(len(population)):
                if not has_budget() or remaining_time() < 25.0:
                    break

                # Select three distinct random individuals different from i
                while True:
                    a, b, c = rng.choice(len(population), size=3, replace=False)
                    if len({a, b, c, i}) == 4:
                        break

                # Mutation
                v = population[a] + F * (population[b] - population[c])
                v = np.clip(v, -5.0, 5.0)

                # Crossover
                trial = population[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                trial = np.clip(trial, -5.0, 5.0)

                # Evaluate trial
                f_trial = counted_func(trial)

                # Selection
                if f_trial <= new_pop_f[i]:
                    new_population[i] = trial
                    new_pop_f[i] = f_trial

            population = new_population
            pop_f = new_pop_f

            # Check if we've made good progress
            if pop_f[0] < 1e-6:
                break

        # Get best from DE
        overall_best_f = best_f[0]
        overall_best_x = best_x[0].copy() if best_x[0] is not None else np.zeros(dim)

        # Phase 2: L-BFGS-B from the best DE candidate
        if has_budget() and remaining_time() > 20.0:
            try:
                res = minimize(
                    counted_func,
                    overall_best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
                )
                if res.fun < overall_best_f:
                    overall_best_f = res.fun
                    overall_best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Phase 3: Dense BFGS polish (exploits small dimension for quadratic convergence)
        if has_budget() and remaining_time() > 10.0:
            try:
                res = minimize(
                    counted_func,
                    overall_best_x,
                    method='BFGS',
                    options={'maxiter': 100, 'gtol': 1e-12, 'maxls': 30}
                )
                if res.fun < overall_best_f:
                    overall_best_f = res.fun
                    overall_best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Phase 4: Final high-precision L-BFGS-B polish
        if has_budget() and remaining_time() > 5.0:
            try:
                res = minimize(
                    counted_func,
                    overall_best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 150, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 30}
                )
                if res.fun < overall_best_f:
                    overall_best_f = res.fun
                    overall_best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Use tracked best if better
        if best_x[0] is not None and best_f[0] < overall_best_f:
            overall_best_f = best_f[0]
            overall_best_x = best_x[0]

        if overall_best_x is None:
            if best_x[0] is not None:
                return best_x[0], best_f[0]
            return np.zeros(dim), 0.0

        return overall_best_x, overall_best_f

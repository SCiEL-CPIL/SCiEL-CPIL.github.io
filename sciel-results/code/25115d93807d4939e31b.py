"""Black-box continuous optimizer on BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """
    Optimizer for BBOB f11 (Discus) using:
    - Phase 1: Sobol sampling for initial diverse population.
    - Phase 2: Iterative population-based L-BFGS-B refinement with adaptive restarts.
    - Deadline guard to ensure termination within time limits.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf
        calls_used = 0

        def check_budget():
            nonlocal calls_used
            if calls_used >= self.budget:
                raise Exception("Budget exceeded")
            if time.time() > deadline:
                raise Exception("Time exceeded")

        def safe_call(x):
            nonlocal calls_used, best_f, best_x
            check_budget()
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Sobol sampling to build initial diverse population
        n_initial = min(48, self.budget // 6)
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
        sobol_points = sampler.random(n_initial)
        X_initial = sobol_points * 10.0 - 5.0

        population = []
        for x in X_initial:
            if calls_used >= self.budget - 30:
                break
            fx = safe_call(x)
            population.append((fx, x))

        if not population:
            x = rng.uniform(-5, 5, self.dim)
            fx = safe_call(x)
            population.append((fx, x))

        population.sort(key=lambda t: t[0])

        # Phase 2: Iterative population-based L-BFGS-B refinement
        # Maintain a small population of top candidates and iteratively refine them
        # with L-BFGS-B, injecting new Sobol points if progress stalls.

        bounds = [(-5, 5)] * self.dim
        pop_size = 6  # Keep top 6 candidates in the population
        stagnation_limit = 15  # If no improvement in 15 calls, inject new point
        last_improvement_call = calls_used

        # Generate additional Sobol points for injection
        extra_sampler = qmc.Sobol(d=self.dim, scramble=True, seed=123)
        extra_sobol = extra_sampler.random(100)
        extra_points = extra_sobol * 10.0 - 5.0
        extra_idx = 0

        # Track best function value for stagnation detection
        prev_best_f = best_f

        while calls_used < self.budget - 10 and time.time() < deadline - 20:
            # Select top candidates from population for refinement
            top_candidates = population[:min(pop_size, len(population))]

            # Refine each top candidate with L-BFGS-B
            for fx0, x0 in top_candidates:
                if calls_used >= self.budget - 10 or time.time() > deadline - 15:
                    break

                # Limit iterations per refinement step to avoid long-running calls
                max_iter = max(5, min(20, (self.budget - calls_used) // 10))

                def lbfgs_func(x):
                    return safe_call(x)

                try:
                    res = minimize(
                        lbfgs_func,
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxiter': max_iter,
                            'gtol': 1e-8,
                            'ftol': 1e-10,
                        }
                    )
                except Exception:
                    pass

                # Check for improvement
                if best_f < prev_best_f:
                    prev_best_f = best_f
                    last_improvement_call = calls_used

                # If no improvement, inject a new Sobol point
                if calls_used - last_improvement_call >= stagnation_limit:
                    if extra_idx < len(extra_points):
                        x_new = extra_points[extra_idx]
                        extra_idx += 1
                    else:
                        x_new = rng.uniform(-5, 5, self.dim)
                    fx_new = safe_call(x_new)
                    population.append((fx_new, x_new))
                    population.sort(key=lambda t: t[0])
                    # Keep population bounded
                    if len(population) > 20:
                        population = population[:20]
                    last_improvement_call = calls_used

            # Update population with current best
            population.sort(key=lambda t: t[0])
            if len(population) > pop_size:
                # Keep top pop_size + a few for diversity
                population = population[:min(pop_size + 4, len(population))]

        # Final refinement: use remaining budget for aggressive local search from best point
        if calls_used < self.budget - 5 and best_x is not None:
            remaining = self.budget - calls_used - 5
            if remaining > 0:
                def final_lbfgs(x):
                    return safe_call(x)
                try:
                    minimize(
                        final_lbfgs,
                        best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxiter': max(10, remaining),
                            'gtol': 1e-10,
                            'ftol': 1e-12,
                        }
                    )
                except Exception:
                    pass

        if best_x is None:
            best_x = np.zeros(self.dim)
            if calls_used < self.budget:
                best_f = safe_call(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END

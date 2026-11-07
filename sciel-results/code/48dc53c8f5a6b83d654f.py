"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES with restarts + L-BFGS-B polish on Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        call_count = 0

        def bounded_func(x):
            nonlocal call_count
            if call_count >= budget:
                return best_f + 1e10
            x_clipped = np.clip(x, -5.0, 5.0)
            fx = float(func(x_clipped))
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Phase 1: CMA-ES with restarts
        # Allocate budget across restarts: 3 restarts with decreasing budgets
        n_restarts = 3
        # First restart gets more budget
        restart_budgets = [budget * 0.5, budget * 0.3, budget * 0.2]
        restart_budgets[-1] = budget - sum(restart_budgets[:-1])

        for r in range(n_restarts):
            if time.time() > deadline:
                break
            if call_count >= budget:
                break

            # Initialize: first at origin-ish, others random
            if r == 0:
                x0 = np.zeros(dim)
            else:
                x0 = rng.uniform(-4, 4, dim)

            # CMA-ES via scipy minimize (Nelder-Mead is not CMA, use COBYLA or custom)
            # Actually scipy doesn't have CMA-ES built-in. Let's use a manual CMA-ES.
            # For simplicity and robustness, use a population-based approach.

            # Use a simple population-based ES with adaptive step
            pop_size = 8
            sigma = 2.0 if r == 0 else 1.0
            remaining = min(restart_budgets[r], budget - call_count)

            if remaining < 10:
                break

            # Population-based search
            population = [x0 + sigma * rng.standard_normal(dim) for _ in range(pop_size)]
            population = [np.clip(p, -5, 5) for p in population]

            # Evaluate initial population
            pop_f = []
            for p in population:
                if call_count >= budget or time.time() > deadline:
                    break
                pop_f.append(bounded_func(p))

            if not pop_f:
                break

            # Track best in population
            pop_best_idx = int(np.argmin(pop_f))
            pop_best_x = population[pop_best_idx].copy()
            pop_best_f = pop_f[pop_best_idx]

            # Iterative improvement
            iters = remaining - len(pop_f)
            for it in range(max(0, iters)):
                if call_count >= budget or time.time() > deadline:
                    break

                # Generate offspring around best
                offspring = np.clip(pop_best_x + sigma * rng.standard_normal(dim), -5, 5)
                f_off = bounded_func(offspring)

                if f_off < pop_best_f:
                    pop_best_x = offspring.copy()
                    pop_best_f = f_off
                    sigma *= 1.2
                else:
                    sigma *= 0.9

                # Occasionally re-seed from random point to escape local minima
                if it % 20 == 0 and it > 0:
                    rand_x = np.clip(rng.uniform(-5, 5, dim), -5, 5)
                    f_rand = bounded_func(rand_x)
                    if f_rand < pop_best_f:
                        pop_best_x = rand_x.copy()
                        pop_best_f = f_rand
                        sigma = 1.5

        # Phase 2: Final polish with L-BFGS-B
        if best_x is not None and call_count < budget and time.time() < deadline:
            try:
                # Use remaining budget for polish iterations
                result = minimize(
                    bounded_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': budget - call_count, 'gtol': 1e-8}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END

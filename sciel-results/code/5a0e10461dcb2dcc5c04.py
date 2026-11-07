"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES-like optimizer with adaptive step size and restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf
        calls_made = 0

        # Phase 1: Random restarts to find a good starting point
        n_restarts = 10
        for i in range(n_restarts):
            x0 = rng.uniform(-4, 4, self.dim)
            fx = func(x0)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x0.copy()

        # Phase 2: Use L-BFGS-B from the best starting point
        # We need to wrap func to count calls and respect the budget
        remaining_budget = self.budget - calls_made

        if remaining_budget <= 0:
            return best_x, best_f

        # Create a wrapper that counts calls
        call_count = [0]

        def wrapped_func(x):
            nonlocal calls_made
            if call_count[0] >= remaining_budget:
                return best_f  # Return best known value to stop
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            call_count[0] += 1
            calls_made += 1
            return fx

        # Try L-BFGS-B
        try:
            result = minimize(
                wrapped_func,
                best_x,
                method='L-BFGS-B',
                bounds=[(-5, 5)] * self.dim,
                options={'maxiter': 100, 'maxls': 10}
            )
            if result.success and result.fun < best_f:
                best_f = result.fun
                best_x = result.x
        except Exception:
            pass

        # Phase 3: If we have budget left, do a fine-tuned local search
        remaining_budget = self.budget - calls_made
        if remaining_budget > 5:
            sigma = 0.5
            for _ in range(remaining_budget - 1):
                y = np.clip(best_x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                if fy < best_f:
                    best_f = fy
                    best_x = y
                    sigma *= 1.1
                else:
                    sigma *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END

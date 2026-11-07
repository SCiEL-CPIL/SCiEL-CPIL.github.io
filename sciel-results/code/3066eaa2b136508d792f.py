"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Derivative-free optimizer combining random restarts, Nelder-Mead, and CMA-ES-like steps."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim
        self.calls = 0

    def __call__(self, func):
        self.calls = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def wrapped_func(x):
            nonlocal best_x, best_f, self
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            val = float(func(x))
            self.calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Random sampling to get a good initial guess
        # Use about 20% of budget for initial exploration
        n_init = max(10, budget // 5)
        for _ in range(n_init):
            if self.calls >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            wrapped_func(x)

        # Phase 2: Nelder-Mead refinement from the best point
        if self.calls < budget:
            # Reserve some budget for potential second round
            remaining = budget - self.calls
            nm_budget = int(remaining * 0.7)
            
            try:
                result = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': nm_budget, 'maxfev': nm_budget, 'xatol': 1e-8, 'fatol': 1e-10}
                )
            except Exception:
                pass

        # Phase 3: If we still have budget, do a few more Nelder-Mead runs from different starting points
        if self.calls < budget:
            remaining = budget - self.calls
            if remaining > 50:
                # Generate a few new random starting points near the current best
                n_extra = min(3, remaining // 50)
                for _ in range(n_extra):
                    if self.calls >= budget - 10:
                        break
                    # Start near best but with some perturbation
                    x_start = best_x + rng.normal(0, 0.5, dim)
                    x_start = np.clip(x_start, -5, 5)
                    try:
                        result = minimize(
                            wrapped_func,
                            x_start,
                            method='Nelder-Mead',
                            options={'maxiter': remaining // (n_extra + 1), 'maxfev': remaining // (n_extra + 1), 'xatol': 1e-8, 'fatol': 1e-10}
                        )
                    except Exception:
                        pass

        return best_x, best_f
# EVOLVE-BLOCK-END

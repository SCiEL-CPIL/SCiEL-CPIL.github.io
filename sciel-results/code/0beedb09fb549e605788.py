"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


class Optimizer:
    """Multi-start Powell with Sobol initialization and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        self.budget_remaining = self.budget

        # Sobol initialization for diverse starting points
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
        # Generate enough points; we'll use the first few as starts
        num_starts = 10
        # Sobol needs power of 2 samples
        m = 1
        while 2**m < num_starts:
            m += 1
        sobol_samples = sampler.random(2**m)
        # Scale to [-5, 5]
        sobol_scaled = -5 + 10 * sobol_samples[:num_starts]

        # Evaluate initial points
        best_x = None
        best_f = np.inf
        starts = []
        for i in range(num_starts):
            if self.budget_remaining <= 0 or time.time() > deadline:
                break
            x = sobol_scaled[i].copy()
            x = np.clip(x, -5, 5)
            fx = func(x)
            self.budget_remaining -= 1
            starts.append((x, fx))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # If we didn't evaluate any, try center
        if best_x is None:
            x = np.zeros(self.dim)
            fx = func(x)
            self.budget_remaining -= 1
            best_x = x
            best_f = fx
            starts.append((x, fx))

        # Run Powell's method from each start, using remaining budget
        # Powell's method doesn't have a fixed budget per se, so we'll
        # run it with maxiter controlled by time/budget
        # We'll run Powell from the best start first, then others if time permits

        for i, (x0, f0) in enumerate(starts):
            if self.budget_remaining <= 0 or time.time() > deadline:
                break

            # Use remaining budget to estimate max iterations
            # Powell's method typically uses dim*(dim+1)/2 function evals per iteration
            # For dim=5, that's 15 evals per iteration
            evals_per_iter = self.dim * (self.dim + 1) // 2
            max_iter = max(1, min(50, self.budget_remaining // max(1, evals_per_iter)))

            # Time check: don't start if we're too close to deadline
            if time.time() > deadline - 5:
                break

            try:
                result = minimize(
                    func,
                    x0,
                    method='Powell',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': max_iter, 'maxfev': self.budget_remaining}
                )
                self.budget_remaining -= result.nfev
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END

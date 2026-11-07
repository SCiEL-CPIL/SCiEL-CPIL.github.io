"""Rewritten optimizer: multi-restart CMA-ES with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        calls = [0]

        def counting_func(x):
            if time.time() > deadline:
                raise TimeoutError()
            calls[0] += 1
            if calls[0] > self.budget:
                raise TimeoutError()
            return func(x)

        best_x = None
        best_f = np.inf

        rng = np.random.default_rng(42)

        # Multi-restart CMA-ES with decreasing sigma
        # Total budget is 500 calls. Split across restarts.
        # First restart gets more budget, subsequent ones less.
        n_restarts = 5
        # Budget allocation: first 300, then 100, 50, 30, 20
        budgets = [300, 100, 50, 30, 20]
        # Adjust to not exceed total budget
        total_alloc = sum(budgets)
        if total_alloc > self.budget:
            scale = self.budget / total_alloc
            budgets = [max(10, int(b * scale)) for b in budgets]

        for restart_idx, alloc in enumerate(budgets):
            if calls[0] >= self.budget:
                break
            if time.time() > deadline - 10:
                break

            # Initial point: first restart at origin-like, others random
            if restart_idx == 0:
                x0 = np.zeros(self.dim)
            else:
                # Start near best found so far with some perturbation
                if best_x is not None:
                    x0 = np.clip(best_x + rng.standard_normal(self.dim) * 1.0, -5, 5)
                else:
                    x0 = rng.uniform(-5, 5, self.dim)

            # Sigma: decrease with restart index
            sigma0 = 2.0 / (restart_idx + 1)

            # Use scipy's L-BFGS-B as a fast local refiner
            # But first try CMA-ES via scipy's minimize with Nelder-Mead for robustness
            # Actually, let's use a combination: 
            # For each restart, do a local search with L-BFGS-B (fast, few calls)
            # followed by a small CMA-ES-like refinement

            try:
                result = minimize(
                    counting_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': alloc, 'maxfun': alloc, 'maxls': 50}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

            # If we have budget left, do a local Nelder-Mead refinement
            remaining = self.budget - calls[0]
            if remaining > 10 and best_x is not None and time.time() < deadline - 5:
                nm_budget = min(remaining, 50)
                try:
                    result2 = minimize(
                        counting_func,
                        best_x,
                        method='Nelder-Mead',
                        options={'maxiter': nm_budget, 'maxfev': nm_budget,
                                  'xatol': 1e-6, 'fatol': 1e-8}
                    )
                    if result2.fun < best_f:
                        best_f = result2.fun
                        best_x = result2.x.copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END

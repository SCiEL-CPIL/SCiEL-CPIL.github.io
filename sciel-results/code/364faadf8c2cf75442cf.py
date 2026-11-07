"""Deterministic Global Optimization with Sobol sequence + local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """DGO with Sobol quasi-random sampling followed by local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Phase 1: Sobol quasi-random sampling for global coverage
        # Use scipy's Sobol sequence
        from scipy.stats.qmc import Sobol

        # Number of Sobol points to use in phase 1
        # We want to leave some budget for local refinement
        sobol_count = int(budget * 0.7)  # 70% for global exploration
        sobol_count = min(sobol_count, budget - 50)  # leave at least 50 for refinement

        # Generate Sobol sequence in [0, 1]^dim
        sobol = Sobol(d=dim, scramble=True, seed=42)
        # Need power of 2 for Sobol
        n_sobol = 1
        while n_sobol < sobol_count:
            n_sobol *= 2
        n_sobol = min(n_sobol, sobol_count)

        samples = sobol.random(n_sobol)

        # Map from [0, 1]^dim to [-5, 5]^dim
        for i in range(n_sobol):
            if time.time() > deadline:
                break
            x = -5 + 10 * samples[i]
            evaluate(x)

        # Phase 2: Local refinement around the best point found
        remaining = budget - calls
        if remaining > 10 and time.time() < deadline:
            def nm_func(x):
                if time.time() > deadline:
                    raise Exception("timeout")
                return evaluate(x)

            try:
                result = minimize(nm_func, best_x, method='Nelder-Mead',
                                   options={'xatol': 1e-8, 'fatol': 1e-10,
                                            'maxiter': min(remaining, 200),
                                            'maxfev': min(remaining, 200)})
            except Exception:
                pass

        # Phase 3: If we still have budget, do a few more local searches from
        # the top-k Sobol points
        if calls < budget and time.time() < deadline:
            # Re-evaluate top candidates with local search
            # We need to track which Sobol points were best
            # Since we didn't store all, let's do a quick local search from best_x
            # with small perturbations
            remaining = budget - calls
            if remaining > 5:
                rng = np.random.default_rng(123)
                for _ in range(min(remaining // 5, 10)):
                    if time.time() > deadline or calls >= budget:
                        break
                    # Small perturbation
                    scale = max(0.1, best_f * 0.01) if best_f != 0 else 0.1
                    perturbation = rng.standard_normal(dim) * 0.5
                    x_try = best_x + perturbation
                    evaluate(x_try)

        return best_x, best_f
# EVOLVE-BLOCK-END

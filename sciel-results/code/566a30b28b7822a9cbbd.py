"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Sobol sampling + Nelder-Mead local refinement for Bent Cigar.
    Uses deadline guard to stay within the 300s limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 270  # safety margin before 300s limit
        dim = self.dim
        calls_used = 0
        best_fx = float('inf')
        best_x = np.zeros(dim)

        # Phase 1: Sobol sequence sampling for global exploration
        # 64 points is a good balance for 5D
        n_sobol = 64
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=dim, seed=42)
        samples = sampler.random(n=n_sobol)
        # Map from [0,1]^dim to [-5,5]^dim
        samples = samples * 10 - 5

        for i in range(n_sobol):
            if calls_used >= self.budget:
                break
            if time.time() > deadline:
                break
            x = samples[i]
            fx = func(x)
            calls_used += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        # Phase 2: Nelder-Mead local refinement on best candidate
        # Use remaining budget for local search
        remaining = self.budget - calls_used
        if remaining > 10 and time.time() < deadline:
            try:
                result = minimize(
                    func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': min(remaining, 200),
                        'fatol': 1e-10,
                        'xatol': 1e-8
                    }
                )
                # Count the function calls made by Nelder-Mead
                nm_calls = result.nfev
                calls_used += nm_calls

                if result.fun < best_fx:
                    best_fx = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        # Phase 3: If we still have budget, do a few more Sobol points
        # or continue local refinement from best point
        if calls_used < self.budget and time.time() < deadline:
            rng = np.random.default_rng(42)
            # Try a few random perturbations around best_x
            for _ in range(min(20, self.budget - calls_used)):
                if time.time() > deadline:
                    break
                perturbation = rng.standard_normal(dim) * 0.5
                x = np.clip(best_x + perturbation, -5, 5)
                fx = func(x)
                calls_used += 1
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """Multi-start L-BFGS-B + (1+1)-ES refinement for Schwefel f20."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        rng = np.random.default_rng(42)
        call_count = [0]

        def safe_call(x):
            if call_count[0] >= budget:
                return np.inf
            call_count[0] += 1
            return func(x)

        best_x = None
        best_f = np.inf

        # Phase 1: Deterministic Sobol exploration to find a good starting basin
        # Use a small number of Sobol points to cover the space, then refine the best one
        sobol_n = 30
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        sobol_samples = sampler.random(sobol_n)
        sobol_samples = sobol_samples * 10 - 5  # Map [0, 1]^d to [-5, 5]^d

        # Evaluate Sobol points to find the best candidate
        sobol_best_x = None
        sobol_best_f = np.inf
        for x_sobol in sobol_samples:
            if call_count[0] >= budget or time.time() > deadline:
                break
            f_sobol = safe_call(x_sobol)
            if f_sobol < sobol_best_f:
                sobol_best_f = f_sobol
                sobol_best_x = x_sobol.copy()
        
        if sobol_best_x is not None:
            best_f = sobol_best_f
            best_x = sobol_best_x

        # Phase 2: L-BFGS-B refinement from the best Sobol point
        if best_x is not None and call_count[0] < budget and time.time() < deadline:
            try:
                res = minimize(
                    safe_call,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 100, 'maxls': 20}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 3: (1+1)-ES refinement from the best L-BFGS-B solution
        if best_x is not None and call_count[0] < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 1.0
            remaining = budget - call_count[0]

            for _ in range(remaining):
                if time.time() > deadline or call_count[0] >= budget:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_call(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9

                if sigma < 1e-10:
                    break

        # Phase 4: Final L-BFGS-B polish if we have budget left
        if best_x is not None and call_count[0] < budget and time.time() < deadline:
            remaining = budget - call_count[0]
            if remaining > 10:
                try:
                    res = minimize(
                        safe_call,
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': remaining, 'maxls': 10}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END

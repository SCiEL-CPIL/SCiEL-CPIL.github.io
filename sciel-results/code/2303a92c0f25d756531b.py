"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid: Sobol-based exploration + Nelder-Mead refinement + (1+1)-ES polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        calls_made = [0]

        def safe_func(x):
            calls_made[0] += 1
            if calls_made[0] > budget:
                raise RuntimeError("Budget exceeded")
            return func(np.clip(x, -5, 5))

        # Sobol sequence for deterministic global coverage
        from scipy.stats import qmc

        sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
        n_sobol = 64
        sobol_points = sobol.random(n_sobol)
        # Map from [0,1]^d to [-5,5]^d
        sobol_points = sobol_points * 10.0 - 5.0

        # Evaluate Sobol points to find good starting points
        best_x = sobol_points[0].copy()
        best_f = safe_func(best_x)

        for i in range(1, n_sobol):
            if time.time() > deadline:
                break
            xi = sobol_points[i]
            fi = safe_func(xi)
            if fi < best_f:
                best_f = fi
                best_x = xi.copy()

        # Nelder-Mead refinement from best Sobol point
        if time.time() < deadline:
            try:
                res = minimize(
                    safe_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'maxfev': 100,
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Additional local searches from other good Sobol points
        # Sort Sobol points by function value and try top few
        sobol_vals = []
        for i in range(n_sobol):
            if calls_made[0] >= budget:
                break
            if time.time() > deadline:
                break
            # Re-evaluate if not already done (we only stored best)
            # Actually we need to track all values; let's redo this properly
            pass

        # Let's do a second round: try NM from a few more diverse points
        # Pick points that are far from best_x
        if time.time() < deadline and calls_made[0] < budget:
            distances = [np.linalg.norm(sobol_points[i] - best_x) for i in range(n_sobol)]
            # Get indices sorted by distance descending, take a few
            idx_sorted = np.argsort(distances)[::-1]
            for idx in idx_sorted[:5]:
                if calls_made[0] >= budget:
                    break
                if time.time() > deadline:
                    break
                try:
                    res = minimize(
                        safe_func,
                        sobol_points[idx],
                        method='Nelder-Mead',
                        options={
                            'maxiter': 100,
                            'maxfev': 50,
                            'xatol': 1e-5,
                            'fatol': 1e-7,
                        }
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        # Final (1+1)-ES polish
        if time.time() < deadline and calls_made[0] < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            rng = np.random.default_rng(42)
            while calls_made[0] < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.85
                    if sigma < 1e-8:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END

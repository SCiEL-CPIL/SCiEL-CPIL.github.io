"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid LHS + Powell + Nelder-Mead optimizer for f7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0  # safety margin

        call_count = 0
        best_x = np.zeros(dim)
        best_fx = np.inf

        def record(x, fx):
            nonlocal best_x, best_fx
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        rng = np.random.default_rng(42)

        # Phase 1: Latin Hypercube Sampling (LHS) for global exploration
        # 10 points provides good coverage in 5D without being too expensive.
        n_lhs = 10
        lhs_points = np.zeros((n_lhs, dim))
        for i in range(n_lhs):
            for j in range(dim):
                # Sample in [-4, 4] to cover the likely minimizer region
                lhs_points[i, j] = -4 + 8 * (i + rng.random()) / n_lhs
        
        # Shuffle to decorrelate dimensions
        rng.shuffle(lhs_points)

        lhs_results = []
        for x0 in lhs_points:
            if call_count >= budget:
                break
            if time.time() > deadline - 10:
                break
            x0_clipped = np.clip(x0, -5, 5)
            fx = func(x0_clipped)
            call_count += 1
            record(x0_clipped, fx)
            lhs_results.append((fx, x0_clipped))

        # Sort results by fitness
        lhs_results.sort(key=lambda t: t[0])

        # Phase 2: Local optimization with Powell from top candidates
        # Powell is robust to steps and flat regions.
        n_powell = 3
        powell_starts = [x for f, x in lhs_results[:n_powell]]

        # Reserve 50 calls for final polish if possible
        reserve_for_polish = 50
        
        def budget_check(xk):
            if call_count > budget - reserve_for_polish:
                raise RuntimeError("Budget limit reached for Powell")

        for start in powell_starts:
            if call_count >= budget - reserve_for_polish:
                break
            if time.time() > deadline - 10:
                break

            def powell_func(x):
                nonlocal call_count
                x_clipped = np.clip(x, -5, 5)
                fx = func(x_clipped)
                call_count += 1
                record(x_clipped, fx)
                return fx

            try:
                # Powell options: 
                # maxiter is the number of iterations. 
                # ftol/xtol control convergence.
                minimize(powell_func, start, method='Powell',
                         options={'maxiter': 50, 'ftol': 1e-8, 'xtol': 1e-8},
                         callback=budget_check)
            except Exception:
                # Stop if budget exceeded or any other error
                pass

        # Phase 3: Final polish with Nelder-Mead
        # Use the best point found so far
        if call_count < budget - 5 and time.time() < deadline - 2:
            def nm_func(x):
                nonlocal call_count
                x_clipped = np.clip(x, -5, 5)
                fx = func(x_clipped)
                call_count += 1
                record(x_clipped, fx)
                return fx

            def nm_callback(xk):
                if call_count > budget - 2:
                    raise RuntimeError("Budget limit reached for NM")

            try:
                minimize(nm_func, best_x, method='Nelder-Mead',
                         options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 50},
                         callback=nm_callback)
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END

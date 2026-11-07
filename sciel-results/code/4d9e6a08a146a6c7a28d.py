"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead with Sobol-based initial points."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Generate Sobol points for initial multi-start
        from scipy.stats import qmc
        sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
        n_sobol = 64
        sobol_points = sobol.random(n_sobol)
        # Map [0,1]^dim to [-5,5]^dim
        sobol_points = sobol_points * 10.0 - 5.0

        # Also add a few random points for diversity
        rng = np.random.default_rng(42)
        extra_points = rng.uniform(-5, 5, size=(16, dim))

        # Combine initial points
        initial_points = np.vstack([sobol_points, extra_points])

        best_x = None
        best_f = np.inf

        # Evaluate all initial points
        for i in range(len(initial_points)):
            if time.time() > deadline:
                break
            x = initial_points[i]
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Use remaining budget for Nelder-Mead refinement from best starts
        # We'll do multi-start: take top few initial points and refine
        # Sort initial points by function value
        f_values = []
        for i in range(len(initial_points)):
            f_values.append((func(initial_points[i]), i))
        f_values.sort(key=lambda x: x[0])

        # Take top 5 starting points for Nelder-Mead
        top_starts = []
        seen = set()
        for fv, idx in f_values:
            if len(top_starts) >= 5:
                break
            # Avoid duplicates (shouldn't happen with Sobol but just in case)
            key = tuple(np.round(initial_points[idx], 6))
            if key not in seen:
                seen.add(key)
                top_starts.append((initial_points[idx], fv))

        # Refine each top start with Nelder-Mead, using remaining budget
        calls_made = len(initial_points) + len(f_values)  # initial + sorted evals
        remaining = budget - calls_made
        if remaining <= 0:
            return best_x, best_f

        # Distribute remaining budget among refinements
        # Give more budget to better starts
        n_refine = min(len(top_starts), 5)
        if n_refine > 0:
            # Allocate budget: 50% to best, 20% to second, 10% to third, 5% to fourth, 5% to fifth
            if n_refine == 1:
                allocs = [remaining]
            elif n_refine == 2:
                allocs = [int(0.6 * remaining), remaining - int(0.6 * remaining)]
            elif n_refine == 3:
                a1 = int(0.5 * remaining)
                a2 = int(0.3 * remaining)
                allocs = [a1, a2, remaining - a1 - a2]
            elif n_refine == 4:
                a1 = int(0.45 * remaining)
                a2 = int(0.25 * remaining)
                a3 = int(0.15 * remaining)
                allocs = [a1, a2, a3, remaining - a1 - a2 - a3]
            else:
                a1 = int(0.4 * remaining)
                a2 = int(0.2 * remaining)
                a3 = int(0.15 * remaining)
                a4 = int(0.1 * remaining)
                allocs = [a1, a2, a3, a4, remaining - a1 - a2 - a3 - a4]

            for i in range(n_refine):
                if time.time() > deadline:
                    break
                if allocs[i] < 5:
                    continue

                x0 = top_starts[i][0]
                f0 = top_starts[i][1]

                # Nelder-Mead with tight tolerances
                # Use maxiter as proxy for function call budget
                max_iter = max(5, allocs[i] - dim - 1)

                try:
                    res = minimize(
                        func,
                        x0,
                        method='Nelder-Mead',
                        options={
                            'maxiter': max_iter,
                            'xatol': 1e-8,
                            'fatol': 1e-10,
                            'adaptive': True,
                            'initial_simplex': None,
                        }
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        # Final polish with very tight Nelder-Mead if budget remains
        if best_x is not None and time.time() < deadline - 5:
            calls_so_far = budget  # approximate
            try:
                res = minimize(
                    func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True,
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END

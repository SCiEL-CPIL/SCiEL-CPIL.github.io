"""Black-box continuous optimizer on BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Optimizer for BBOB f11 (Discus) using:
    - Phase 1: Sobol sampling for initial exploration.
    - Phase 2: Nelder-Mead local refinement from top candidates.
    - Phase 3: Additional random restarts with local search if budget remains.
    - Deadline guard to ensure termination within time limits.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf
        calls_used = 0

        def check_budget():
            nonlocal calls_used
            if calls_used >= self.budget:
                raise Exception("Budget exceeded")
            if time.time() > deadline:
                raise Exception("Time exceeded")

        def safe_call(x):
            nonlocal calls_used, best_f, best_x
            check_budget()
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Sobol sampling to build initial diverse set
        from scipy.stats import qmc
        n_initial = min(128, self.budget // 4)
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
        sobol_points = sampler.random(n_initial)
        X_initial = sobol_points * 10.0 - 5.0

        evaluations = []
        for x in X_initial:
            if calls_used >= self.budget - 30:
                break
            fx = safe_call(x)
            evaluations.append((fx, x))

        if not evaluations:
            x = rng.uniform(-5, 5, self.dim)
            fx = safe_call(x)
            evaluations.append((fx, x))

        # Sort by function value
        evaluations.sort(key=lambda t: t[0])

        # Phase 2: Local refinement from the best few points using Nelder-Mead
        remaining_budget = self.budget - calls_used

        # Select top candidates for local refinement
        n_restarts = min(6, len(evaluations))
        max_iter_per_restart = max(20, remaining_budget // (n_restarts + 1))

        for i in range(n_restarts):
            if time.time() > deadline - 15:
                break
            if calls_used >= self.budget - 10:
                break

            fx0, x0 = evaluations[i]

            def nm_func(x):
                return safe_call(x)

            try:
                res = minimize(
                    nm_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter_per_restart,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True,
                    }
                )
            except Exception:
                pass

        # Phase 3: If budget remains, try additional random restarts with local search
        if calls_used < self.budget - 30 and time.time() < deadline - 40:
            # Estimate how many more restarts we can afford
            possible_restarts = max(1, (self.budget - calls_used) // 40)
            for _ in range(possible_restarts):
                if time.time() > deadline - 15:
                    break
                if calls_used >= self.budget - 15:
                    break

                # Try points near the current best and random points
                if rng.random() < 0.5 and best_x is not None:
                    # Perturb the best point
                    sigma = 1.0
                    x0 = best_x + rng.normal(0, sigma, self.dim)
                else:
                    x0 = rng.uniform(-5, 5, self.dim)

                fx0 = safe_call(x0)

                if calls_used < self.budget - 10:
                    def nm_func2(x):
                        return safe_call(x)
                    try:
                        res = minimize(
                            nm_func2,
                            x0,
                            method='Nelder-Mead',
                            options={
                                'maxiter': 25,
                                'xatol': 1e-8,
                                'fatol': 1e-10,
                                'adaptive': True,
                            }
                        )
                    except Exception:
                        pass

        # Phase 4: Final refinement from the best point found
        if best_x is not None and calls_used < self.budget - 10 and time.time() < deadline - 10:
            def final_refine(x):
                return safe_call(x)
            try:
                res = minimize(
                    final_refine,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 30,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True,
                    }
                )
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(self.dim)
            if calls_used < self.budget:
                best_f = safe_call(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END

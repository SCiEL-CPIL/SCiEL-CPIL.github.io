"""CMA-ES based optimizer for Rosenbrock with adaptive population and L-BFGS-B polish."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize import cma


class Optimizer:
    """CMA-ES based optimizer for Rosenbrock with structured initialization and L-BFGS-B polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        call_count = 0
        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= self.budget:
                raise ValueError("Budget exceeded")
            if time.time() > deadline - 5:
                raise ValueError("Time exceeded")
            call_count += 1
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = np.array(x, copy=True)
            return f

        # Phase 1: Structured initial sampling
        n_structured = 5
        structured_points = [
            np.zeros(self.dim),
            np.ones(self.dim) * 0.5,
            np.ones(self.dim) * -0.5,
            np.ones(self.dim) * 1.0,
            np.ones(self.dim) * -1.0,
        ]

        n_random = min(20, max(10, self.budget // 15))
        random_points = rng.uniform(-5, 5, (n_random, self.dim))

        all_points = np.vstack([structured_points, random_points])

        for x in all_points:
            try:
                safe_func(x)
            except Exception:
                break

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            call_count += 1

        remaining = self.budget - call_count

        if time.time() > deadline - 20:
            return best_x, best_f

        # Phase 2: CMA-ES for global search with adaptive population
        # Start from the best point found, with moderate initial sigma
        x0 = best_x.copy()
        sigma0 = 2.0  # Moderate initial step size

        # Population size: for dim=5, typical is 4 + floor(3*ln(5)) ≈ 9-10
        # Use a smaller population to conserve budget for more iterations
        pop_size = max(6, min(12, int(4 + 3 * np.log(self.dim))))

        cma_options = {
            'maxiter': 200,
            'maxfevals': remaining,
            'tolopt': 1e-10,
            'tolfun': 1e-12,
            'tolfun2': 1e-12,
            'verbose': -9,
            'seed': 42,
            'bounds': [(-5, 5)] * self.dim,
        }

        try:
            es = cma.CMA(x0, sigma0, cma_options)
            while not es.stop() and call_count < self.budget and time.time() < deadline - 30:
                solutions = es.ask()
                for sol in solutions:
                    try:
                        safe_func(sol)
                    except Exception:
                        break
                if call_count >= self.budget or time.time() > deadline - 30:
                    break
                es.tell(solutions, [func(s) for s in solutions])

        except Exception:
            pass

        remaining = self.budget - call_count

        if time.time() > deadline - 20:
            return best_x, best_f

        # Phase 3: L-BFGS-B local refinement from the best point
        if remaining > 20:
            lbfgs_remaining = min(remaining - 5, 100)

            def lbfgs_wrapper(x):
                nonlocal call_count
                if call_count >= self.budget:
                    raise ValueError("Budget exceeded")
                if time.time() > deadline - 5:
                    raise ValueError("Time exceeded")
                call_count += 1
                f = func(x)
                if f < best_f:
                    best_f = f
                    best_x = np.array(x, copy=True)
                return f

            try:
                res = minimize(
                    lbfgs_wrapper,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 200, 'maxfun': lbfgs_remaining, 'maxls': 20, 'gtol': 1e-8}
                )
            except Exception:
                pass

        remaining = self.budget - call_count

        if time.time() > deadline - 15:
            return best_x, best_f

        # Phase 4: Nelder-Mead polish if budget remains
        if remaining > 30:
            nm_remaining = min(remaining - 10, 80)

            def nm_wrapper(x):
                nonlocal call_count
                if call_count >= self.budget:
                    raise ValueError("Budget exceeded")
                if time.time() > deadline - 5:
                    raise ValueError("Time exceeded")
                call_count += 1
                f = func(x)
                if f < best_f:
                    best_f = f
                    best_x = np.array(x, copy=True)
                return f

            try:
                res_nm = minimize(
                    nm_wrapper,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 80, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
            except Exception:
                pass

        return best_x, best_f

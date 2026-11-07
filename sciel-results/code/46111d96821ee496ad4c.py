import numpy as np
from scipy.optimize import minimize, Powell
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.call_count = 0
        self.start_time = time.time()
        self.time_limit = 295.0  # Safety margin before 330s kill

    def __call__(self, func):
        self.call_count = 0
        self.start_time = time.time()
        self.best_x = None
        self.best_f = float('inf')

        def wrapped_func(x):
            self.call_count += 1
            if time.time() - self.start_time > self.time_limit:
                raise TimeoutError
            x = np.clip(x, -5.0, 5.0)
            return func(x)

        def update_best(x, f):
            if f < self.best_f:
                self.best_f = f
                self.best_x = np.clip(x, -5.0, 5.0).copy()

        n = self.dim
        budget = self.budget
        bounds = [(-5.0, 5.0) for _ in range(n)]

        # Phase 1: Latin Hypercube Sampling for diverse global coverage
        # LHS ensures one sample per stratum in each dimension, giving better
        # coverage than random or Halton for moderate sample sizes
        n_starts = 30

        def latin_hypercube(n_points, dim):
            pts = np.zeros((n_points, dim))
            for d in range(dim):
                # Random permutation of strata
                perm = np.random.permutation(n_points)
                # Uniform within each stratum
                u = (perm + np.random.random(n_points)) / n_points
                pts[:, d] = u
            # Scale to [-5, 5]
            pts = pts * 10.0 - 5.0
            return pts

        # Use a fixed seed for reproducibility within the run
        np.random.seed(42)
        X0 = latin_hypercube(n_starts, n)

        # Evaluate initial points
        candidates = []
        for x in X0:
            if self.call_count >= budget or time.time() - self.start_time > self.time_limit:
                break
            try:
                f = wrapped_func(x)
                candidates.append((f, x.copy()))
                update_best(x, f)
            except Exception:
                pass

        if not candidates:
            return self.best_x, self.best_f

        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Local Optimization using Powell's method
        # Powell's method is gradient-free and adapts to anisotropic curvature,
        # making it well-suited for f14's different-power structure
        remaining = budget - self.call_count

        # Try to polish the top candidates
        # Powell's method typically needs ~50-150 evaluations per optimization
        # for 5D problems
        top_k = min(6, len(candidates))
        if top_k > 0:
            # Allocate budget: give more to better candidates
            # But ensure at least some for each
            calls_per_opt = max(20, remaining // (top_k + 1))

            for i in range(top_k):
                if self.call_count >= budget - 5 or time.time() - self.start_time > self.time_limit:
                    break

                f0, x0 = candidates[i]

                try:
                    res = minimize(wrapped_func, x0, method='Powell', bounds=bounds,
                                   options={'maxiter': 50, 'maxfev': calls_per_opt})
                    if res.fun < self.best_f:
                        self.best_f = res.fun
                        self.best_x = np.clip(res.x, -5.0, 5.0).copy()
                except Exception:
                    pass

        # Phase 3: Final Polish on the absolute best using Nelder-Mead
        # Nelder-Mead can make fine adjustments that Powell might miss
        if self.call_count < budget - 10 and self.best_x is not None:
            try:
                remaining_final = budget - self.call_count - 2
                res = minimize(wrapped_func, self.best_x, method='Nelder-Mead',
                               options={'maxiter': 100, 'maxfev': remaining_final,
                                        'xatol': 1e-8, 'fatol': 1e-10})
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = np.clip(res.x, -5.0, 5.0).copy()
            except Exception:
                pass

        # Phase 4: If we still have budget, do a second round of local search
        # on the best point with a slightly perturbed start
        if self.call_count < budget - 10 and self.best_x is not None:
            try:
                # Small perturbation to potentially escape any local minimum
                perturbation = np.random.normal(0, 0.1, n)
                x_pert = np.clip(self.best_x + perturbation, -5.0, 5.0)
                remaining_final = budget - self.call_count - 2
                res = minimize(wrapped_func, x_pert, method='Powell', bounds=bounds,
                               options={'maxiter': 30, 'maxfev': remaining_final})
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = np.clip(res.x, -5.0, 5.0).copy()
            except Exception:
                pass

        return self.best_x, self.best_f

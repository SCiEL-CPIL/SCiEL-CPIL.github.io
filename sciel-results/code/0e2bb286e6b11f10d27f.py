import numpy as np
import time
import sys

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._start_time = None
        self._deadline = None
        self._best_x = None
        self._best_f = np.inf

    def __call__(self, func):
        self._start_time = time.time()
        self._deadline = self._start_time + 320
        self._call_count = 0
        self.budget_remaining = self.budget
        self._best_x = np.zeros(self.dim)
        self._best_f = np.inf
        self._rng = np.random.default_rng(42)

        def safe_call(x):
            if self._call_count >= self.budget or time.time() >= self._deadline:
                return 1e10
            self._call_count += 1
            self.budget_remaining -= 1
            val = func(x)
            if not np.isfinite(val):
                return 1e10
            if val < self._best_f:
                self._best_f = val
                self._best_x = x.copy()
            return val

        # Generate Sobol sequence for global exploration points
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
            sobol_points = sampler.random(500)
            sobol_points = sobol_points * 10.0 - 5.0
        except ImportError:
            sobol_points = self._rng.uniform(-5, 5, (500, self.dim))

        sobol_idx = 0
        total_sobol = len(sobol_points)

        # Phase 1: Quick global exploration with 25 Sobol points
        for i in range(25):
            if self._call_count >= self.budget or time.time() >= self._deadline:
                break
            x = sobol_points[sobol_idx]
            sobol_idx += 1
            safe_call(x)

        # Phase 2: (1+1)-ES with stagnation-based restarts
        sigma = 0.5
        sigma_min = 1e-6
        sigma_max = 3.0
        stagnation_limit = 15
        stagnation_counter = 0

        while self._call_count < self.budget and time.time() < self._deadline:
            # (1+1)-ES step
            noise = self._rng.standard_normal(self.dim)
            candidate_x = self._best_x + sigma * noise
            candidate_x = np.clip(candidate_x, -5.0, 5.0)

            f_cand = safe_call(candidate_x)

            if f_cand < self._best_f:
                # Accept: move to candidate
                self._best_f = f_cand
                self._best_x = candidate_x
                stagnation_counter = 0
                # Slightly increase sigma to explore
                sigma = min(sigma * 1.02, sigma_max)
            else:
                # Reject: stay at current best
                stagnation_counter += 1
                # Decrease sigma to refine
                sigma = max(sigma * 0.97, sigma_min)

                # If stagnated, restart from a new Sobol point
                if stagnation_counter >= stagnation_limit:
                    if sobol_idx < total_sobol:
                        restart_x = sobol_points[sobol_idx]
                        sobol_idx += 1
                    else:
                        restart_x = self._rng.uniform(-5, 5, self.dim)
                    f_restart = safe_call(restart_x)
                    if f_restart < self._best_f:
                        self._best_f = f_restart
                        self._best_x = restart_x
                        sigma = 0.5
                        stagnation_counter = 0
                    else:
                        # Stay at best, reset stagnation
                        stagnation_counter = 0

        # Phase 3: Final polish with L-BFGS-B if time permits
        if self._call_count < self.budget - 5 and time.time() < self._deadline - 10:
            try:
                from scipy.optimize import minimize
                
                def wrapper(x):
                    if self._call_count >= self.budget or time.time() >= self._deadline:
                        return 1e10
                    self._call_count += 1
                    self.budget_remaining -= 1
                    val = func(x)
                    if not np.isfinite(val):
                        return 1e10
                    return val
                
                bounds = [(-5.0, 5.0)] * self.dim
                result = minimize(wrapper, self._best_x, method='L-BFGS-B', 
                                  bounds=bounds, options={'maxiter': 20, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if result.fun < self._best_f and np.isfinite(result.fun):
                    self._best_f = result.fun
                    self._best_x = result.x.copy()
            except Exception:
                pass

        return self._best_x, self._best_f

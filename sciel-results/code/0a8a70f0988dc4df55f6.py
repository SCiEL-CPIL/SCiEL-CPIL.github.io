import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = time.time() + 290  # Safety margin before 330s limit

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def wrapped_func(x):
            if time.time() > self.deadline:
                raise TimeoutError()
            if self.calls >= budget:
                raise ValueError("Budget exceeded")
            self.calls += 1
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5, 5)
            val = func(x)
            if np.isnan(val) or np.isinf(val):
                val = 1e10
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            return val

        # Phase 1: Coarse random sampling to find a good starting region
        # Sample 50 points randomly in [-4, 4]^5
        for _ in range(50):
            if self.calls >= budget or time.time() > self.deadline:
                break
            x = rng.uniform(-4, 4, dim)
            try:
                wrapped_func(x)
            except (TimeoutError, ValueError):
                break
            except Exception:
                continue

        if self.best_x is None:
            self.best_x = np.zeros(dim)
            try:
                self.best_f = wrapped_func(self.best_x)
            except:
                self.best_f = 1e10
                self.best_x = np.zeros(dim)

        # Phase 2: Multiple Nelder-Mead restarts from best and random points near best
        # Do up to 5 restarts, each with limited iterations
        best_candidate = self.best_x.copy()
        best_candidate_val = self.best_f

        for restart in range(5):
            if self.calls >= budget - 20 or time.time() > self.deadline:
                break

            if restart == 0:
                start = best_candidate.copy()
            else:
                # Perturb around the current best
                start = best_candidate + rng.normal(0, 0.5, dim)
                start = np.clip(start, -5, 5)

            try:
                res = minimize(wrapped_func, start, method='Nelder-Mead',
                               options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
                if res.fun < best_candidate_val:
                    best_candidate_val = res.fun
                    best_candidate = res.x.copy()
            except (TimeoutError, ValueError):
                break
            except Exception:
                continue

        # Phase 3: Powell polish from the best candidate
        if self.calls < budget - 5 and time.time() < self.deadline:
            try:
                res_powell = minimize(wrapped_func, best_candidate, method='Powell',
                                      options={'maxiter': 200, 'xtol': 1e-10, 'ftol': 1e-12})
                if res_powell.fun < best_candidate_val:
                    best_candidate_val = res_powell.fun
                    best_candidate = res_powell.x.copy()
            except (TimeoutError, ValueError):
                pass
            except Exception:
                pass

        # Phase 4: If budget remains, do a few more L-BFGS-B runs from different starts
        remaining = budget - self.calls
        if remaining > 20 and time.time() < self.deadline:
            # Use L-BFGS-B with bounds for a few more attempts
            bounds = [(-5.0, 5.0)] * dim
            for attempt in range(3):
                if self.calls >= budget - 5 or time.time() > self.deadline:
                    break
                if attempt == 0:
                    start = best_candidate.copy()
                else:
                    start = best_candidate + rng.normal(0, 1.0, dim)
                    start = np.clip(start, -5, 5)

                try:
                    res = minimize(wrapped_func, start, method='L-BFGS-B', bounds=bounds,
                                   options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
                    if res.fun < best_candidate_val:
                        best_candidate_val = res.fun
                        best_candidate = res.x.copy()
                except (TimeoutError, ValueError):
                    break
                except Exception:
                    continue

        # Final: make sure best_x and best_f are set
        if self.best_x is None:
            self.best_x = np.zeros(dim)
            self.best_f = 1e10
        else:
            # Update with best_candidate if it's better
            if best_candidate_val < self.best_f:
                self.best_f = best_candidate_val
                self.best_x = best_candidate.copy()

        return self.best_x, self.best_f

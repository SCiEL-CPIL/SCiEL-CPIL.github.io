import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Multi-start L-BFGS-B + Nelder-Mead polish for Rosenbrock-like functions.
    
    Uses multiple diverse starting points, L-BFGS-B for fast convergence,
    and Nelder-Mead for final polishing. Budget-aware with time guards.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf
        self.calls_made = 0
        self.start_time = 0.0
        self.deadline = 0.0

    def __call__(self, func):
        self.start_time = time.time()
        self.deadline = self.start_time + 280.0

        def wrapped_func(x):
            if self.calls_made >= self.budget:
                raise StopIteration("Budget exhausted")
            if time.time() > self.deadline:
                raise StopIteration("Time limit exceeded")
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (self.dim,):
                x = x.reshape(self.dim)
            x = np.clip(x, -5, 5)
            self.calls_made += 1
            val = float(func(x))
            if not np.isfinite(val):
                val = 1e10
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            return val

        rng = np.random.default_rng(123)

        # Diverse starting points covering the likely region [-4, 4]^5
        starts = []
        starts.append(np.zeros(self.dim))
        starts.append(np.ones(self.dim))
        starts.append(np.full(self.dim, -1.0))
        starts.append(np.full(self.dim, 2.0))
        starts.append(np.full(self.dim, -2.0))
        starts.append(np.full(self.dim, 3.0))
        starts.append(np.full(self.dim, -3.0))
        starts.append(np.full(self.dim, 0.5))
        starts.append(np.full(self.dim, -0.5))

        # Random points in the likely region
        for _ in range(10):
            starts.append(rng.uniform(-4, 4, self.dim))

        # Remove duplicates
        unique_starts = []
        for s in starts:
            is_dup = False
            for u in unique_starts:
                if np.allclose(s, u, atol=1e-5):
                    is_dup = True
                    break
            if not is_dup:
                unique_starts.append(s)

        # Phase 1: L-BFGS-B for fast convergence from multiple starts
        for x0 in unique_starts:
            if self.calls_made >= self.budget - 30:
                break
            if time.time() > self.deadline:
                break

            x0 = np.clip(x0, -5, 5)
            remaining_budget = self.budget - self.calls_made
            if remaining_budget < 20:
                break

            try:
                res_lbfgsb = scipy.optimize.minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 80,
                        'gtol': 1e-6,
                        'ftol': 1e-10,
                        'maxfun': remaining_budget
                    }
                )

                if np.isfinite(res_lbfgsb.fun) and res_lbfgsb.fun < self.best_f:
                    self.best_f = res_lbfgsb.fun
                    self.best_x = np.clip(res_lbfgsb.x, -5, 5)

            except StopIteration:
                break
            except Exception:
                pass

        # Phase 2: Nelder-Mead polish from the best point found
        if self.calls_made < self.budget - 5 and self.best_x is not None:
            try:
                x_start = self.best_x.copy()
                x_start = np.clip(x_start, -5, 5)

                res_nm = scipy.optimize.minimize(
                    wrapped_func,
                    x_start,
                    method='Nelder-Mead',
                    options={
                        'xatol': 1e-8,
                        'fatol': 1e-8,
                        'maxiter': 100,
                        'adaptive': True
                    }
                )

                if np.isfinite(res_nm.fun) and res_nm.fun < self.best_f:
                    self.best_f = res_nm.fun
                    self.best_x = np.clip(res_nm.x, -5, 5)

            except StopIteration:
                pass
            except Exception:
                pass

        # Phase 3: Final high-precision polish if budget allows
        if self.calls_made < self.budget - 2 and self.best_x is not None and self.best_f < 1e4:
            try:
                x_start = self.best_x.copy()
                x_start = np.clip(x_start, -5, 5)

                res_nm2 = scipy.optimize.minimize(
                    wrapped_func,
                    x_start,
                    method='Nelder-Mead',
                    options={
                        'xatol': 1e-10,
                        'fatol': 1e-10,
                        'maxiter': 50,
                        'adaptive': True
                    }
                )

                if np.isfinite(res_nm2.fun) and res_nm2.fun < self.best_f:
                    self.best_f = res_nm2.fun
                    self.best_x = np.clip(res_nm2.x, -5, 5)

            except StopIteration:
                pass
            except Exception:
                pass

        # Fallback if no calls were made
        if self.best_x is None:
            x0 = np.ones(self.dim)
            try:
                self.best_f = wrapped_func(x0)
                self.best_x = x0.copy()
            except Exception:
                self.best_x = x0
                self.best_f = 1e10

        return self.best_x, self.best_f

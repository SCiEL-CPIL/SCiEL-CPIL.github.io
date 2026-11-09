import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Hybrid Powell + L-BFGS-B + Nelder-Mead Optimizer.
    
    Uses Powell's method for robust initial convergence (derivative-free, good for Rosenbrock valleys),
    followed by L-BFGS-B for fast gradient-based convergence,
    and Nelder-Mead for final high-precision polishing.
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
        self.deadline = self.start_time + 285.0  # Safety margin

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

        rng = np.random.default_rng(42)
        
        # Diverse starting points
        starts = [
            np.zeros(self.dim),
            np.ones(self.dim),
            np.full(self.dim, 2.0),
            np.full(self.dim, -1.0),
            np.full(self.dim, 3.0),
            np.full(self.dim, -2.0),
        ]
        
        # Random starts in likely region
        for _ in range(8):
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

        # Phase 1: Powell's Method for robust exploration
        # Powell is derivative-free and handles the Rosenbrock valley well
        for x0 in unique_starts:
            if self.calls_made >= self.budget - 100:  # Leave budget for later phases
                break
            if time.time() > self.deadline:
                break
            
            x0 = np.clip(x0, -5, 5)
            remaining_budget = self.budget - self.calls_made
            if remaining_budget < 50:
                break

            try:
                res_powell = scipy.optimize.minimize(
                    wrapped_func,
                    x0,
                    method='Powell',
                    options={
                        'maxiter': 100,
                        'ftol': 1e-8,
                        'xtol': 1e-8,
                        'maxfev': remaining_budget
                    }
                )
                
                if np.isfinite(res_powell.fun) and res_powell.fun < self.best_f:
                    self.best_f = res_powell.fun
                    self.best_x = np.clip(res_powell.x, -5, 5)
                    
            except StopIteration:
                break
            except Exception:
                pass

        # Phase 2: L-BFGS-B for fast convergence from the best point found
        if self.calls_made < self.budget - 20 and self.best_x is not None:
            try:
                x_start = self.best_x.copy()
                x_start = np.clip(x_start, -5, 5)
                
                res_lbfgsb = scipy.optimize.minimize(
                    wrapped_func,
                    x_start,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 100,
                        'gtol': 1e-6,
                        'ftol': 1e-10,
                        'maxfun': self.budget - self.calls_made
                    }
                )
                
                if np.isfinite(res_lbfgsb.fun) and res_lbfgsb.fun < self.best_f:
                    self.best_f = res_lbfgsb.fun
                    self.best_x = np.clip(res_lbfgsb.x, -5, 5)
                    
            except StopIteration:
                pass
            except Exception:
                pass

        # Phase 3: Nelder-Mead for final polishing
        if self.calls_made < self.budget - 5 and self.best_x is not None:
            try:
                x_start = self.best_x.copy()
                x_start = np.clip(x_start, -5, 5)
                
                res_nm = scipy.optimize.minimize(
                    wrapped_func,
                    x_start,
                    method='Nelder-Mead',
                    options={
                        'xatol': 1e-10,
                        'fatol': 1e-10,
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

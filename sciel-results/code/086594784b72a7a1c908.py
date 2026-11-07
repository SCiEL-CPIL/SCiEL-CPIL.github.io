import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = float('inf')
        self.start_time = None
        self.calls_made = 0
        self.max_time = 290.0

    def __call__(self, func):
        self.start_time = time.time()
        self.calls_made = 0
        self.best_x = None
        self.best_f = float('inf')

        def safe_func(x):
            if time.time() - self.start_time > self.max_time:
                return float('inf')
            if self.calls_made >= self.budget:
                return float('inf')
            self.calls_made += 1
            try:
                fx = float(func(x))
            except Exception:
                return float('inf')
            if not np.isfinite(fx):
                return 1e10
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = np.array(x, copy=True)
            return fx

        def run_l_bfgs_b(x0, maxiter, ftol, gtol):
            """
            Run L-BFGS-B with given tolerances. Returns (x, f).
            """
            if self.calls_made >= self.budget - 10:
                return x0, safe_func(x0)
            if time.time() - self.start_time > self.max_time - 5:
                return x0, safe_func(x0)
            
            try:
                res = minimize(
                    safe_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': maxiter, 'maxls': 20, 'ftol': ftol, 'gtol': gtol}
                )
                if res.fun < self.best_f:
                    return res.x, res.fun
                return x0, self.best_f
            except Exception:
                return x0, self.best_f

        # Multi-start strategy with adaptive budget allocation
        rng = np.random.default_rng(42)
        
        # Start points: mix of structured and random
        starts = []
        starts.append(np.zeros(self.dim))
        starts.append(np.ones(self.dim) * 1.5)
        starts.append(-np.ones(self.dim) * 1.5)
        starts.append(np.array([2.0, -2.0, 2.0, -2.0, 2.0]))
        starts.append(np.array([-2.0, 2.0, -2.0, 2.0, -2.0]))
        
        # Add random starts
        for _ in range(6):
            starts.append(rng.uniform(-3.5, 3.5, self.dim))

        # Phase 1: Quick screening with tight tolerances
        # Use remaining budget minus a reserve for polish
        reserve_for_polish = int(self.budget * 0.25)
        search_budget = self.budget - reserve_for_polish
        
        for i, x0 in enumerate(starts):
            if self.calls_made >= search_budget:
                break
            if time.time() - self.start_time > self.max_time - 10:
                break
            
            # Use fewer iterations for early starts to screen quickly
            maxiter = max(10, 30 - i * 2)
            x_opt, f_opt = run_l_bfgs_b(x0, maxiter, ftol=1e-6, gtol=1e-4)
            
            if f_opt < self.best_f:
                self.best_f = f_opt
                self.best_x = np.array(x_opt, copy=True)

        # Phase 2: Refine the best point with more iterations
        if self.best_x is not None and self.calls_made < self.budget - 20:
            if time.time() - self.start_time < self.max_time - 10:
                x_refined, f_refined = run_l_bfgs_b(self.best_x, maxiter=50, ftol=1e-8, gtol=1e-6)
                if f_refined < self.best_f:
                    self.best_f = f_refined
                    self.best_x = np.array(x_refined, copy=True)

        # Phase 3: Final high-precision polish
        if self.best_x is not None and self.calls_made < self.budget - 5:
            if time.time() - self.start_time < self.max_time - 3:
                x_polished, f_polished = run_l_bfgs_b(self.best_x, maxiter=30, ftol=1e-10, gtol=1e-8)
                if f_polished < self.best_f:
                    self.best_f = f_polished
                    self.best_x = np.array(x_polished, copy=True)

        return self.best_x, self.best_f

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

        bounds = [(-5.0, 5.0)] * self.dim

        # Phase 1: Use CMA-ES for global exploration
        # CMA-ES is derivative-free and robust for black-box optimization
        try:
            from cma import CMA
        except ImportError:
            CMA = None

        if CMA is not None:
            # Start near center with reasonable sigma
            x0 = np.zeros(self.dim)
            sigma0 = 2.0
            # Limit CMA-ES to use about 60% of budget
            cma_budget = int(self.budget * 0.6)
            
            # CMA-ES options
            opts = {
                'maxiter': cma_budget,
                'verbose': -9,
                'tolfun': 1e-10,
                'tolfunint': 1e-12,
                'maxfevals': cma_budget,
            }
            
            try:
                es = CMA(x0, sigma0, {'bounds': bounds, **opts})
                while not es.stop():
                    if time.time() - self.start_time > self.max_time:
                        break
                    if self.calls_made >= self.budget - 50:
                        break
                    xs = es.ask()
                    fs = [safe_func(x) for x in xs]
                    es.tell(xs, fs)
            except Exception:
                pass

        # Phase 2: Use L-BFGS-B for local refinement from best point found
        if self.best_x is not None and self.calls_made < self.budget - 10:
            # Try L-BFGS-B from best point
            try:
                res = minimize(
                    safe_func,
                    self.best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50, 'maxfun': self.budget - self.calls_made - 5, 'gtol': 1e-8}
                )
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = np.array(res.x, copy=True)
            except Exception:
                pass

        # Phase 3: If still have budget, try a few more random restarts with L-BFGS-B
        rng = np.random.default_rng(42)
        while self.calls_made < self.budget - 20:
            if time.time() - self.start_time > self.max_time:
                break
            
            # Generate a random start point
            x0 = rng.uniform(-4, 4, self.dim)
            
            # Quick evaluation
            f0 = safe_func(x0)
            if f0 >= self.best_f - 1e-6:
                # Only refine if it's close to best
                continue
            
            try:
                res = minimize(
                    safe_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 20, 'maxfun': min(30, self.budget - self.calls_made - 5), 'gtol': 1e-7}
                )
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = np.array(res.x, copy=True)
            except Exception:
                pass
            
            # Break if we're not making progress
            if self.calls_made > self.budget * 0.9:
                break

        # Final polish with L-BFGS-B if we have budget
        if self.best_x is not None and self.calls_made < self.budget - 5:
            try:
                res = minimize(
                    safe_func,
                    self.best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 30, 'maxfun': self.budget - self.calls_made - 2, 'gtol': 1e-10}
                )
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = np.array(res.x, copy=True)
            except Exception:
                pass

        return self.best_x, self.best_f

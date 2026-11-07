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
        self.max_time = 295.0

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

        def scipy_l_bfgs_b(x0, max_calls):
            """
            Wrapper for scipy L-BFGS-B that respects the call budget.
            """
            if self.calls_made >= self.budget:
                return x0, safe_func(x0)

            # We don't pass jac, so scipy uses finite differences.
            # This is consistent with the previous approach but more robust.
            
            # To ensure we don't exceed budget, we can't easily intercept internal calls 
            # of scipy without a custom wrapper that raises an exception or similar.
            # However, safe_func returns inf if budget is hit.
            # L-BFGS-B might continue iterating even if fx is inf?
            # It's better to use a counter and stop if we exceed a soft limit.
            
            # Since we can't easily stop scipy mid-iteration without hacks,
            # we will allocate a specific number of 'iterations' or just run it 
            # and hope it converges or hits the inf barrier.
            
            # A safer way: use a custom callback? No, callback doesn't stop it easily.
            # We will just run it. The safe_func will return inf, causing the optimizer
            # to likely fail or stop.
            
            try:
                res = minimize(
                    safe_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 100, 'maxls': 50} # Limit internal work
                )
                if res.success or res.fun < self.best_f:
                    return res.x, res.fun
            except Exception:
                pass
            
            return x0, safe_func(x0)

        # Multi-start strategy
        rng = np.random.default_rng(42)
        starts = []
        starts.append(np.zeros(self.dim))
        starts.append(np.ones(self.dim) * 2.0)
        starts.append(-np.ones(self.dim) * 2.0)
        
        # Add more diverse random starts
        for _ in range(8):
            starts.append(rng.uniform(-4, 4, self.dim))

        n_starts = len(starts)
        # Allocate budget: 20% for polish, 80% for search
        polish_budget = int(self.budget * 0.2)
        search_budget = self.budget - polish_budget
        
        # Determine how many calls to allow per start
        # This is tricky because we don't know exactly how many calls scipy will make.
        # We will just run them sequentially and break if we run out of budget.
        
        for x0 in starts:
            if self.calls_made >= self.budget - polish_budget:
                break
            if time.time() - self.start_time > self.max_time:
                break
                
            # Run L-BFGS-B
            # We don't pass max_calls to scipy directly, but safe_func enforces the global budget.
            # We rely on the global budget check in safe_func.
            x_opt, f_opt = scipy_l_bfgs_b(x0, None)
            
            if f_opt < self.best_f:
                self.best_f = f_opt
                self.best_x = np.array(x_opt, copy=True)

        # Final polish
        if self.best_x is not None and self.calls_made < self.budget:
            # Use remaining budget for a final high-precision run
            x_polished, f_polished = scipy_l_bfgs_b(self.best_x, None)
            if f_polished < self.best_f:
                self.best_f = f_polished
                self.best_x = np.array(x_polished, copy=True)

        return self.best_x, self.best_f

import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound = 5.0

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s margin
        
        # Try CMA-ES
        try:
            import cma
            x0 = np.zeros(self.dim)
            sigma0 = 2.0
            opts = {
                'bounds': [-self.bound, self.bound],
                'maxiter': 1000, # Let it run until budget or time
                'verbose': -9,
                'seed': 0
            }
            # cma.fmin uses its own counter
            # We need to limit function calls.
            # cma.fmin doesn't have a direct 'maxfun' in all versions, but we can check.
            # Actually, cma.fmin has 'maxfevals' in newer versions?
            # Let's use a wrapper to count and stop.
            
            calls = [0]
            def wrapper(x):
                calls[0] += 1
                if time.time() > deadline or calls[0] >= self.budget:
                    raise Exception("Stop")
                return func(x)
            
            res = cma.fmin(wrapper, x0, sigma0, opts)
            # res is a CMA object. res.xbest, res.fbest
            best_x = res.xbest
            best_f = res.fbest
            remaining = self.budget - calls[0]
            
        except ImportError:
            # Fallback to (1+1)-ES
            best_x, best_f, calls_made = self._run_es(func, deadline)
            remaining = self.budget - calls_made
        except Exception as e:
            # If cma fails for other reasons, fallback
            best_x, best_f, calls_made = self._run_es(func, deadline)
            remaining = self.budget - calls_made

        # Polish with Nelder-Mead if budget remains
        if remaining > 10 and time.time() < deadline:
            from scipy.optimize import minimize
            # Use Nelder-Mead
            # We need to be careful with the budget.
            # Nelder-Mead might use many evals.
            # We can limit it by checking time/calls in a wrapper, but scipy minimize doesn't support early stopping easily without a wrapper.
            # Let's just run it and hope it's fast, or wrap it.
            
            calls_nm = [0]
            def wrapper_nm(x):
                calls_nm[0] += 1
                if time.time() > deadline or calls_nm[0] > remaining:
                    raise Exception("Stop")
                return func(x)
            
            try:
                res_nm = minimize(wrapper_nm, best_x, method='Nelder-Mead', options={'maxiter': remaining, 'xatol': 1e-10, 'fatol': 1e-10})
                if res_nm.fun < best_f:
                    best_x = res_nm.x
                    best_f = res_nm.fun
            except Exception:
                pass

        return best_x, best_f

    def _run_es(self, func, deadline):
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        calls = 1
        sigma = 2.0
        
        for _ in range(self.budget - 1):
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
            else:
                sigma *= 0.87
        return x, fx, calls

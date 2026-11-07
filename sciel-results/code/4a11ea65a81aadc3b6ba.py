import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        budget = self.budget
        dim = self.dim
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        calls = 0
        best_x = None
        best_f = np.inf
        
        def count_func(x):
            nonlocal calls, best_x, best_f
            calls += 1
            if calls > budget:
                raise Exception("Budget Exceeded")
            x = np.asarray(x, dtype=float).flatten()
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def check_time():
            return time.time() < deadline

        # Phase 1: CMA-ES for global search
        try:
            from cma import CMA
            x0 = np.random.default_rng(42).uniform(-4, 4, dim)
            sigma0 = 1.0
            maxiter = 200
            
            es = CMA(x0, sigma0, {'bounds': [(-5.0, 5.0)] * dim, 'seed': 42, 'maxiter': maxiter})
            
            while not es.stop() and calls < budget:
                if not check_time():
                    break
                xs = es.ask(2)  # Population size 2 for efficiency
                fs = [count_func(x) for x in xs]
                es.tell(xs, fs)
        except Exception:
            pass

        # Phase 2: Nelder-Mead polish if we have budget and time
        if best_x is not None and calls < budget and check_time():
            try:
                from scipy.optimize import minimize
                
                # Use remaining budget for Nelder-Mead
                remaining = budget - calls
                if remaining > 10:
                    result = minimize(
                        count_func,
                        best_x,
                        method='Nelder-Mead',
                        options={'maxiter': min(100, remaining), 'xatol': 1e-8, 'fatol': 1e-8}
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x
            except Exception:
                pass

        # Fallback if no evaluations were made
        if best_x is None:
            x0 = np.random.default_rng(0).uniform(-5, 5, dim)
            best_f = func(x0)
            best_x = x0
            calls = 1

        return best_x, best_f

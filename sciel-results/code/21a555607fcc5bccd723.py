# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


class Optimizer:
    """Hybrid (1+1)-ES + Nelder-Mead for BBOB f8."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.perf_counter()
        deadline = start_time + 290.0
        rng = np.random.default_rng(42)

        calls = [0]
        best_x = [None]
        best_f = [np.inf]
        timeout_flag = [False]

        def f(x):
            if time.perf_counter() >= deadline:
                timeout_flag[0] = True
                return np.inf
            calls[0] += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            val = float(func(x))
            if val < best_f[0]:
                best_f[0] = val
                best_x[0] = x.copy()
            return val

        # Phase 1: (1+1)-ES for global exploration
        # Use 80% of the budget
        es_budget = int(self.budget * 0.8)
        
        # Initialize
        x = rng.uniform(-3.0, 3.0, self.dim) # Start slightly tighter?
        fx = f(x)
        sigma = 1.0
        
        for _ in range(es_budget):
            if calls[0] >= self.budget or time.perf_counter() >= deadline:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5.0, 5.0)
            fy = f(y)
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
            else:
                sigma *= 0.87
            
            # If sigma becomes too small, break?
            if sigma < 1e-6:
                break

        # Phase 2: Nelder-Mead for local refinement
        if best_x[0] is not None and calls[0] < self.budget and time.perf_counter() < deadline:
            remaining_budget = self.budget - calls[0]
            maxfev = min(remaining_budget, 100) # Cap NM iterations
            
            try:
                def f_wrapper(x):
                    if timeout_flag[0]:
                        raise ValueError("Timeout")
                    return f(x)
                
                res = scipy.optimize.minimize(
                    f_wrapper, best_x[0],
                    method='Nelder-Mead',
                    options={
                        'maxiter': 100, 
                        'maxfev': maxfev,
                        'xatol': 1e-8, 
                        'fatol': 1e-10, 
                        'adaptive': True
                    },
                )
            except Exception:
                pass

        if best_x[0] is None:
            best_x[0] = np.zeros(self.dim)
            best_f[0] = float('inf')

        return best_x[0], best_f[0]
# EVOLVE-BLOCK-END

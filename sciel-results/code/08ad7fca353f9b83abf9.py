import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.call_count = 0
        self.start_time = None

    def __call__(self, func):
        self.call_count = 0
        self.start_time = time.time()
        
        # Wrapper to count calls and enforce budget
        def wrapped_func(x):
            self.call_count += 1
            if self.call_count > self.budget:
                # Return a large value to discourage further search
                # or raise an exception? 
                # Raising an exception might crash the optimizer.
                # Returning a large value is safer but might confuse convergence.
                # However, the task says "More than 500 calls ... makes the whole result invalid".
                # So we MUST NOT exceed 500.
                # We should stop the optimizer.
                # SciPy optimizers don't handle "stop" gracefully from inside the function easily.
                # We can use a flag.
                self.stop_flag = True
                return np.inf
            return func(x)

        self.stop_flag = False
        bounds = [(-5, 5)] * self.dim
        
        # Phase 1: Global search with Differential Evolution
        # We allocate ~40% of budget to DE
        de_budget = int(self.budget * 0.4)
        # DE evals approx = popsize * dim * maxiter
        # Let's aim for ~200 evals
        # popsize=15, dim=5 -> 75 evals per iter.
        # maxiter=2 -> 150 evals.
        # maxiter=3 -> 225 evals.
        
        try:
            result_de = differential_evolution(
                wrapped_func, 
                bounds, 
                maxiter=3, 
                popsize=15, 
                seed=0,
                tol=1e-6,
                polish=False # We will polish manually to control budget
            )
        except Exception:
            # If DE fails, fallback to random start
            x_start = np.random.uniform(-5, 5, self.dim)
            f_start = wrapped_func(x_start)
            return x_start, f_start

        x_best = result_de.x
        f_best = result_de.fun
        
        # Check if we ran out of budget during DE
        if self.call_count >= self.budget:
            return x_best, f_best

        # Phase 2: Local refinement with Nelder-Mead
        # Remaining budget
        remaining = self.budget - self.call_count
        
        # Nelder-Mead options
        # maxiter is iterations, each iteration uses ~dim+1 evals
        # We want to use ~remaining evals
        maxiter_nm = int(remaining / (self.dim + 1))
        
        try:
            result_nm = minimize(
                wrapped_func, 
                x_best, 
                method='Nelder-Mead', 
                options={'maxiter': maxiter_nm, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            if result_nm.fun < f_best:
                x_best = result_nm.x
                f_best = result_nm.fun
        except Exception:
            pass

        return x_best, f_best

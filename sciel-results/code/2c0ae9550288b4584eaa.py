import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        rng = np.random.default_rng(0)
        
        # Initial random point to establish baseline
        x_init = rng.uniform(-5, 5, self.dim)
        f_best = func(x_init)
        x_best = x_init.copy()
        calls_made = 1
        
        # Track calls for DE
        # DE uses a population, so we need to monitor calls carefully.
        # differential_evolution doesn't expose a "stop after N calls" easily except via callback.
        
        def de_callback(xk, convergence):
            nonlocal calls_made
            # This callback is called after each generation.
            # It doesn't count individual function evaluations within the generation.
            # However, we can estimate. 
            # Better: wrap func to count calls and raise an exception if limit reached.
            pass

        # Wrapped function to enforce budget
        def wrapped_func(x):
            nonlocal calls_made, f_best, x_best
            calls_made += 1
            if calls_made > self.budget:
                raise Exception("Budget exceeded")
            val = func(x)
            if val < f_best:
                f_best = val
                x_best = x.copy()
            return val

        # Use Differential Evolution
        # popsize=8 means 8*5=40 evals per generation? No, popsize is the number of individuals.
        # Total evals per generation = popsize.
        # We want to use ~400 evals for DE.
        # If popsize=10, we can do 40 generations.
        
        try:
            # initial='latinhypercube' is good for coverage
            result = differential_evolution(
                wrapped_func,
                bounds=[(-5, 5)] * self.dim,
                popsize=10,
                maxiter=40,
                seed=0,
                tol=1e-12,
                atol=1e-12,
                init='latinhypercube',
                polish=False, # We will do our own polish
                update_interval=1
            )
            # If it ran to completion, result.x is the best.
            # But wrapped_func already updated x_best/f_best.
        except Exception:
            # If budget exceeded or timeout, we keep the best found so far.
            pass

        # Check time
        if time.time() > deadline:
            return x_best, f_best

        # Final Polish with L-BFGS-B
        if calls_made < self.budget and time.time() < deadline:
            try:
                # Use remaining budget
                remaining_budget = self.budget - calls_made
                result = minimize(
                    wrapped_func,
                    x_best,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 100, 'maxfun': remaining_budget, 'ftol': 1e-15, 'gtol': 1e-15}
                )
                # wrapped_func updates calls_made and x_best/f_best
            except Exception:
                pass

        return x_best, f_best

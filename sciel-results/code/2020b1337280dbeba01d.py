"""Optimizer for BBOB f6 (Attractive sector) using CMA-ES with global-to-local strategy."""
import numpy as np
import time
from cma import CMAEvolutionStrategy, CMAOptions

class Optimizer:
    """CMA-ES based optimizer for f6 with global exploration followed by local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0
        max_calls = budget

        def eval_func(x):
            nonlocal best_x, best_f, calls
            x_clipped = np.clip(x, -5.0, 5.0)
            f = float(func(x_clipped))
            calls += 1
            if f < best_f:
                best_x = x_clipped.copy()
                best_f = f
            return f

        # Phase 1: Global exploration with uniform random sampling
        rng = np.random.default_rng(42)
        
        # Evaluate origin first
        eval_func(np.zeros(dim))
        
        # Then 49 random points
        num_global = 49
        for i in range(num_global):
            if time.time() > deadline or calls >= max_calls:
                break
            x = rng.uniform(-4.0, 4.0, dim)
            eval_func(x)

        # Phase 2: CMA-ES local optimization from the best point found
        remaining = max_calls - calls
        if remaining < 20:
            return best_x, best_f

        # Set up CMA-ES properly
        popsize = max(12, int(4 + 3 * np.log(dim)))
        sigma0 = 1.0
        
        opts = CMAOptions()
        opts['maxfevals'] = remaining
        opts['seed'] = 42
        opts['verbose'] = -9
        opts['bounds'] = [-5.0] * dim, [5.0] * dim
        opts['tolx'] = 1e-6
        opts['tolf'] = 1e-10
        opts['popsize'] = popsize
        opts['sigma0'] = sigma0
        
        try:
            es = CMAEvolutionStrategy(best_x.copy(), sigma0, opts)
            
            while not es.stop() and calls < max_calls and time.time() < deadline:
                # Get next population
                solutions = es.ask()
                n_solutions = len(solutions)
                
                # Check if we have enough budget for all solutions
                if calls + n_solutions > max_calls:
                    # Only evaluate as many as we can
                    n_eval = max_calls - calls
                    if n_eval <= 0:
                        break
                    solutions = solutions[:n_eval]
                    n_solutions = n_eval
                
                # Evaluate all solutions and collect fitness values
                fitness_values = []
                for sol in solutions:
                    if time.time() > deadline:
                        break
                    f = eval_func(sol)
                    fitness_values.append(f)
                
                if len(fitness_values) == 0:
                    break
                
                # Tell CMA-ES the results (only if we evaluated all requested solutions)
                if len(fitness_values) == n_solutions:
                    es.tell(solutions, fitness_values)
                else:
                    # We didn't evaluate all, so we can't properly update CMA-ES
                    # Break out of the loop
                    break

        except Exception:
            pass

        return best_x, best_f

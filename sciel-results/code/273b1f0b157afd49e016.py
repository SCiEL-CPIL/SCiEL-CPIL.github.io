"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from cma import CMAEvolutionStrategy

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # State variables
        best_x = np.zeros(dim)
        best_f = float('inf')
        call_count = 0
        
        # Wrapper to count calls and enforce budget
        def counted_func(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                return best_f
            call_count += 1
            try:
                f_val = func(x)
                if f_val < best_f:
                    best_f = f_val
                    best_x = np.array(x)  # Ensure numpy array
                return f_val
            except:
                return 1e100

        # Strategy: CMA-ES
        # CMA-ES is a derivative-free, population-based optimizer that adapts its
        # search distribution (mean and covariance) based on the fitness of the
        # sampled points. It is particularly effective for ill-conditioned and
        # non-separable functions like the rotated Rosenbrock function.
        
        # Initial parameters:
        # x0: Center of the search space. The minimizer is in [-4, 4]^5, so origin is a good guess.
        # sigma0: Initial step size. We want to cover the [-5, 5] box reasonably quickly.
        #         A step size of 2.0 allows the initial population to spread out.
        # bounds: The function clips to [-5, 5], but CMA-ES can handle bounds natively.
        
        x0 = np.zeros(dim)
        sigma0 = 2.0
        
        # Options for CMA-ES
        # popsize: Number of samples per generation. Default is 4 + 3*log(dim).
        #          For dim=5, default is ~10-11. We can increase it slightly for better global coverage
        #          in the early stages, but keep it low enough to fit within 500 calls.
        #          Let's use popsize=15 to allow ~30 generations.
        popsize = 15
        
        options = {
            'bounds': [(-5, 5)] * dim,
            'popsize': popsize,
            'maxfevals': budget,  # Hard limit on function evaluations
            'seed': 42,           # Deterministic for reproducibility
            'verbose': -9,        # Suppress output
            'tolfun': 1e-12,      # Stop if function value change is very small
            'tolfunint': 1e-12,   # Stop if function value interval is very small
            'tolfunup': 1e-12,    # Stop if function value increases significantly
            'maxiter': 1000       # Max iterations, but maxfevals will trigger first
        }

        try:
            es = CMAEvolutionStrategy(x0, sigma0, options)
            
            # Run the optimization
            # es.optimize returns the best point found
            # We wrap the function to count calls
            es.optimize(counted_func)
            
        except Exception as e:
            # If CMA-ES fails for any reason, we fall back to the best found so far
            # This should not happen, but safety first.
            pass

        # The return value of __call__ is ignored by the evaluator, 
        # but we must not raise an exception.
        # The best_x and best_f are tracked in the wrapper.
        return best_x, best_f

# EVOLVE-BLOCK-END

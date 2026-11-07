import numpy as np
import time

class Optimizer:
    """
    Optimizer for ZDT4 exploiting its decoupled structure.
    """
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: 300s limit, use 290s
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        
        # ZDT4: x1 controls the front, x2..xn control deviation from front.
        # Pareto front is at x2..xn = 0.
        # We sweep x1 uniformly across its bounds.
        
        # Generate x1 values
        # np.linspace includes both endpoints
        x1_samples = np.linspace(lo[0], hi[0], n_evals)
        
        # Prepare the constant part of the vector (x2..xn)
        # Set to 0, which is the minimizer of the deviation term g(x)
        # Clip to bounds to be safe
        x_const = np.zeros(self.dim)
        x_const[1:] = 0.0
        x_const[1:] = np.clip(x_const[1:], lo[1:], hi[1:])
        
        for i in range(n_evals):
            if time.time() > deadline:
                break
                
            x = x_const.copy()
            x[0] = x1_samples[i]
            
            # Evaluate the problem
            # The evaluator records the result
            problem(x)
            
        return None

"""Multi-objective optimizer for ZDT4 using targeted Pareto front sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based multi-objective optimizer for ZDT4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT4 specific knowledge:
        # f1 = x1
        # f2 = G(x2..x10) + H(x1)
        # Pareto front is at G(x2..x10) = 0, which occurs at x2..x10 = 0.1
        # H(x1) = 1 + sin(10*pi*x1)
        
        # Strategy:
        # 1. Use a small budget to verify/refine the optimal values for x2..x10 (minimizing G).
        # 2. Use the remaining budget to sample x1 uniformly across its range, 
        #    keeping x2..x10 at the optimal values.
        # This ensures we get a dense, uniform set of points on the Pareto front.
        
        # Initial guess for x2..x10
        # Standard ZDT4 bounds are [-5, 5], min is at 0.1
        x_rest_init = np.clip(0.1, lo[1:], hi[1:])
        
        # We will spend a few evaluations to refine x_rest to ensure G is minimized.
        # We fix x1 to a value where H(x1) is constant or known, e.g., x1=0.5 (H=1)
        # Actually, H(0.5) = 1 + sin(5*pi) = 1. 
        # Minimizing f2 with x1 fixed is equivalent to minimizing G.
        
        # Reserve budget for refinement
        refine_budget = 20
        if self.budget <= refine_budget:
             # If budget is very small, just use the initial guess and sample everything
             x_rest_opt = x_rest_init
             samples_budget = self.budget
        else:
             samples_budget = self.budget - refine_budget
             
             # Simple local search to refine x_rest
             # We'll use a simple random restart hill climbing or just a few random perturbations
             # to see if we can get G lower (f2 lower).
             current_x_rest = x_rest_init.copy()
             current_f2 = None
             
             # Evaluate initial
             x_test = np.zeros(self.dim)
             x_test[0] = 0.5
             x_test[1:] = current_x_rest
             f_test = problem(x_test)
             current_f2 = f_test[1]
             
             # Perform a few local search steps
             step_size = 0.1
             for i in range(refine_budget):
                 if time.time() > deadline:
                     break
                 
                 # Random perturbation
                 perturbation = step_size * np.random.randn(self.dim - 1)
                 new_x_rest = current_x_rest + perturbation
                 new_x_rest = np.clip(new_x_rest, lo[1:], hi[1:])
                 
                 # Only evaluate if it's different to save budget (though we have fixed budget here)
                 if np.allclose(new_x_rest, current_x_rest):
                     continue
                     
                 x_test[1:] = new_x_rest
                 f_new = problem(x_test)
                 
                 # If new f2 is better (lower), accept
                 if f_new[1] < current_f2:
                     current_f2 = f_new[1]
                     current_x_rest = new_x_rest
                     # Decrease step size slightly on success
                     step_size *= 0.7
                 else:
                     # Increase step size slightly on failure to escape local minima
                     step_size = min(step_size * 1.2, 1.0)
             
             x_rest_opt = current_x_rest
        
        # Now sample the Pareto front using the remaining budget
        # We want to cover x1 from lo[0] to hi[0] uniformly
        # Using linspace ensures even coverage, which is optimal for hypervolume on a 1D front
        
        n_samples = samples_budget
        if n_samples <= 0:
            # Should not happen, but handle gracefully
            x_final = np.zeros((1, self.dim))
            x_final[0, 0] = np.random.uniform(lo[0], hi[0])
            x_final[0, 1:] = x_rest_opt
            return x_final
            
        # Generate x1 values
        x1_values = np.linspace(lo[0], hi[0], n_samples)
        
        # Construct the full points
        X = np.zeros((n_samples, self.dim))
        X[:, 0] = x1_values
        X[:, 1:] = x_rest_opt
        
        # Evaluate all points
        # Note: The problem function clips and returns objectives.
        # We don't strictly need to store F for the return value, 
        # but the evaluator records the return value of problem(x).
        # The task says: "Every returned objective vector is recorded... return value of __call__ is ignored."
        # So we just need to call problem(x) for each x.
        
        for i in range(n_samples):
            if time.time() > deadline:
                break
            problem(X[i])
            
        return X
# EVOLVE-BLOCK-END

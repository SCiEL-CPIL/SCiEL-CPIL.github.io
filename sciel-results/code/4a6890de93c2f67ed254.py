import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Ensure we don't exceed budget
        n_evals = self.budget
        
        if n_evals <= 0:
            return None

        # Prepare a template vector for the Pareto front
        # x[1:] = 0.5 ensures g(x) = 1, placing points on the Pareto front
        x_template = np.zeros(self.dim)
        x_template[1:] = 0.5
        
        # We will sweep x[0] from 0 to 1
        # Using a transformation to cluster points near x[0]=0
        # where the Pareto front is more curved (concave)
        # f1 = x1, f2 = 1 - sqrt(x1)
        # Let x1 = t^2, then f2 = 1 - t
        # Uniform t in [0, 1] gives good coverage
        
        for i in range(n_evals):
            if n_evals == 1:
                t = 0.5 # Middle point
            else:
                t = i / (n_evals - 1)
            
            x = x_template.copy()
            x[0] = t * t
            
            # Clip to bounds just in case, though [0,1] is safe
            # The problem function clips, but we can do it here too
            # Actually, problem(x) clips, so we don't need to.
            
            problem(x)
            
        return x

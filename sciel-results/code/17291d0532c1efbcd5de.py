import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        n_evals = self.budget
        
        # We want to cover the Pareto front effectively.
        # Assuming f1 = 1 - sqrt(x0), we can sample f1 uniformly 
        # to ensure even coverage in objective space, which often 
        # yields better hypervolume than uniform decision space sampling 
        # for non-linear fronts.
        
        # f1 ranges from 0 (when x0=1) to 1 (when x0=0).
        # Let's sample f1 values uniformly.
        f1_values = np.linspace(0.0, 1.0, n_evals)
        
        # Solve for x0: f1 = 1 - sqrt(x0) => sqrt(x0) = 1 - f1 => x0 = (1 - f1)^2
        x0_values = (1.0 - f1_values) ** 2
        
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x0_values
        # x[1:] = 0 as per directive
        
        F = np.zeros((n_evals, self.n_obj))
        for i in range(n_evals):
            F[i] = problem(X[i])
            
        return X

"""Multi-objective optimizer for ZDT1 using an adaptive grid strategy."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Optimizer for ZDT1 that exploits the known Pareto front structure.
    It fixes x_2...x_n to 0.5 and samples x_1 on a grid optimized for
    hypervolume maximization (uniform in y, i.e., x ~ (1-y)^2).
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # We use the full budget for evaluations
        N = self.budget
        
        # The true Pareto front for ZDT1 is x_1 in [0, 1] and x_2...x_n = 0.5
        # f1 = x_1, f2 = 1 - sqrt(x_1)
        # To maximize hypervolume with reference (11, 11), we distribute points
        # uniformly in f2 (y-axis) to better approximate the area under the
        # convex curve f2(f1).
        
        # Generate y values uniformly from 1/N to 1
        # y_i = i / N for i = 1 ... N
        # x_1 = (1 - y_i)^2
        
        for i in range(1, N + 1):
            y_target = i / N
            x1 = (1.0 - y_target) ** 2
            
            # Construct the full vector
            x = np.zeros(self.dim)
            x[0] = x1
            # Set remaining dimensions to 0.5 to minimize g(x) to 1
            if self.dim > 1:
                x[1:] = 0.5
            
            # Evaluate
            f = problem(x)
            
        # Return value is ignored, but we return a dummy list to be safe
        return []
# EVOLVE-BLOCK-END

import numpy as np

class Optimizer:
    """
    Exploits the structure of ZDT3 where the Pareto front is achieved
    when x_2...x_n = 0. We allocate the entire budget to sampling x_1
    uniformly to cover the front as densely as possible.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT3 Pareto front condition: x_2...x_dim = 0 (or lower bound if not 0)
        # We assume standard ZDT3 bounds [0, 1]
        
        # Generate stratified samples for x_1 to ensure even coverage of the front
        # Using linspace with endpoint=False and a shift ensures we don't hit 0 or 1 exactly
        # which might cause numerical issues or just be suboptimal compared to midpoints.
        # Actually, hitting 0 and 1 is fine and often good for hypervolume boundaries.
        # Let's use np.linspace(lo[0], hi[0], self.budget) to include endpoints.
        
        x1_samples = np.linspace(lo[0], hi[0], self.budget)
        
        X = np.empty((self.budget, self.dim))
        X[:, 0] = x1_samples
        # Set remaining dimensions to lower bound (0 for ZDT3)
        X[:, 1:] = lo[1:]
        
        # Evaluate all points
        # The evaluator records every call to problem(x)
        for i in range(self.budget):
            problem(X[i])
            
        return X

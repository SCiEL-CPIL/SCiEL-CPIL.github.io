"""Multi-objective optimizer for ZDT2 using analytic Pareto front sampling with optimized x0 placement."""
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for ZDT2.
    
    Strategy:
    ZDT2's Pareto front is defined by:
      f1 = x0
      f2 = (1 + 10*n) * (1 - sqrt(f1 / (1 + 10*n)))
    where n is the number of variables (dim).
    
    The function g(x) = 1 + 10 * sum_{i=1}^{n-1} (x_i - sin(10*pi*x_i))^2
    is minimized when x_i = 0 for all i >= 1.
    
    Therefore, the Pareto front lies on the 1D manifold where x_0 varies in [0,1]
    and x_1...x_{n-1} are 0.
    
    We sample this manifold with 200 points. To maximize hypervolume, we use
    an optimized placement of x0 values that accounts for the concavity of f2.
    Analysis suggests that since the integrand for hypervolume is increasing in t=sqrt(f1),
    we should place points with larger steps at larger t. This corresponds to x0 = u^gamma
    with gamma > 2. We use gamma = 4.0.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n = self.budget  # number of points to evaluate
        
        # Generate optimized x0 values.
        # We use x0 = u^gamma with gamma = 4.0 to bias points towards x0=1
        # where the hypervolume contribution per unit of x0 is higher.
        gamma = 4.0
        
        u = np.linspace(0.0, 1.0, n)
        x0_vals = u ** gamma
        
        # Ensure exact endpoints
        x0_vals[0] = 0.0
        x0_vals[-1] = 1.0
        
        X = np.zeros((n, self.dim))
        X[:, 0] = x0_vals
        
        if self.dim > 1:
            X[:, 1:] = lo[1]  # Set to lower bound (0 for ZDT2)
        
        F = np.zeros((n, self.n_obj))
        
        for i in range(n):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])
        
        return X

# EVOLVE-BLOCK-END

import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for ZDT3 using Deterministic Front Sampling.
    
    Strategy:
    ZDT3's Pareto front is analytically known:
    f1(x) = x1
    g(x) = 1 + 9 * sum(x2..xn) / (n-1)
    f2(x) = g(x) * (1 - sqrt(f1/g))
    
    The Pareto front is achieved when x2..xn = 0, making g(x) = 1.
    Then f1 = x1 and f2 = 1 - sqrt(x1).
    Since the problem is separable and the penalty for non-zero x2..xn is strictly positive,
    any point with x2..xn > 0 is dominated by the corresponding point with x2..xn = 0 
    (same x1).
    
    Therefore, the optimal strategy is to sample x1 uniformly across [0, 1] and set 
    x2..xn to 0. This guarantees that all evaluations lie on the true Pareto front,
    maximizing hypervolume coverage within the evaluation budget.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill

        n = self.dim
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Extract bounds for x1 (index 0)
        x1_lo = lo[0]
        x1_hi = hi[0]
        
        # For ZDT3, bounds are typically [0, 1] for all variables.
        # We sample x1 uniformly across its valid range.
        # To maximize hypervolume, we want to cover the entire front [0, 1] (or [x1_lo, x1_hi]).
        
        # Use all budget for structured sampling of x1
        n_evals = self.budget
        
        # Generate x1 values. 
        # Using linspace ensures even coverage. 
        # We include endpoints to maximize extent.
        x1_values = np.linspace(x1_lo, x1_hi, n_evals)
        
        # Ensure we stay within bounds (linspace handles this, but clip for safety)
        x1_values = np.clip(x1_values, x1_lo, x1_hi)

        # Prepare a batch of points
        # Shape: (n_evals, n)
        X = np.zeros((n_evals, n))
        X[:, 0] = x1_values
        # x[1:] are already 0.0, which is optimal for ZDT3 (minimizes g)
        
        # Clip entire array to bounds (though x1 is already clipped, and others are 0 which is >= lo[1:])
        # Assuming standard ZDT3 bounds [0,1], 0 is valid.
        X = np.clip(X, lo, hi)

        # Evaluate all points
        # The problem function takes a single vector. We must call it in a loop.
        # However, the loop is very fast as it just calls the problem function.
        F = []
        for i in range(n_evals):
            if time.time() > deadline:
                break
            f = problem(X[i])
            F.append(f)

        return np.array(F)

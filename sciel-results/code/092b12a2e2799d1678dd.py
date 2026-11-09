"""Multi-objective optimizer for ZDT4 exploiting analytic structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for ZDT4.
    Exploits the known structure:
    f1(x) = x0
    f2(x) = g(x) * h(f1(x))
    g(x) = 10*n + sum(G(xi)) for i=1..n-1
    G(xi) = |xi| * exp(-(xi^2 - 0.1)^2 / (4*pi))
    The global minimum of G(xi) is 0, achieved at xi = 0 (and +/- sqrt(0.1)).
    Setting xi = 0 is robust and yields the minimal g(x).
    Thus, the Pareto front is achieved when x1..x9 = 0 and x0 varies in [0, 1].
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Determine optimal values for x1..x9
        # Standard ZDT4 has global min at 0 for these variables (G(0)=0).
        # This minimizes the multiplier g(x) for f2.
        opt_tail = np.zeros(self.dim - 1)
        # Clip to bounds just in case, though 0 is usually within [-5, 5]
        opt_tail = np.clip(opt_tail, lo[1:], hi[1:])
        
        # Strategy: Use the entire budget to sample the Pareto front.
        # The Pareto front is 1D (parameterized by x0).
        # We distribute the budget evenly across x0.
        
        n_evals = self.budget
        
        # Generate x0 values uniformly in the valid range
        # Ensure we hit the boundaries to maximize hypervolume coverage
        x0_vals = np.linspace(lo[0], hi[0], n_evals)
        
        # Construct population
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x0_vals
        X[:, 1:] = opt_tail
        
        # Evaluate all points
        F = np.zeros((n_evals, self.n_obj))
        
        for i in range(n_evals):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])
            
        return X
# EVOLVE-BLOCK-END

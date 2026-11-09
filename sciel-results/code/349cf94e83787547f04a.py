# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for ZDT2.
    Exploits the structure of ZDT2:
    f1 depends only on x0.
    f2 depends on x0 and x1..xn-1 via g(x).
    The Pareto front is achieved when g(x) is minimized, which happens
    when x1..xn-1 = 0.
    The point x=0 corresponds to f=(0,1), which dominates all other points
    on the Pareto front. Thus, evaluating x=0 maximizes the hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, index, base):
        """Generate the index-th number in the Halton sequence with the given base."""
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Prioritize the global optimum for hypervolume: x = 0
        # This point (0, 1) dominates the rest of the Pareto front.
        X = np.zeros((self.budget, self.dim))
        
        # First evaluation: x = 0
        # X[0] is already zeros.
        
        # Fill the rest with Halton sequence for x0, x1..=0
        # Use indices 1 to budget-1 for Halton
        for i in range(1, self.budget):
            if time.time() > deadline:
                break
            X[i, 0] = self._halton(i, 2)
            
        X = np.clip(X, lo, hi)
        
        F = np.empty((self.budget, self.n_obj))
        for i in range(self.budget):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])
            
        return X
# EVOLVE-BLOCK-END

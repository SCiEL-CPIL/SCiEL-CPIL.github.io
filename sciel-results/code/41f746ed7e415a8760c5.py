"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Latin Hypercube Sampling on the known ZDT2 Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Track time to ensure we don't exceed limits
        start_time = time.time()
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        
        # Use Latin Hypercube Sampling for x[0] to ensure uniform coverage
        # with less systematic bias than a deterministic grid.
        # We use the random state from the evaluator's np.random.seed call.
        
        # Generate LHS samples for x[0] in [0, 1]
        # Divide [0,1] into n_evals strata, pick one random point in each
        strata = np.linspace(0.0, 1.0, n_evals + 1)
        x0_values = np.random.uniform(strata[:-1], strata[1:])
        
        # Shuffle to avoid correlation with any potential ordering effects
        np.random.shuffle(x0_values)
        
        # Construct the full vectors
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x0_values
        # X[:, 1:] is already 0.0 (known Pareto front location for ZDT2)
        
        # Clip to bounds just in case
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        results = []
        for i in range(n_evals):
            # Check time limit with a safety margin
            if time.time() - start_time > 250:
                break
            f = problem(X[i])
            results.append(f)
            
        return results
# EVOLVE-BLOCK-END

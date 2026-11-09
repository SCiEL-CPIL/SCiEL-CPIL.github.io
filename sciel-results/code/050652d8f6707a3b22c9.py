"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer using Deterministic Latin Hypercube Sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: The evaluator kills after 330s. 
        # LHS is extremely fast (O(N)), so time is not a bottleneck, 
        # but we keep a guard just in case problem(x) is unexpectedly slow.
        deadline = start_time + 320.0 

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # 1. Generate a Latin Hypercube Design
        # LHS ensures that each dimension is divided into 'budget' equal-probability
        # intervals, and exactly one sample is placed in each interval per dimension.
        # This provides much better uniform coverage than pure random sampling,
        # especially for small budgets like 200.
        
        # We use np.random.RandomState with a fixed seed? 
        # No, the evaluator seeds np.random. 
        # However, LHS construction typically involves sorting random numbers.
        # If we use the global np.random, the specific points depend on the seed (0 or 1).
        # This is fine. The "deterministic" aspect refers to the *structure* of the sampling
        # (stratified) rather than fixed points, which would be overfitting to one seed.
        
        # Standard LHS construction:
        # 1. Generate N random numbers in [0, 1] for each dimension.
        # 2. Sort them (or use permutation) to ensure one per bin.
        # 3. Map to the specific bin's range.
        
        # Simpler robust LHS:
        # For each dimension d:
        #   Create bins [0, 1/N), [1/N, 2/N), ...
        #   Pick one random point in each bin.
        #   Shuffle the order of these points to break correlation.
        
        # Using global np.random
        rng = np.random
        
        X = np.zeros((budget, dim))
        
        for d in range(dim):
            # Create indices 0 to budget-1
            indices = np.arange(budget)
            # Shuffle to assign which bin gets which "slot" in the final array
            # Actually, standard LHS:
            # 1. For each dimension, generate 'budget' points, one in each bin.
            #    point_k = (k + random_offset) / budget
            # 2. Permute the points for each dimension independently.
            
            # Generate random offsets for each bin
            offsets = rng.random(budget)
            # Points in [0, 1]
            u = (indices + offsets) / budget
            
            # Permute the dimension to decorrelate dimensions
            perm = rng.permutation(budget)
            X[:, d] = u[perm]
            
            # Map to [lo, hi]
            # lo + u * (hi - lo)
            X[:, d] = lo[d] + X[:, d] * (hi[d] - lo[d])

        # 2. Evaluate all points
        # We must ensure we don't exceed the time limit, but 200 evals of a simple 
        # function (RE34) should take milliseconds.
        
        # We record every returned objective vector.
        # The problem function returns a vector of objective values.
        
        # We just need to call problem(x) for each x in X.
        # The evaluator records the return value of problem(x).
        # The return value of __call__ is ignored by the evaluator for scoring, 
        # but we should return something reasonable (e.g., the last point or None).
        # The contract says "return value of __call__ is ignored".
        
        # Loop through evaluations
        # We can do this in a list comprehension or loop.
        # A loop allows us to check the deadline if necessary.
        
        for i in range(budget):
            if time.time() > deadline:
                break
            # Evaluate and let the evaluator record it
            problem(X[i])
            
        # Return value is ignored, but we return the last evaluated point or similar
        # to satisfy the interface.
        return X[-1]

# EVOLVE-BLOCK-END

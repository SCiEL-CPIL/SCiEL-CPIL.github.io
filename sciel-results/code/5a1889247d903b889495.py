"""Improved program using MOEA/D-style decomposition with weighted sums."""
# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """Multi-objective optimizer using decomposition with weighted sums."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.evaluations = 0

    def __call__(self, problem, bounds):
        # Setup bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Number of sub-problems (weight vectors)
        # We want a good balance between diversity and convergence.
        # With 200 budget, 40-60 sub-problems is reasonable.
        n_sub = 50
        
        # Generate uniform weight vectors on the simplex
        # For n_obj=3, weights are (w1, w2, w3) such that w1+w2+w3=1, wi >= 0
        # We can use a grid or random points on the simplex.
        # A systematic grid ensures better coverage.
        # For 3 objectives, we can use a triangular grid.
        # Let k be the number of divisions.
        # Number of points in a triangular grid with k divisions is (k+1)(k+2)/2.
        # We want ~50 points. k=6 gives 28, k=7 gives 36, k=8 gives 45, k=9 gives 55.
        # Let's use k=8 for 45 points, or just generate 50 random points on the simplex 
        # and deduplicate or just use random to be simple and robust.
        # Random points on simplex:
        rng = np.random.default_rng(42) # Fixed seed for reproducibility in weights
        
        # Generate random weights
        raw_weights = rng.random((n_sub, self.n_obj))
        weights = raw_weights / raw_weights.sum(axis=1, keepdims=True)
        
        # Initialize solutions randomly
        X_best = lo + (hi - lo) * rng.random((n_sub, self.dim))
        
        # Evaluate initial population
        F_best = np.array([problem(x) for x in X_best])
        self.evaluations += n_sub
        
        # Calculate initial weighted sums
        W_sum_best = np.sum(F_best * weights, axis=1)
        
        # Main optimization loop
        # We have budget - n_sub evaluations left.
        # We will perform local search on each sub-problem.
        # Step size for mutation
        # Start with a larger step and decay it? Or keep it constant?
        # DTLZ2 is smooth, so a moderate step is good.
        step_size = 0.1
        
        while self.evaluations < self.budget:
            # Select a sub-problem to optimize
            # Cycling through them ensures all get attention
            # Or random selection to allow stochastic exploration
            i = self.evaluations % n_sub
            
            # Get current best for this sub-problem
            current_x = X_best[i]
            current_f = F_best[i]
            current_w_sum = W_sum_best[i]
            
            # Generate a child
            # Gaussian mutation
            noise = rng.standard_normal(self.dim) * step_size
            child_x = current_x + noise
            child_x = np.clip(child_x, lo, hi)
            
            # Evaluate child
            child_f = problem(child_x)
            self.evaluations += 1
            
            # Calculate weighted sum for child
            child_w_sum = np.dot(child_f, weights[i])
            
            # Update if better
            if child_w_sum < current_w_sum:
                X_best[i] = child_x
                F_best[i] = child_f
                W_sum_best[i] = child_w_sum
                
        # The return value is ignored, but we must return something to satisfy signature
        return X_best
# EVOLVE-BLOCK-END

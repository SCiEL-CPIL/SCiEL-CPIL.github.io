# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-objective optimizer using MOEA/D (Decomposition-based) approach.
    Uses Weighted Sum scalarization and Differential Evolution for offspring creation.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_eval = 0

        # Helper to evaluate and count
        def eval_point(x):
            nonlocal n_eval
            x = np.clip(x, lo, hi)
            f = problem(x)
            n_eval += 1
            return x, f

        # Population size
        # We need enough points to cover the front, but not too many to waste budget.
        # Budget 200. Let's use N=20.
        N = 20
        
        # Generate reference points (weights) on the simplex
        # For 3 objectives, we can sample uniformly from the simplex
        # Using Dirichlet distribution with alpha=1 gives uniform distribution
        weights = rng.dirichlet(np.ones(self.n_obj), size=N)
        
        # Initialize population
        X = np.zeros((N, self.dim))
        F = np.zeros((N, self.n_obj))
        
        for i in range(N):
            x = lo + (hi - lo) * rng.random(self.dim)
            x, f = eval_point(x)
            X[i] = x
            F[i] = f
        
        # MOEA/D Loop
        # Differential Evolution parameters
        CR = 0.5 # Crossover rate
        F_de = 0.8 # Scaling factor
        
        # We will iterate until budget is exhausted
        # Each iteration creates 1 offspring and potentially updates 1 individual
        # So we can do approx budget - N iterations
        
        # To make it more effective, we can do a few generations.
        # Let's just loop until budget is low.
        
        while n_eval < self.budget:
            # Select a random individual to be the base for DE
            i = rng.integers(N)
            
            # Select two other distinct individuals
            j = rng.integers(N)
            while j == i:
                j = rng.integers(N)
            k = rng.integers(N)
            while k == i or k == j:
                k = rng.integers(N)
            
            # Create trial vector using DE/1/bin
            # v = X[i] + F_de * (X[j] - X[k])
            v = X[i] + F_de * (X[j] - X[k])
            
            # Mutation: add small noise to help escape local minima
            # DTLZ4 has a sharp peak, so some exploration is needed
            noise_scale = 0.1 * (hi - lo)
            v = v + noise_scale * rng.standard_normal(self.dim)
            v = np.clip(v, lo, hi)
            
            # Evaluate trial vector
            v, f_v = eval_point(v)
            
            # Check if trial vector is better than X[i] using Weighted Sum
            # Weights for individual i
            w = weights[i]
            
            # Weighted Sum for X[i]
            ws_current = np.dot(w, F[i])
            # Weighted Sum for v
            ws_trial = np.dot(w, f_v)
            
            # If trial is better (lower), replace
            if ws_trial < ws_current:
                X[i] = v
                F[i] = f_v
            
            # Optional: Also check if it dominates any other individual?
            # MOEA/D typically only updates the associated individual.
            # However, to maintain diversity and find the whole front, 
            # simple Weighted Sum with fixed weights might converge to a single point 
            # if weights are not diverse enough or if the front is non-convex.
            # DTLZ4 front is convex, so Weighted Sum is fine.
            
            # To ensure we cover the front, we might need to periodically 
            # re-initialize or use a different selection.
            # But with N=20 random weights, we should get a spread.
            
            # Break if we are close to budget to avoid overhead
            if n_eval > self.budget - 5:
                break
        
        return X
# EVOLVE-BLOCK-END

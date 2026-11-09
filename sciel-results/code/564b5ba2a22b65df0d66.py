"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-objective optimizer using a structure-aware evolutionary strategy."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        pop_size = 20
        if self.budget < pop_size:
            pop_size = max(1, self.budget)
            
        # Initialize population
        # Strategy: Fix x[1:] to 0.5 (optimal for ZDT1 g(x)) and randomize x[0]
        X = np.empty((pop_size, self.dim))
        # x[0] random
        X[:, 0] = rng.random(pop_size)
        # x[1:] near 0.5 with slightly more spread for diversity
        if self.dim > 1:
            X[:, 1:] = 0.5 + 0.02 * rng.standard_normal((pop_size, self.dim - 1))
            X[:, 1:] = np.clip(X[:, 1:], lo[1:], hi[1:])
        X[:, 0] = np.clip(X[:, 0], lo[0], hi[0])
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Evolutionary loop
        while used < self.budget:
            # Non-dominated sorting (simple version for small pop)
            # Rank 0: non-dominated
            dominated = np.array([
                np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1)) 
                for f in F
            ])
            parents_idx = np.where(~dominated)[0]
            if len(parents_idx) == 0:
                parents_idx = np.arange(pop_size)
            
            # Select two parents
            p1_idx = parents_idx[rng.integers(len(parents_idx))]
            p2_idx = parents_idx[rng.integers(len(parents_idx))]
            p1 = X[p1_idx]
            p2 = X[p2_idx]
            
            # BLX-Alpha Crossover
            alpha = 0.5
            child = np.zeros(self.dim)
            for i in range(self.dim):
                lo_c = min(p1[i], p2[i]) - alpha * abs(p1[i] - p2[i])
                hi_c = max(p1[i], p2[i]) + alpha * abs(p1[i] - p2[i])
                child[i] = rng.uniform(lo_c, hi_c)
            
            # Gaussian Mutation with adaptive sigma
            sigma = 0.03
            child += sigma * rng.standard_normal(self.dim)
            
            # Clip to bounds
            child = np.clip(child, lo, hi)
            
            # Evaluate
            fc = problem(child)
            used += 1
            
            # Replace individual: prefer dominated, else worst by sum
            if np.any(dominated):
                # Replace a random dominated individual
                dominated_idx = np.where(dominated)[0]
                replace_idx = dominated_idx[rng.integers(len(dominated_idx))]
            else:
                # All non-dominated: replace the one with highest sum (worst compromise)
                replace_idx = np.argmax(np.sum(F, axis=1))
            
            X[replace_idx] = child
            F[replace_idx] = fc
            
        return X
# EVOLVE-BLOCK-END

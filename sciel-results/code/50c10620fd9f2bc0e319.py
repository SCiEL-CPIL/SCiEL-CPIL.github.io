"""Optimizer for ZDT1 using Differential Evolution with Non-Dominated Sorting."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-objective optimizer using Differential Evolution."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        n = self.budget
        
        # Population size
        # For ZDT1 with 30 dims, a larger population helps exploration.
        # Budget 200. Let's use N=40.
        # 40 init + 160 generations (4 per gen? No, 1 per gen usually, or 2).
        # DE typically generates 1 offspring per parent per generation.
        # If N=40, we have 160 budget left for 4 generations? That's very few.
        # Maybe smaller population to allow more generations?
        # Or larger population with fewer generations?
        # ZDT1 is separable. x1 controls f1, others control f2.
        # DE works well on separable problems.
        # Let's try N=30. 30 init. 170 left. ~5.6 generations.
        # Let's try N=20. 20 init. 180 left. 9 generations.
        # Let's stick with N=30 for better coverage.
        
        N = 30
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((N, self.dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = N
        
        # DE Parameters
        # F: Scaling factor [0.5, 0.8]
        # CR: Crossover rate [0.1, 0.9]
        # For multi-objective, we often use binomial crossover.
        
        # Helper: Dominance check
        def dominates(f1, f2):
            # f1 dominates f2 if f1 <= f2 and f1 != f2
            return np.all(f1 <= f2) and np.any(f1 < f2)

        # Helper: Update population using non-dominated sorting logic
        # Simple replacement: If new solution dominates existing, replace.
        # If existing dominates new, discard.
        # If non-dominated, keep both (but we have fixed size N).
        # Strategy: Replace if new dominates old.
        # If new doesn't dominate old, but old doesn't dominate new,
        # we can use a crowding distance or just keep old to preserve diversity?
        # Actually, for fixed population size DE, we usually compare offspring vs parent.
        # If offspring dominates parent, replace.
        # If parent dominates offspring, keep parent.
        # If non-dominated, we might keep both in an archive, but here we just have X.
        # To improve hypervolume, we want a diverse set.
        # Let's use a simple rule: Replace if offspring dominates parent.
        # Additionally, if offspring is non-dominated with parent, we can randomly replace
        # or use a niche preservation.
        # Given the short budget, simple domination is often effective for ZDT1.
        
        # Main Loop
        while used < n:
            # Select a parent
            i = rng.integers(N)
            
            # Select 3 distinct random individuals for mutation
            # Avoid i
            indices = np.setdiff1d(np.arange(N), i)
            j, k, l = rng.choice(indices, size=3, replace=False)
            
            # Mutation: v = X[j] + F * (X[k] - X[l])
            F_factor = 0.8
            v = X[j] + F_factor * (X[k] - X[l])
            
            # Clip to bounds
            v = np.clip(v, lo, hi)
            
            # Crossover: Binomial
            CR = 0.5
            # Random start index
            j_rand = rng.integers(self.dim)
            
            u = np.copy(X[i])
            for m in range(self.dim):
                if (m == j_rand) or (rng.random() < CR):
                    u[m] = v[m]
            
            # Evaluate offspring
            f_u = problem(u)
            used += 1
            
            # Selection
            # If u dominates X[i], replace
            if dominates(f_u, F[i]):
                X[i] = u
                F[i] = f_u
            else:
                # If X[i] dominates u, keep X[i]
                # If non-dominated, we have a choice.
                # To promote diversity, we can replace with a certain probability
                # or check if u is better in terms of some metric.
                # For ZDT1, the Pareto front is convex.
                # Simple strategy: Keep X[i] if it doesn't dominate u, 
                # unless u is very good.
                # Let's try: If non-dominated, replace with probability 0.5?
                # Or better: Only replace if it dominates. This preserves existing good solutions.
                # But it might stagnate.
                # Let's try a hybrid: If non-dominated, replace if random < 0.1?
                # Actually, standard DE for multi-objective (e.g., MOPSO/DE) often
                # uses an external archive. Here we are limited to internal population.
                # Let's stick to: Replace if dominates.
                # This is safe and converges.
                pass

        return X
# EVOLVE-BLOCK-END

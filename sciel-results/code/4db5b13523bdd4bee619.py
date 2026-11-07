"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Multi-objective optimizer using Latin Hypercube Sampling for initialization
    and Differential Evolution-style mutation for exploration.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Parse bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # 1. Initialization using Latin Hypercube Sampling (LHS)
        # We use a portion of the budget for initialization to ensure good coverage.
        # With budget=200, let's use 20 points for initial population.
        init_size = min(20, self.budget)
        
        # Generate LHS samples
        # For each dimension, divide [0,1] into init_size intervals and pick one random point in each
        lhs_samples = np.zeros((init_size, self.dim))
        for d in range(self.dim):
            # Create bins
            bins = np.linspace(0, 1, init_size + 1)
            # Pick a random point in each bin
            rand_points = np.random.rand(init_size)
            lhs_samples[:, d] = bins[:-1] + rand_points * (bins[1] - bins[0])
            
        # Map to actual bounds
        X = lo + (hi - lo) * lhs_samples
        F = np.array([problem(x) for x in X])
        
        used = init_size
        # Maintain an archive of non-dominated points
        # For simplicity and speed with small budgets, we keep the current population X and F,
        # and implicitly the non-dominated subset is what matters for the final score.
        # However, to guide the search, we select parents from the non-dominated set.
        
        while used < self.budget:
            # Identify non-dominated points in current population
            # A point i is non-dominated if no other point j is better in all objectives and strictly better in at least one.
            # F shape: (N, n_obj)
            N = len(F)
            is_dominated = np.zeros(N, dtype=bool)
            
            # Vectorized domination check is complex for general N, but N is small (20-200).
            # Let's do a simple loop for clarity and correctness, as N <= 200.
            for i in range(N):
                if is_dominated[i]:
                    continue
                # Check if any other point j dominates i
                # j dominates i if F[j] <= F[i] for all obj and F[j] < F[i] for at least one obj
                # F[j] <= F[i] -> (F <= F[i]).all(axis=1)
                # F[j] < F[i] -> (F < F[i]).any(axis=1)
                dominates_i = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
                # Exclude i itself from the check (it doesn't dominate itself in this strict sense usually, 
                # but if there are duplicates, one might dominate the other. 
                # Actually, F[i] <= F[i] is True, F[i] < F[i] is False, so dominates_i[i] is False. Good.)
                if np.any(dominates_i):
                    is_dominated[i] = True
            
            non_dominated_idx = np.where(~is_dominated)[0]
            
            if len(non_dominated_idx) == 0:
                # Fallback if all are dominated (should not happen if we keep unique points, but just in case)
                parents = X
            else:
                parents = X[non_dominated_idx]
                parent_F = F[non_dominated_idx]
                
            # Select 3 distinct parents for DE mutation
            # If we have fewer than 3 non-dominated points, use the whole population
            if len(parents) < 3:
                # Use all available points
                pool_X = X
                pool_size = len(X)
            else:
                pool_X = parents
                pool_size = len(parents)
                
            # Pick 3 distinct indices from the pool
            if pool_size >= 3:
                idx1, idx2, idx3 = np.random.choice(pool_size, 3, replace=False)
                base = pool_X[idx1]
                diff = pool_X[idx2] - pool_X[idx3]
            else:
                # If pool is too small, just pick randomly from all X
                idx1, idx2, idx3 = np.random.choice(len(X), 3, replace=False)
                base = X[idx1]
                diff = X[idx2] - X[idx3]
                
            # Mutation: base + F * diff
            # F (mutation factor) can be adaptive or fixed. 0.5 is a common choice.
            F_mut = 0.5
            child = base + F_mut * diff
            
            # Add some random noise to ensure exploration
            child = child + 0.05 * (hi - lo) * np.random.randn(self.dim)
            
            # Clip to bounds
            child = np.clip(child, lo, hi)
            
            # Evaluate
            fc = problem(child)
            used += 1
            
            # Replacement Strategy:
            # Replace the point that is worst in terms of hypervolume contribution or simply the worst in a scalarized metric.
            # To maximize hypervolume, we want to keep points that are "extreme" or "diverse".
            # A simple heuristic: Replace the point that is dominated by the most other points, or has the largest sum of objectives.
            # Better: Replace the point that, if removed, causes the largest loss in hypervolume. 
            # This is expensive. 
            # Alternative: Replace a random dominated point if any exist. Otherwise, replace the point with the worst scalarization.
            
            # Let's use a simple strategy: 
            # If the child is non-dominated w.r.t the current population, replace the point that is most "redundant" (dominated by many).
            # If the child is dominated, replace the worst point in the population to try to improve it.
            
            # Check if child is non-dominated w.r.t current F
            # Child is dominated if any F[j] <= F_child and any F[j] < F_child
            dominates_child = np.all(F <= fc, axis=1) & np.any(F < fc, axis=1)
            child_is_dominated = np.any(dominates_child)
            
            if not child_is_dominated:
                # Child is non-dominated. Replace a point from the dominated set if possible.
                dominated_idx = np.where(is_dominated)[0]
                if len(dominated_idx) > 0:
                    # Replace a random dominated point
                    replace_idx = dominated_idx[np.random.randint(len(dominated_idx))]
                else:
                    # All points are non-dominated. Replace the one with the largest sum of objectives (worst in a sense)
                    # Or replace the one that is "least extreme". 
                    # Let's replace the one with the highest sum of normalized objectives.
                    # Normalize F to [0,1] roughly based on current min/max
                    f_min = F.min(axis=0)
                    f_max = F.max(axis=0)
                    f_range = f_max - f_min
                    f_range[f_range == 0] = 1.0
                    F_norm = (F - f_min) / f_range
                    sums = F_norm.sum(axis=1)
                    replace_idx = np.argmax(sums)
            else:
                # Child is dominated. Still evaluate it, but replace the worst point in the population.
                # This helps escape local optima if the population is stuck.
                f_min = F.min(axis=0)
                f_max = F.max(axis=0)
                f_range = f_max - f_min
                f_range[f_range == 0] = 1.0
                F_norm = (F - f_min) / f_range
                sums = F_norm.sum(axis=1)
                replace_idx = np.argmax(sums)
                
            X[replace_idx] = child
            F[replace_idx] = fc
            
        return X
# EVOLVE-BLOCK-END

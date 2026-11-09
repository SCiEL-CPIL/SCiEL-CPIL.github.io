"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Fast Multi-Objective Optimizer for WFG9.
    Combines the efficient 'Replace Worst' loop structure of Inspiration 1 
    with improved initialization (LHS) and adaptive mutation strategies.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290.0  # Safety margin before 330s limit

        rng = np.random.default_rng(0)
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Population size: Small enough for fast non-dominated checks, 
        # large enough for diversity. 30 is a good balance for 200 budget.
        pop_size = 30
        if pop_size > self.budget:
            pop_size = max(10, self.budget // 2)

        # --- Improved Initialization: Latin Hypercube Sampling (LHS) ---
        # This ensures better initial coverage of the search space than pure random.
        X = np.zeros((pop_size, self.dim))
        for d in range(self.dim):
            # Generate LHS samples in [0, 1]
            indices = rng.permutation(pop_size)
            samples = (indices + rng.random(pop_size)) / pop_size
            X[:, d] = lo[d] + span[d] * samples
        
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Helper to find non-dominated indices
        # For small pop_size, O(N^2) is fast enough
        def get_non_dominated_indices(F):
            n = len(F)
            non_dominated = []
            for i in range(n):
                is_dominated = False
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_dominated = True
                        break
                if not is_dominated:
                    non_dominated.append(i)
            return non_dominated

        # Main optimization loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Find current non-dominated points
            nd_indices = get_non_dominated_indices(F)
            
            # If all are dominated (shouldn't happen usually, but safe check)
            if not nd_indices:
                nd_indices = list(range(len(F)))

            # Select parent randomly from non-dominated set
            parent_idx = nd_indices[rng.integers(len(nd_indices))]
            parent = X[parent_idx]

            # Adaptive mutation step size
            # Start with larger steps for exploration, shrink for exploitation
            progress = used / self.budget
            # Step size decays from 0.25 to 0.05 of the span
            step_size = 0.25 - 0.20 * progress
            
            # Gaussian mutation
            child = parent + step_size * span * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1

            # --- Replacement Strategy ---
            # Simple and effective: Replace the point with the worst sum of objectives.
            # This is O(N) and very fast.
            # To add a bit of robustness, if the child is non-dominated, 
            # we are confident in replacing the worst. 
            # If the child is dominated, we still replace the worst to maintain population size,
            # effectively acting as a "survival of the fittest" on the sum metric,
            # which often correlates well with proximity to the front in low-budget scenarios.
            
            # Calculate sums for replacement
            sums = np.sum(F, axis=1)
            worst_idx = int(np.argmax(sums))
            
            # Update population
            X[worst_idx] = child
            F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END

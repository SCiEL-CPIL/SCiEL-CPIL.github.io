"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Improved evolutionary multi-objective search with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270  # 300s limit, keep 30s margin
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Population size
        pop_size = min(50, budget // 2)
        if pop_size < 10:
            pop_size = 10
            
        # Initial population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        def compute_dominance(F):
            """Compute domination matrix: dom[i,j] is True if i dominates j"""
            n = len(F)
            dom = np.zeros((n, n), dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                            dom[i, j] = True
            return dom
        
        # WFG9 has a specific structure. Let's use a more targeted approach.
        # WFG9 is known to have a convex Pareto front in the last dimensions.
        # But we don't have the formula, so we stick to evolutionary search.
        
        # Adaptive mutation rate
        base_mutation = 0.1
        min_mutation = 0.01
        max_mutation = 0.5
        
        while used < budget:
            if time.time() > deadline:
                break
                
            # Find non-dominated set
            dom = compute_dominance(F)
            is_dominated = np.any(dom, axis=0)
            non_dominated_idx = np.where(~is_dominated)[0]
            
            if len(non_dominated_idx) == 0:
                parents = X
            else:
                # Select from non-dominated set with some diversity
                # Use crowding distance or just random selection
                parents = X[non_dominated_idx]
                
            # Select parent
            parent_idx = rng.integers(len(parents))
            parent = parents[parent_idx]
            
            # Adaptive mutation rate based on progress
            progress = used / budget
            mutation_rate = base_mutation * (1 - 0.5 * progress) + 0.05 * progress
            mutation_rate = max(min_mutation, min(max_mutation, mutation_rate))
            
            # Create child
            child = parent + mutation_rate * (hi - lo) * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Replace worst individual
            # Use a simple metric: sum of objectives, or better, use domination
            # For efficiency, just replace the one that is dominated by the most
            # or simply replace a random dominated individual if exists
            
            # Find individuals dominated by the child
            child_dominates = np.array([
                np.all(fc <= f) and np.any(fc < f) for f in F
            ])
            
            if np.any(child_dominates):
                # Replace one of the dominated individuals
                dominated_by_child = np.where(child_dominates)[0]
                replace_idx = dominated_by_child[rng.integers(len(dominated_by_child))]
            else:
                # Replace the worst individual (highest sum)
                replace_idx = np.argmax(np.sum(F, axis=1))
                
            X[replace_idx] = child
            F[replace_idx] = fc
            
            # Occasional restart to escape local optima
            if used % 50 == 0 and used < budget - 10:
                restart_idx = rng.integers(pop_size)
                X[restart_idx] = lo + (hi - lo) * rng.random(dim)
                F[restart_idx] = problem(X[restart_idx])
                used += 1
        
        return X
# EVOLVE-BLOCK-END

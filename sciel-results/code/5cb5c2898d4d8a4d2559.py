"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Simple evolutionary multi-objective search: random init, mutate non-dominated parents."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        pop = int(min(20, self.budget // 4))
        X = lo + (hi - lo) * rng.random((pop, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop
        
        # Track the best known points (non-dominated set)
        non_dom_idx = self._get_non_dominated_indices(F)
        
        while used < self.budget:
            if time.time() > deadline:
                break
                
            # Get non-dominated parents
            parents = X[non_dom_idx] if len(non_dom_idx) > 0 else X
            parent = parents[rng.integers(len(parents))]
            
            # Mutation: adaptive step size
            # Use a smaller step for better convergence, occasionally larger for exploration
            step_size = 0.1 if rng.random() > 0.1 else 0.5
            child = parent + step_size * (hi - lo) * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Update population: replace worst point
            # Use a simple metric: sum of objectives (normalized if possible, but sum is okay for now)
            # To be more robust, we can replace the point that is dominated by the new one,
            # or the point with the worst "potential" (e.g., furthest from origin in objective space, or crowded).
            # For simplicity and speed, let's stick to replacing the point that is dominated by the child if any,
            # otherwise replace the point with the highest sum of objectives.
            
            # Check if child dominates any point in population
            # This is O(N) per iteration, acceptable for N=20
            dominated_mask = np.array([
                np.all(F[i] >= fc) and np.any(F[i] > fc) for i in range(len(F))
            ])
            
            if np.any(dominated_mask):
                # Replace one of the dominated points (randomly chosen from dominated)
                dominated_indices = np.where(dominated_mask)[0]
                replace_idx = dominated_indices[rng.integers(len(dominated_indices))]
            else:
                # Replace the point with the worst sum of objectives
                # This is a crude approximation of hypervolume contribution
                replace_idx = int(np.argmax(np.sum(F, axis=1)))
                
            X[replace_idx] = child
            F[replace_idx] = fc
            
            # Update non-dominated set
            non_dom_idx = self._get_non_dominated_indices(F)
            
        return X

    def _get_non_dominated_indices(self, F):
        """Return indices of non-dominated points in F."""
        n = len(F)
        non_dominated = np.ones(n, dtype=bool)
        
        for i in range(n):
            if not non_dominated[i]:
                continue
            # Check if i is dominated by any j
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    non_dominated[i] = False
                    break
                    
        return np.where(non_dominated)[0]
# EVOLVE-BLOCK-END

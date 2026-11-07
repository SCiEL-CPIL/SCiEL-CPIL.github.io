"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective search with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
        pop_size = min(50, self.budget // 4)
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        def fast_nondominated_mask(F):
            """Return boolean mask of non-dominated points in F."""
            n = len(F)
            is_dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if is_dominated[i]:
                    continue
                # Check if any other point dominates i
                # Point j dominates i if F[j] <= F[i] and F[j] != F[i]
                # We can vectorize this check
                dominates_i = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
                dominates_i[i] = False
                if np.any(dominates_i):
                    is_dominated[i] = True
            return ~is_dominated

        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Find non-dominated parents
            nondom_mask = fast_nondominated_mask(F)
            if np.any(nondom_mask):
                parents_idx = np.where(nondom_mask)[0]
            else:
                parents_idx = np.arange(len(F))
            
            # Select a parent
            parent_idx = parents_idx[rng.integers(len(parents_idx))]
            parent = X[parent_idx]
            
            # Mutate with adaptive step size
            # Start with larger steps, decrease over time
            progress = used / self.budget
            step_size = 0.5 * (1 - progress) + 0.05
            noise = rng.standard_normal(self.dim)
            child = parent + step_size * (hi - lo) * noise
            child = np.clip(child, lo, hi)
            
            # Evaluate child
            fc = problem(child)
            used += 1
            
            # Replace worst point (simplest strategy: replace random or worst)
            # To maintain diversity, we replace a point that is dominated by the child if possible
            # Otherwise, replace the point with the largest sum (worst in terms of magnitude)
            # But a better strategy for hypervolume is to replace points that are dominated
            # or points that are far from the ideal point.
            
            # Let's use a simple replacement: replace the point that is dominated by the child.
            # If none, replace the point with the highest objective sum.
            
            # Check if child dominates any existing point
            child_dominates = np.all(F >= fc, axis=1) & np.any(F > fc, axis=1)
            if np.any(child_dominates):
                # Replace one of the dominated points (randomly)
                dominated_indices = np.where(child_dominates)[0]
                replace_idx = dominated_indices[rng.integers(len(dominated_indices))]
            else:
                # Replace the point with the largest sum
                replace_idx = int(np.argmax(np.sum(F, axis=1)))
            
            X[replace_idx] = child
            F[replace_idx] = fc
            
            # Occasional restarts to escape local optima?
            # Maybe not needed if step size is adaptive.
            
        return X
# EVOLVE-BLOCK-END

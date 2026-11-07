"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Improved multi-objective evolutionary search with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Larger initial population for better coverage
        pop_size = min(50, self.budget // 4)
        X = lo + range_ * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        def is_dominated(f, F):
            """Check if f is dominated by any point in F."""
            return np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1))
        
        def get_non_dominated_indices(F):
            """Return indices of non-dominated points in F."""
            n = len(F)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                for j in range(n):
                    if i != j and nd[j]:
                        if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                            nd[j] = False
                        elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            nd[i] = False
                            break
            return np.where(nd)[0]
        
        # Adaptive mutation step size
        base_step = 0.1
        min_step = 0.01
        max_step = 0.5
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Get non-dominated front
            nd_idx = get_non_dominated_indices(F)
            parents = X[nd_idx] if len(nd_idx) > 0 else X
            
            # Select parent
            parent_idx = rng.integers(len(parents))
            parent = parents[parent_idx]
            
            # Adaptive mutation: start large, shrink over time
            progress = used / self.budget
            step = base_step * (1.0 - progress * 0.8) + min_step * progress
            step = np.clip(step, min_step, max_step)
            
            child = parent + step * range_ * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1
            
            # Replace the worst dominated point, or a random point if none dominated
            dominated_mask = np.array([is_dominated(F[i], F) for i in range(len(F))])
            if np.any(dominated_mask):
                # Among dominated points, replace the one with highest sum of objectives
                dominated_indices = np.where(dominated_mask)[0]
                worst_in_dominated = dominated_indices[np.argmax(np.sum(F[dominated_indices], axis=1))]
                X[worst_in_dominated] = child
                F[worst_in_dominated] = fc
            else:
                # No dominated points; replace a random point from the non-dominated front
                # to maintain diversity
                replace_idx = rng.integers(len(F))
                X[replace_idx] = child
                F[replace_idx] = fc
        
        return X
# EVOLVE-BLOCK-END

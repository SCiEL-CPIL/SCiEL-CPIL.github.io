"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        deadline = time.time() + 300.0  # safety margin before 330s limit
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(0)
        
        # Initial random sampling
        pop_size = min(50, self.budget // 4)
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)
        
        def non_dominated_indices(F):
            n = len(F)
            nd = []
            for i in range(n):
                is_nd = True
                for j in range(n):
                    if i != j and dominates(F[j], F[i]):
                        is_nd = False
                        break
                if is_nd:
                    nd.append(i)
            return np.array(nd)
        
        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Get non-dominated parents
            nd_idx = non_dominated_indices(F)
            parents = X[nd_idx] if len(nd_idx) > 0 else X
            
            # Select parent with random mutation
            p_idx = rng.integers(len(parents))
            parent = parents[p_idx]
            
            # Adaptive mutation step
            step = 0.1 * (hi - lo) * rng.standard_normal(self.dim)
            child = parent + step
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Replace worst point
            # Use a simple score: sum of normalized objectives
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            X[worst_idx] = child
            F[worst_idx] = fc
            
            # Occasionally do a larger jump
            if rng.random() < 0.05:
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                big_child = lo + (hi - lo) * rng.random(self.dim)
                big_fc = problem(big_child)
                used += 1
                worst_idx = int(np.argmax(np.sum(F, axis=1)))
                X[worst_idx] = big_child
                F[worst_idx] = big_fc
        
        return X
# EVOLVE-BLOCK-END

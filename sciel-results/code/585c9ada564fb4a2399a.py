"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Simple evolutionary multi-objective search: random init, mutate non-dominated parents."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # 10s safety margin before 330s kill
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
        pop = int(min(20, self.budget // 4))
        X = lo + (hi - lo) * rng.random((pop, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop
        
        # Track all non-dominated points found so far for better hypervolume
        all_X = [X]
        all_F = [F]
        
        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Find non-dominated indices
            dominated = np.zeros(len(F), dtype=bool)
            for i in range(len(F)):
                if dominated[i]:
                    continue
                for j in range(len(F)):
                    if i == j or dominated[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            
            parents_idx = np.where(~dominated)[0]
            if len(parents_idx) == 0:
                parents_idx = np.arange(len(F))
            
            parent_idx = parents_idx[rng.integers(len(parents_idx))]
            child = X[parent_idx] + 0.1 * (hi - lo) * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1
            
            # Find worst point to replace (using sum of objectives as a simple metric)
            worst = int(np.argmax(np.sum(F, axis=1)))
            X[worst], F[worst] = child, fc
            
            # Add to archive if non-dominated
            all_X.append(child)
            all_F.append(fc)
        
        return X
# EVOLVE-BLOCK-END

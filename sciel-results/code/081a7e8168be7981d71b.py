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
        deadline = start_time + 320.0  # safety margin before 330s kill
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
        pop = int(min(20, self.budget // 4))
        X = lo + (hi - lo) * rng.random((pop, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop
        
        # Track best known points for hypervolume improvement
        # We want to keep a diverse set of non-dominated points
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Compute domination mask
            # A point i dominates j if F[i] <= F[j] and F[i] != F[j]
            # Here we find for each point if it is dominated by any other
            dominated = np.zeros(pop, dtype=bool)
            for i in range(pop):
                if dominated[i]:
                    continue
                for j in range(pop):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated[j] = True
            
            parents = X[~dominated] if np.any(~dominated) else X
            parent_idx = rng.integers(len(parents))
            parent = parents[parent_idx]
            
            # Adaptive step size: start large, shrink if stuck? 
            # For now, keep 0.1 but maybe adjust based on progress?
            # Let's try a slightly larger step initially or random step size
            step_size = 0.1
            child = parent + step_size * (hi - lo) * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1
            
            # Replace worst point (highest sum of objectives)
            worst = int(np.argmax(np.sum(F, axis=1)))
            X[worst], F[worst] = child, fc
            
            # Optional: if child is non-dominated, we might want to keep it more carefully
            # But the current replacement strategy is simple.
            
        return X
# EVOLVE-BLOCK-END

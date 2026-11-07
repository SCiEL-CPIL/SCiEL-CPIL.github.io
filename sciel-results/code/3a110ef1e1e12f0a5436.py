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
        deadline = start_time + 295  # 5 second safety margin before 300s limit
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
        
        # Use a larger initial population to better sample the space
        pop_size = int(min(50, self.budget // 4))
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        while used < self.budget:
            # Check time limit
            if time.time() > deadline:
                break
            
            # Find non-dominated solutions
            dominated = np.array([np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1)) for f in F])
            parents = X[~dominated] if np.any(~dominated) else X
            
            # Adaptive mutation: larger steps early, smaller later
            progress = used / self.budget
            mutation_scale = 0.3 * (1.0 - progress) + 0.05
            
            # Select parent and create child
            parent_idx = rng.integers(len(parents))
            child = parents[parent_idx] + mutation_scale * (hi - lo) * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Replace the worst solution in the population
            worst = int(np.argmax(np.sum(F, axis=1)))
            X[worst], F[worst] = child, fc
        
        return X
# EVOLVE-BLOCK-END

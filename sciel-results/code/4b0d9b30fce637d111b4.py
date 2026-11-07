"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer with time guard and adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # 300s wall-clock limit
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Initialize population
        pop_size = min(20, self.budget // 4)
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        def compute_dominance(F):
            """Return boolean array: True if point i is dominated by any other point."""
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            return dominated
        
        # Main evolution loop
        while used < self.budget:
            # Check time limit with safety margin
            if time.time() > deadline - 2.0:
                break
            
            dominated = compute_dominance(F)
            parents = X[~dominated] if np.any(~dominated) else X
            
            # Adaptive mutation: start with larger steps, reduce over time
            progress = used / self.budget
            step_size = 0.2 * (1 - 0.5 * progress)  # 0.2 -> 0.1
            
            child = parents[rng.integers(len(parents))] + step_size * (hi - lo) * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1
            
            # Replace worst point (highest sum of objectives)
            worst = int(np.argmax(np.sum(F, axis=1)))
            X[worst], F[worst] = child, fc
        
        return X
# EVOLVE-BLOCK-END

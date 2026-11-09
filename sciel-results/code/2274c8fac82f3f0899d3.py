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
        deadline = start_time + 320.0  # Safety margin before 330s limit
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
        pop = int(min(20, self.budget // 4))
        X = lo + (hi - lo) * rng.random((pop, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop
        
        # Track best known hypervolume-contributing points (Pareto front approximation)
        # We'll just keep the population and let the evaluator record all evaluations.
        # The key is to use the budget effectively.
        
        while used < self.budget:
            # Check time limit
            if time.time() > deadline - 5.0:  # 5s safety margin
                break
                
            # Find non-dominated points
            dominated = np.zeros(len(F), dtype=bool)
            for i in range(len(F)):
                for j in range(len(F)):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            
            parents = X[~dominated] if np.any(~dominated) else X
            n_parents = len(parents)
            
            # Adaptive mutation step: start larger, shrink over time
            progress = used / self.budget
            step_size = 0.15 * (1.0 - 0.5 * progress)
            
            # Try multiple children per iteration if time permits
            n_children = 1
            if progress < 0.5:
                n_children = 2
            elif progress < 0.8:
                n_children = 1
            
            for _ in range(n_children):
                if used >= self.budget or time.time() > deadline - 5.0:
                    break
                    
                parent = parents[rng.integers(n_parents)]
                # Gaussian mutation with adaptive step
                child = parent + step_size * (hi - lo) * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1
                
                # Replace worst point based on sum of objectives (simple heuristic)
                worst = int(np.argmax(np.sum(F, axis=1)))
                X[worst], F[worst] = child, fc
        
        return X
# EVOLVE-BLOCK-END

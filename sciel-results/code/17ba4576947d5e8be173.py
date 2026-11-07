"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Improved evolutionary multi-objective search with deadline guard and better selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin before 330s limit
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
        pop_size = min(40, self.budget // 5)
        
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Track all evaluated points for final hypervolume calculation
        all_X = [x.copy() for x in X]
        all_F = [f.copy() for f in F]
        
        while used < self.budget:
            # Check deadline with safety margin
            if time.time() > deadline - 1.0:
                break
            
            # Find non-dominated set
            dominated = np.zeros(len(F), dtype=bool)
            for i in range(len(F)):
                for j in range(len(F)):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            
            non_dom_idx = np.where(~dominated)[0]
            
            if len(non_dom_idx) == 0:
                parents = X
            else:
                # Select from non-dominated set, prefer diverse points
                parents = X[non_dom_idx]
            
            # Generate child via mutation
            parent = parents[rng.integers(len(parents))]
            
            # Adaptive mutation step: larger early, smaller later
            progress = used / self.budget
            step_size = 0.15 * (1.0 - progress) + 0.02
            child = parent + step_size * (hi - lo) * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Replace worst point in population
            worst = int(np.argmax(np.sum(F, axis=1)))
            X[worst], F[worst] = child, fc
            
            # Record for final result
            all_X.append(child.copy())
            all_F.append(fc.copy())
        
        return X
# EVOLVE-BLOCK-END

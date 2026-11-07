import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # Population size: 20 is a good balance for budget 200
        # Allows for 10 generations of full population updates if we were doing that,
        # but here we do 1 evaluation per step.
        pop_size = 20
        
        # Initialize population
        # Using random initialization as per the top scorer, but ensuring we cover the space
        rng = np.random.default_rng(42) # Fixed seed for reproducibility in local tests, but evaluator seeds np.random
        # The evaluator seeds np.random.seed(seed) before calling Optimizer.
        # So we should use np.random directly or a rng seeded from np.random.
        # However, the top scorer used np.random.default_rng(0). 
        # If the evaluator seeds np.random, default_rng(0) is deterministic regardless of evaluator seed.
        # The task says: "For seeds 0 and 1 the evaluator calls np.random.seed(seed)".
        # This implies we should probably respect the global seed or use a derived seed.
        # But the top scorer (1.0000) used default_rng(0). This suggests the evaluator's seed might not matter 
        # if the program is deterministic, OR the evaluator's seed is meant to initialize the program's randomness.
        # Let's use np.random directly to be safe and aligned with "np.random.seed(seed)".
        
        X = lo + span * np.random.rand(pop_size, self.dim)
        F = np.array([problem(x) for x in X])
        used = pop_size

        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Identify non-dominated points in the current population
            # Vectorized dominance check
            # F is (pop_size, n_obj)
            # For each i, check if any j dominates i
            # j dominates i if F[j] <= F[i] and F[j] != F[i]
            
            is_dominated = np.zeros(pop_size, dtype=bool)
            for i in range(pop_size):
                # Check if any other point dominates i
                # F <= F[i] means all objectives of F are <= objectives of F[i]
                # F < F[i] means at least one objective is strictly less
                dominates_i = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
                dominates_i[i] = False  # Exclude self
                if np.any(dominates_i):
                    is_dominated[i] = True
            
            # Select parents from non-dominated set if available, else from all
            if not np.all(is_dominated):
                parents_idx = np.where(~is_dominated)[0]
            else:
                parents_idx = np.arange(pop_size)
            
            # Pick a random parent
            parent_idx = parents_idx[np.random.randint(len(parents_idx))]
            parent = X[parent_idx]
            
            # Adaptive step size
            # Start with larger steps to explore, reduce to exploit
            progress = used / self.budget
            step_size = 0.5 * (1.0 - progress) + 0.05
            
            # Gaussian mutation
            noise = np.random.randn(self.dim)
            child = parent + step_size * span * noise
            child = np.clip(child, lo, hi)
            
            # Evaluate child
            fc = problem(child)
            used += 1
            
            # Replacement Strategy:
            # 1. If child dominates any point in population, replace a random dominated point.
            # 2. Else, replace the point with the largest sum of objectives (worst).
            
            # Check if child dominates any existing point
            # F >= fc and F != fc
            child_dominates_mask = np.all(F >= fc, axis=1) & np.any(F > fc, axis=1)
            
            if np.any(child_dominates_mask):
                # Replace a random point that is dominated by the child
                dominated_indices = np.where(child_dominates_mask)[0]
                replace_idx = dominated_indices[np.random.randint(len(dominated_indices))]
            else:
                # Replace the worst point (highest sum)
                replace_idx = np.argmax(np.sum(F, axis=1))
            
            X[replace_idx] = child
            F[replace_idx] = fc

        return X

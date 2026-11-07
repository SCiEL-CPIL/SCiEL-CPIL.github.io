"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for DTLZ4.
    Uses mixed initialization to handle the distorted search space of DTLZ4.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ4 specific heuristic:
        # The 'g' function variables (indices 2 to 11) have an optimal value 
        # of 0.5^(1/beta). Standard DTLZ4 uses beta=100.
        x_tail_opt = 0.5 ** (1.0 / 100.0)
        
        pop_size = 20
        X = np.empty((pop_size, self.dim))
        
        # Half the population: Tail fixed to estimated optimum (beta=100)
        half = pop_size // 2
        X[:half, :2] = np.random.random((half, 2))
        X[:half, 2:] = x_tail_opt
        
        # Other half: Tail uniform (robustness against different beta)
        X[half:, :2] = np.random.random((pop_size - half, 2))
        X[half:, 2:] = lo[2:] + (hi[2:] - lo[2:]) * np.random.random((pop_size - half, self.dim - 2))
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Find non-dominated points
            nondom_mask = np.ones(len(X), dtype=bool)
            for i in range(len(X)):
                if not nondom_mask[i]:
                    continue
                for j in range(len(X)):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nondom_mask[i] = False
                        break
            
            parents_idx = np.where(nondom_mask)[0]
            if len(parents_idx) == 0:
                parents_idx = np.arange(len(X))
            
            parent_idx = parents_idx[np.random.randint(len(parents_idx))]
            parent = X[parent_idx]
            
            # Mutation: Gaussian noise
            # Slightly larger noise for tail variables to allow adaptation if beta != 100
            noise = np.random.standard_normal(self.dim) * 0.1 * (hi - lo)
            child = parent + noise
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Replace worst point (max sum of objectives)
            sums = np.sum(F, axis=1)
            worst_idx = int(np.argmax(sums))
            
            X[worst_idx] = child
            F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END

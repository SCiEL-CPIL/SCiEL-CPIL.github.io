# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(0)
        
        # Population size: 20 is a good balance between diversity and speed
        pop_size = 20
        
        # --- Initialization ---
        # Simple random initialization
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size
        
        if time.time() > deadline:
            return X
        
        # --- Evolutionary Loop ---
        # Use adaptive step size
        step_scale = 0.1
        shrink_rate = 0.999
        
        while used < self.budget and time.time() < deadline:
            # Find non-dominated points
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            
            for i in range(n):
                if dominated[i]:
                    continue
                le = np.all(F <= F[i], axis=1)
                lt = np.any(F < F[i], axis=1)
                le[i] = False
                if np.any(le & lt):
                    dominated[i] = True
            
            # Select parent from non-dominated set
            nd_idx = np.where(~dominated)[0]
            if len(nd_idx) > 0:
                parent_idx = nd_idx[rng.integers(len(nd_idx))]
            else:
                parent_idx = rng.integers(n)
            
            parent = X[parent_idx].copy()
            
            # Mutate with Gaussian noise
            mutation = step_scale * (hi - lo) * rng.standard_normal(self.dim)
            child = parent + mutation
            child = np.clip(child, lo, hi)
            
            fc = np.array(problem(child), dtype=float)
            used += 1
            
            # Replace the worst point by sum of objectives
            sums = np.sum(F, axis=1)
            worst_idx = int(np.argmax(sums))
            
            X[worst_idx] = child
            F[worst_idx] = fc
            
            # Shrink step size
            step_scale *= shrink_rate
            
            # Polish phase
            if used > self.budget * 0.9:
                step_scale = min(step_scale, 0.005)
                if used > self.budget * 0.95:
                    step_scale = min(step_scale, 0.001)

        return X
# EVOLVE-BLOCK-END

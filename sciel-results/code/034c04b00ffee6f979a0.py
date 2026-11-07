"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using hypervolume-guided search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref = np.array([2.5] * n_obj)
        
        # Track all evaluated points
        all_F = []
        
        # DTLZ4-aware initialization: first 7 vars biased to boundaries
        pop_size = min(25, budget // 4)
        
        def generate_individual():
            x = np.zeros(dim)
            for i in range(min(7, dim)):
                if rng.random() < 0.5:
                    x[i] = rng.random() * 0.1
                else:
                    x[i] = 1.0 - rng.random() * 0.1
            for i in range(7, dim):
                x[i] = lo[i] + (hi[i] - lo[i]) * rng.random()
            return np.clip(x, lo, hi)
        
        X = np.array([generate_individual() for _ in range(pop_size)])
        F = np.array([problem(x) for x in X])
        all_F.extend(F.tolist())
        used = pop_size
        
        while used < budget:
            if time.time() > deadline:
                break
            
            # Find non-dominated set
            n = len(F)
            nd_mask = np.zeros(n, dtype=bool)
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
                if not dominated:
                    nd_mask[i] = True
            
            nd_indices = np.where(nd_mask)[0]
            if len(nd_indices) == 0:
                nd_indices = np.array([np.argmin(np.sum(F, axis=1))])
            
            # Tournament selection from non-dominated
            if len(nd_indices) >= 2:
                i1 = nd_indices[rng.integers(len(nd_indices))]
                i2 = nd_indices[rng.integers(len(nd_indices))]
                parent_idx = i1 if np.sum(F[i1]) <= np.sum(F[i2]) else i2
            else:
                parent_idx = nd_indices[0]
            
            # Adaptive mutation
            progress = used / budget
            sigma = 0.25 * (1.0 - 0.5 * progress)
            
            child = X[parent_idx].copy()
            noise = rng.standard_normal(dim) * sigma * (hi - lo)
            child = child + noise
            child = np.clip(child, lo, hi)
            
            # Boundary snap for DTLZ4 deceptive variables
            if rng.random() < 0.3:
                for i in range(min(7, dim)):
                    child[i] = 0.0 if rng.random() < 0.5 else 1.0
            
            fc = problem(child)
            all_F.append(fc)
            used += 1
            
            # Replace worst point
            dominance_count = np.zeros(n)
            for i in range(n):
                for j in range(n):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominance_count[i] += 1
            
            dominated_mask = dominance_count > 0
            if np.any(dominated_mask):
                dom_idx = np.where(dominated_mask)[0]
                worst_idx = dom_idx[np.argmax(dominance_count[dom_idx])]
            else:
                worst_idx = np.argmax(np.sum(F, axis=1))
            
            X[worst_idx] = child
            F[worst_idx] = fc
        
        return X
# EVOLVE-BLOCK-END

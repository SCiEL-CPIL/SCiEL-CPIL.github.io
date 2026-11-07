# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE34 using adaptive mutation with non-dominated sorting."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _non_dominated_mask(self, F):
        """Return a boolean mask of non-dominated points in F (n x n_obj array)."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=bool)
        
        # F[i] dominates F[j] if F[i] <= F[j] in all objectives and F[i] < F[j] in at least one
        # Compute dominance matrix
        # diff[i,j,k] = F[i,k] - F[j,k]
        diff = F[:, np.newaxis, :] - F[np.newaxis, :, :]  # shape (n, n, n_obj)
        
        # leq[i,j] = all(diff[i,j,:] <= 0)
        leq = np.all(diff <= 0, axis=2)  # shape (n, n)
        
        # lt[i,j] = any(diff[i,j,:] < 0)
        lt = np.any(diff < 0, axis=2)  # shape (n, n)
        
        # dominates[i,j] = leq[i,j] & lt[i,j]
        dominates = leq & lt  # shape (n, n)
        
        # A point is non-dominated if no other point dominates it
        # dominated[i] = any(dominates[j,i] for j != i)
        # Since dominates[i,i] is False (lt[i,i] is False), we can just check any over axis 0
        dominated = np.any(dominates, axis=0)  # shape (n,)
        
        return ~dominated

    def _find_worst_index(self, F):
        """Find the index of the point to replace based on crowding and quality."""
        n = len(F)
        if n == 0:
            return 0
        
        # Use sum of normalized objectives as a simple quality metric
        F_min = F.min(axis=0)
        F_max = F.max(axis=0)
        F_range = F_max - F_min + 1e-10
        F_norm = (F - F_min) / F_range
        quality = np.sum(F_norm, axis=1)
        
        return np.argmax(quality)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_obj = self.n_obj
        dim = self.dim
        
        # Initial population
        pop_size = min(30, self.budget // 3)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Track best hypervolume contributors
        all_F = F.copy()
        
        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Find non-dominated points using vectorized mask
            nd_mask = self._non_dominated_mask(F)
            nd_idx = np.where(nd_mask)[0]
            if len(nd_idx) == 0:
                nd_idx = np.array([np.argmin(np.sum(F, axis=1))])
            
            # Adaptive mutation scale: start larger, shrink over time
            progress = used / self.budget
            mut_scale = 0.3 * (1 - progress) + 0.05
            
            # Generate children from non-dominated parents
            n_children = min(10, self.budget - used)
            for _ in range(n_children):
                if time.time() > deadline:
                    break
                if used >= self.budget:
                    break
                    
                parent_idx = nd_idx[rng.integers(len(nd_idx))]
                parent = X[parent_idx]
                
                # Gaussian mutation
                child = parent + mut_scale * (hi - lo) * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)
                
                fc = problem(child)
                used += 1
                
                # Replace worst point in population
                worst_idx = self._find_worst_index(F)
                
                X[worst_idx] = child
                F[worst_idx] = fc
                
                # Update all_F for hypervolume tracking
                all_F = np.vstack([all_F, fc])
        
        return X
# EVOLVE-BLOCK-END

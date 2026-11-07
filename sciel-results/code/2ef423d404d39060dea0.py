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
        
        # F[i] dominates F[j] if F[i] <= F[j] in all objectives and < in at least one
        # For each point i, check if any other point j dominates it
        dominated = np.zeros(n, dtype=bool)
        
        for i in range(n):
            if dominated[i]:
                continue
            # Check if any j dominates i
            # F[j] <= F[i] for all objectives, and F[j] < F[i] for at least one
            leq = np.all(F <= F[i], axis=1)
            lt = np.any(F < F[i], axis=1)
            # Exclude i itself
            leq[i] = False
            if np.any(leq & lt):
                dominated[i] = True
        
        return ~dominated

    def _find_worst_index(self, F, ref_point):
        """Find the index of the worst point based on hypervolume contribution (approx)."""
        # Simple heuristic: point with largest sum of (F[i] - ref_point) is worst
        # But we want to replace points that are dominated or far from the front
        # Use a simple quality metric: sum of normalized objectives
        if len(F) == 0:
            return 0
        
        # Normalize and sum
        F_min = F.min(axis=0)
        F_max = F.max(axis=0)
        F_range = F_max - F_min
        F_range[F_range == 0] = 1.0
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
        budget = self.budget
        
        # Reference point for RE34
        ref_point = np.array([1705.0, 12.0, 0.3])
        
        # Initial population: mix of random and boundary points
        pop_size = min(40, budget // 3)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        
        # Add some boundary points for diversity
        n_boundary = min(10, pop_size // 4)
        for i in range(n_boundary):
            # Randomly choose which dimensions to set to boundary
            n_dims_to_boundary = rng.integers(1, dim + 1)
            dims_to_boundary = rng.choice(dim, size=n_dims_to_boundary, replace=False)
            for d in dims_to_boundary:
                if rng.random() < 0.5:
                    X[i, d] = lo[d]
                else:
                    X[i, d] = hi[d]
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Track all evaluated points for hypervolume
        all_F = F.copy()
        
        # Adaptive parameters
        base_mut_scale = 0.4
        decay_rate = 0.995
        
        while used < budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Find non-dominated points
            nd_mask = self._non_dominated_mask(F)
            nd_idx = np.where(nd_mask)[0]
            
            if len(nd_idx) == 0:
                # Fallback: use best single objective points
                nd_idx = np.array([np.argmin(F[:, 0]), np.argmin(F[:, 1]), np.argmin(F[:, 2])])
                nd_idx = np.unique(nd_idx)
            
            # Adaptive mutation scale
            progress = used / budget
            mut_scale = base_mut_scale * (decay_rate ** used)
            mut_scale = max(mut_scale, 0.05)
            
            # Generate children
            n_children = min(15, budget - used)
            for _ in range(n_children):
                if time.time() > deadline:
                    break
                if used >= budget:
                    break
                
                # Choose parent from non-dominated set with some exploration
                if rng.random() < 0.7:
                    parent_idx = nd_idx[rng.integers(len(nd_idx))]
                else:
                    parent_idx = rng.integers(len(F))
                
                parent = X[parent_idx].copy()
                
                # Gaussian mutation with occasional larger jumps
                if rng.random() < 0.1:
                    jump_scale = mut_scale * 3.0
                else:
                    jump_scale = mut_scale
                
                child = parent + jump_scale * (hi - lo) * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)
                
                fc = problem(child)
                used += 1
                
                # Update all_F
                all_F = np.vstack([all_F, fc])
                
                # Decide whether to replace a point in the population
                # Replace if child is non-dominated or better than worst
                worst_idx = self._find_worst_index(F, ref_point)
                
                # Check if child is better than worst in at least one objective
                # and not worse in all
                child_better = np.any(fc < F[worst_idx])
                child_not_worse_all = not np.all(fc > F[worst_idx])
                
                if child_better and child_not_worse_all:
                    X[worst_idx] = child
                    F[worst_idx] = fc
        
        return X
# EVOLVE-BLOCK-END

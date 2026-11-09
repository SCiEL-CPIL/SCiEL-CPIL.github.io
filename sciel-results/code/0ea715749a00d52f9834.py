"""Multi-objective optimizer for ZDT4 using population-based search with hypervolume-guided selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based multi-objective optimizer for ZDT4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hypervolume_2d(self, points, ref):
        """Compute hypervolume for 2D points given reference point."""
        if len(points) == 0:
            return 0.0
        # Sort by first objective
        points = sorted(points, key=lambda p: p[0])
        hv = 0.0
        for i, p in enumerate(points):
            if p[0] >= ref[0] or p[1] >= ref[1]:
                continue
            # Find the minimum f2 among all points with f1 <= p[0]
            # Actually, for 2D HV, we need: for each point, the area it contributes
            # Sort by f1 ascending, then for each point, the width is (next_f1 - this_f1) or (ref_f1 - this_f1)
            # and height is (ref_f2 - this_f2)
            pass
        
        # Better approach: sort by f1, then sweep
        valid = [p for p in points if p[0] < ref[0] and p[1] < ref[1]]
        if not valid:
            return 0.0
        valid.sort(key=lambda p: p[0])
        
        hv = 0.0
        prev_f1 = valid[0][0]
        min_f2 = valid[0][1]
        for i in range(1, len(valid)):
            cur_f1, cur_f2 = valid[i]
            # The region from prev_f1 to cur_f1 with height (ref_f2 - min_f2)
            if min_f2 < ref[1]:
                hv += (cur_f1 - prev_f1) * (ref[1] - min_f2)
            min_f2 = min(min_f2, cur_f2)
            prev_f1 = cur_f1
        # Last segment
        if min_f2 < ref[1]:
            hv += (ref[0] - prev_f1) * (ref[1] - min_f2)
        
        return hv

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([11.0, 400.0])
        
        # Initial population: spread x1 across range, random other variables
        pop_size = min(20, self.budget // 5)
        X = np.zeros((pop_size, self.dim))
        # Spread x1 uniformly
        X[:, 0] = np.linspace(lo[0], hi[0], pop_size)
        # Random other variables
        for d in range(1, self.dim):
            X[:, d] = lo[d] + (hi[d] - lo[d]) * np.random.rand(pop_size)
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Track best hypervolume
        best_hv = self._hypervolume_2d(F.tolist(), ref)
        
        # Mutation strategies
        step_sizes = [0.05, 0.1, 0.2, 0.5, 1.0]
        
        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Select parent: prefer non-dominated, then diverse
            # Simple approach: random selection with bias toward better points
            # Compute dominance
            dominated = np.zeros(pop_size, dtype=bool)
            for i in range(pop_size):
                for j in range(pop_size):
                    if i != j:
                        if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            dominated[i] = True
                            break
            
            non_dom = ~dominated
            if np.any(non_dom):
                # Weighted selection from non-dominated
                weights = np.ones(pop_size) * 0.1
                weights[non_dom] = 1.0
                weights /= weights.sum()
                parent_idx = np.random.choice(pop_size, p=weights)
            else:
                parent_idx = np.random.randint(pop_size)
            
            parent = X[parent_idx].copy()
            
            # Choose mutation strategy
            step = step_sizes[np.random.randint(len(step_sizes))]
            
            # Mutate: with some probability, reset some variables to random
            child = parent.copy()
            if np.random.rand() < 0.3:
                # Reset 1-3 random variables
                n_reset = np.random.randint(1, 4)
                reset_idx = np.random.choice(self.dim, n_reset, replace=False)
                for d in reset_idx:
                    child[d] = lo[d] + (hi[d] - lo[d]) * np.random.rand()
            
            # Add Gaussian noise
            child += step * (hi - lo) * np.random.randn(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Replace worst point (highest sum of objectives, or random if tied)
            sums = np.sum(F, axis=1)
            worst = np.argmax(sums)
            X[worst] = child
            F[worst] = fc
            
            # Update best hypervolume
            hv = self._hypervolume_2d(F.tolist(), ref)
            if hv > best_hv:
                best_hv = hv
        
        return X
# EVOLVE-BLOCK-END

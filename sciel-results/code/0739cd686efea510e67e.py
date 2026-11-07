"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Simple evolutionary multi-objective search: random init, mutate non-dominated parents."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 320.0  # Safety margin before 330s kill
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        pop_size = int(min(20, self.budget // 4))
        
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        while used < self.budget:
            # Time guard
            if time.time() > deadline:
                break
            
            # Vectorized dominance check
            # F[i] dominates F[j] if F[i] <= F[j] for all obj and F[i] < F[j] for at least one
            # dominated[i] = True if there exists j != i such that F[j] dominates F[i]
            # F[j] <= F[i] for all obj: np.all(F <= F[i], axis=1)
            # F[j] < F[i] for at least one obj: np.any(F < F[i], axis=1)
            # We need to exclude j=i. Since F[i] <= F[i] and F[i] < F[i] is false, 
            # the condition F[i] < F[i] is false, so j=i will not mark i as dominated by itself.
            # Wait, np.any(F < F[i], axis=1) for j=i is np.any(F[i] < F[i], axis=0) which is False.
            # So j=i never dominates i.
            
            dominated = np.zeros(len(F), dtype=bool)
            for i in range(len(F)):
                # Check if any other point dominates F[i]
                # F[j] <= F[i] and F[j] < F[i] (strictly in at least one)
                # Note: we can vectorize this slightly better but for N=20 it's fast enough
                # To speed up:
                # less_or_equal = np.all(F <= F[i], axis=1)
                # strictly_less = np.any(F < F[i], axis=1)
                # dominated[i] = np.any(less_or_equal & strictly_less)
                
                # Actually, let's just do it in a vectorized way for the whole population if possible?
                # No, it's pairwise.
                # For N=20, 20*20*3 operations is trivial.
                
                # Optimization: 
                # F <= F[i] -> shape (N,)
                # F < F[i] -> shape (N,)
                # We want to know if there is ANY j such that (F[j] <= F[i]) AND (F[j] < F[i])
                # This is equivalent to:
                # np.any( (np.all(F <= F[i], axis=1)) & (np.any(F < F[i], axis=1)) )
                
                # Let's try to compute this for all i at once?
                # F_le = F[:, np.newaxis, :] <= F[np.newaxis, :, :] # (N, N, 3)
                # F_lt = F[:, np.newaxis, :] < F[np.newaxis, :, :]  # (N, N, 3)
                # all_le = np.all(F_le, axis=2) # (N, N) - F[i] <= F[j]
                # any_lt = np.any(F_lt, axis=2) # (N, N) - F[i] < F[j] in some dim
                # # We want dominated[i] = any_j (F[j] <= F[i] and F[j] < F[i])
                # # This is transpose of the above.
                # # F[j] <= F[i] is all_le[j, i]
                # # F[j] < F[i] is any_lt[j, i]
                # # So dominated[i] = any_j (all_le[j, i] and any_lt[j, i])
                # # = any_j (all_le.T[i, j] and any_lt.T[i, j])
                
                # This is much faster for larger N, but N=20 is small.
                # However, let's use the vectorized version for robustness/speed if N grows.
                # But N is fixed at 20.
                
                # Let's stick to the loop for clarity and low overhead, or just use the vectorized block.
                # The vectorized block allocates (20, 20, 3) which is tiny.
                pass
            
            # Let's just use the vectorized block for the whole population at once.
            F_le = F[:, np.newaxis, :] <= F[np.newaxis, :, :]  # (N, N, 3)
            F_lt = F[:, np.newaxis, :] < F[np.newaxis, :, :]   # (N, N, 3)
            all_le = np.all(F_le, axis=2)  # (N, N)
            any_lt = np.any(F_lt, axis=2)  # (N, N)
            
            # all_le[i, j] is True if F[i] <= F[j]
            # any_lt[i, j] is True if F[i] < F[j] in any dim
            # We want dominated[j] = any_i (F[i] <= F[j] and F[i] < F[j])
            # dominated[j] = any_i (all_le[i, j] and any_lt[i, j])
            # dominated = np.any(all_le.T & any_lt.T, axis=1)
            
            dominated = np.any(all_le.T & any_lt.T, axis=1)
            
            parents_idx = np.where(~dominated)[0]
            if len(parents_idx) == 0:
                parents_idx = np.arange(len(F))
            
            parent_idx = parents_idx[rng.integers(len(parents_idx))]
            parent = X[parent_idx]
            
            # Adaptive mutation step size
            # Start with 0.1, decrease over time
            # Or use a simple decay
            step_size = 0.1 * (0.9 ** (used // 10))
            if step_size < 0.01:
                step_size = 0.01
                
            mutation = step_size * (hi - lo) * rng.standard_normal(self.dim)
            child = parent + mutation
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Replace worst point
            # Use sum of objectives as a simple fitness proxy
            # Or use a more robust metric?
            # For now, sum is okay.
            # Better: use the point with the largest distance to the ideal point?
            # Or just replace the dominated point with the highest rank?
            # Since we have the dominated mask, we can replace a dominated point if available.
            # Otherwise, replace the worst non-dominated point.
            
            # Let's replace the point with the largest "sum" among dominated ones if any, else among all.
            # Actually, the original code replaced the global worst (max sum).
            # Let's keep that but maybe improve the metric.
            
            # Sum of objectives is not scale invariant.
            # Let's normalize? No, too expensive per step.
            # Let's just use the sum.
            
            worst_idx = np.argmax(np.sum(F, axis=1))
            X[worst_idx] = child
            F[worst_idx] = fc
            
        return X
# EVOLVE-BLOCK-END

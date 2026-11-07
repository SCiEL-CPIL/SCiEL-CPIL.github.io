import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290 # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # 1. Initialization: LHS
        n_init = 20
        # Simple LHS implementation
        indices = np.array([np.random.permutation(n_init) for _ in range(self.dim)])
        X_init = np.zeros((n_init, self.dim))
        for d in range(self.dim):
            X_init[:, d] = (indices[d] + 0.5) / n_init
        X_init = lo + (hi - lo) * X_init
        
        # Evaluate
        F_init = np.array([problem(x) for x in X_init])
        used = n_init
        
        # Store all evaluated points in an archive
        archive_X = X_init.copy()
        archive_F = F_init.copy()
        
        # 2. Refinement Loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Find non-dominated points in current archive
            # Simple O(N^2) check
            nd_mask = np.zeros(len(archive_F), dtype=bool)
            for i in range(len(archive_F)):
                for j in range(len(archive_F)):
                    if i == j: continue
                    if np.all(archive_F[j] <= archive_F[i]) and np.any(archive_F[j] < archive_F[i]):
                        nd_mask[i] = False
                        break
                if nd_mask[i]:
                    nd_mask[i] = True # Default, only set false if dominated
            
            # Actually, simpler:
            nd_indices = []
            for i in range(len(archive_F)):
                dominated = False
                for j in range(len(archive_F)):
                    if i == j: continue
                    if np.all(archive_F[j] <= archive_F[i]) and np.any(archive_F[j] < archive_F[i]):
                        dominated = True
                        break
                if not dominated:
                    nd_indices.append(i)
            
            if not nd_indices:
                # Fallback to best point
                nd_indices = [np.argmin(np.sum(archive_F, axis=1))]
            
            # Pick a non-dominated point to refine
            idx = nd_indices[np.random.randint(len(nd_indices))]
            x = archive_X[idx].copy()
            f = archive_F[idx].copy()
            
            # Local Search: Coordinate Descent style
            improved = False
            for d in range(self.dim):
                if used >= self.budget or time.time() > deadline:
                    break
                # Try small step
                step = 0.05 * (hi[d] - lo[d])
                for direction in [-1, 1]:
                    if used >= self.budget or time.time() > deadline:
                        break
                    x_new = x.copy()
                    x_new[d] += direction * step
                    x_new = np.clip(x_new, lo, hi)
                    
                    f_new = problem(x_new)
                    used += 1
                    
                    # Check if x_new is better than x
                    # If x_new dominates x, it's better.
                    # If x is dominated by x_new, replace.
                    # If they are non-dominated, we might want to keep the one with lower sum or distance.
                    # For simplicity, if x_new dominates x, replace.
                    if np.all(f_new <= f) and np.any(f_new < f):
                        x = x_new
                        f = f_new
                        improved = True
                    # Optional: If non-dominated, check if it's closer to ideal
                    # But sticking to dominance is safer for Pareto front tracking.
            
            # Update archive
            # Replace the old point with the new one if it's improved
            if improved:
                archive_X[idx] = x
                archive_F[idx] = f
            else:
                # Maybe try a larger step or different point next time
                pass
                
        return archive_X

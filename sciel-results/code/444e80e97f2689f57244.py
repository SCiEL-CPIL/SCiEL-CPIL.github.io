import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Archive storage
        X = []
        F = []
        evals = 0
        
        def is_dominated(f_new, F_set):
            """Check if f_new is dominated by any in F_set."""
            for f in F_set:
                if np.all(f <= f_new) and np.any(f < f_new):
                    return True
            return False

        def dominates(f1, f2):
            """Check if f1 dominates f2."""
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def prune_archive(X_list, F_list, max_size):
            """Prune the archive to max_size using crowding distance."""
            if len(X_list) <= max_size:
                return X_list, F_list
            
            F_arr = np.array(F_list)
            # Normalize objectives for crowding distance calculation
            # Handle case where min == max (single point or no variation)
            mins = np.min(F_arr, axis=0)
            maxs = np.max(F_arr, axis=0)
            ranges = maxs - mins
            ranges[ranges == 0] = 1.0 # Avoid division by zero
            F_norm = (F_arr - mins) / ranges
            
            n = len(F_list)
            m = self.n_obj
            crowd = np.zeros(n)
            
            for i in range(m):
                # Sort by objective i
                idx = np.argsort(F_norm[:, i])
                crowd[idx[0]] = np.inf
                crowd[idx[-1]] = np.inf
                for k in range(1, n - 1):
                    crowd[idx[k]] += (F_norm[idx[k+1], i] - F_norm[idx[k-1], i])
            
            # Remove points with lowest crowding distance
            # We want to keep the most diverse ones (highest crowd)
            # We need to remove (len - max_size) points
            to_remove = len(X_list) - max_size
            # Get indices sorted by crowd distance (ascending)
            remove_indices = np.argsort(crowd)[:to_remove]
            
            # Build new lists excluding removed indices
            keep_mask = np.ones(len(X_list), dtype=bool)
            keep_mask[remove_indices] = False
            
            X_new = [X_list[i] for i in range(len(X_list)) if keep_mask[i]]
            F_new = [F_list[i] for i in range(len(X_list)) if keep_mask[i]]
            
            return X_new, F_new

        # 1. Initialization
        init_pop = 20
        for _ in range(init_pop):
            x = lo + np.random.rand(self.dim) * range_
            f = problem(x)
            evals += 1
            if not is_dominated(f, F):
                # Check if f dominates existing ones
                new_X = []
                new_F = []
                for xi, fi in zip(X, F):
                    if not dominates(f, fi):
                        new_X.append(xi)
                        new_F.append(fi)
                X = new_X
                F = new_F
                X.append(x)
                F.append(f)
        
        if evals >= self.budget:
            return np.array(X) if X else np.array([lo])

        # 2. Evolution
        max_archive = 50
        initial_step = 0.5 # Relative to range
        
        while evals < self.budget:
            if time.time() > deadline:
                break
            
            # Select parent
            # Random selection from current non-dominated set
            parent_idx = np.random.randint(len(X))
            parent = X[parent_idx]
            
            # Mutation
            progress = evals / self.budget
            # Step size decreases as we progress
            step = initial_step * (1.0 - 0.9 * progress)
            # Add some randomness to step size itself
            step *= np.random.uniform(0.5, 1.5)
            
            child = parent + step * range_ * np.random.randn(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            evals += 1
            
            # Update Archive
            if not is_dominated(fc, F):
                new_X = []
                new_F = []
                for xi, fi in zip(X, F):
                    if not dominates(fc, fi):
                        new_X.append(xi)
                        new_F.append(fi)
                X = new_X
                F = new_F
                X.append(child)
                F.append(fc)
                
                # Prune if necessary
                if len(X) > max_archive:
                    X, F = prune_archive(X, F, max_archive)

        return np.array(X) if X else np.array([lo])

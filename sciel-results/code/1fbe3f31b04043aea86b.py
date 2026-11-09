import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # 1. Latin Hypercube Sampling for initial spread
        n_init = 50
        X_init = np.zeros((n_init, self.dim))
        for j in range(self.dim):
            # Permute indices to create stratified samples
            perm = np.random.permutation(n_init)
            X_init[:, j] = (perm + np.random.rand(n_init)) / n_init
        
        # Scale to bounds
        X_init = lo + X_init * span
        
        # Evaluate initial population
        F_init = np.array([problem(x) for x in X_init])
        used = n_init
        
        # Maintain Non-Dominated Set
        nds = []
        nds_f = []
        
        # Helper to check if a point is dominated by any in NDS
        def is_dominated(f_new, nds_f_list):
            for f in nds_f_list:
                if np.all(f <= f_new) and np.any(f < f_new):
                    return True
            return False
            
        # Helper to add to NDS and remove dominated
        def add_to_nds(x_new, f_new, nds_list, nds_f_list):
            # Check if new point is dominated by existing
            if is_dominated(f_new, nds_f_list):
                return False
            
            # Check if new point dominates existing, remove them
            keep_indices = []
            for i, f_old in enumerate(nds_f_list):
                if np.all(f_new <= f_old) and np.any(f_new < f_old):
                    continue # f_new dominates f_old
                keep_indices.append(i)
                
            # Rebuild lists
            new_nds = [x_new]
            new_nds_f = [f_new]
            
            for i in keep_indices:
                new_nds.append(nds_list[i])
                new_nds_f.append(nds_f_list[i])
                
            # Update external lists
            nds_list[:] = new_nds
            nds_f_list[:] = new_nds_f
            return True

        # Populate initial NDS
        for i in range(n_init):
            add_to_nds(X_init[i], F_init[i], nds, nds_f)
            
        # 2. Local Refinement Loop
        total_remaining = self.budget - used
        
        for step in range(total_remaining):
            if used >= self.budget:
                break
                
            # Select a parent from NDS
            if not nds:
                parent = lo + np.random.rand(self.dim) * span
            else:
                idx = np.random.randint(len(nds))
                parent = nds[idx]
                
            # Determine step size (Linear decay)
            progress = step / max(1, total_remaining - 1)
            sigma = 0.2 * (1.0 - progress) + 0.01 * progress
            
            # Generate perturbation
            noise = np.random.normal(0, sigma, self.dim)
            child = parent + noise
            
            # Clip to bounds
            child = np.clip(child, lo, hi)
            
            # Evaluate
            f_child = problem(child)
            used += 1
            
            # Try to add to NDS
            add_to_nds(child, f_child, nds, nds_f)
            
        return nds

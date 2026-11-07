import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Archive of non-dominated points
        archive_X = []
        archive_F = []
        
        evals = 0
        
        # Helper to add to archive
        def add_to_archive(x, f):
            nonlocal archive_X, archive_F
            # Check if x is dominated by any in archive
            for i, f_arch in enumerate(archive_F):
                if np.all(f_arch <= f) and np.any(f_arch < f):
                    return False # x is dominated
            # Check if x dominates any in archive
            new_archive_X = []
            new_archive_F = []
            for j, f_arch in enumerate(archive_F):
                if not (np.all(f <= f_arch) and np.any(f < f_arch)):
                    new_archive_X.append(archive_X[j])
                    new_archive_F.append(f_arch)
            new_archive_X.append(x)
            new_archive_F.append(f)
            archive_X = new_archive_X
            archive_F = new_archive_F
            return True

        # 1. Initialization
        pop_size = 10
        X_init = lo + (hi - lo) * np.random.random((pop_size, self.dim))
        for x in X_init:
            f = problem(x)
            evals += 1
            add_to_archive(x, f)
            
        # 2. Main Loop
        stagnation_counter = 0
        while evals < self.budget:
            # Select parent
            if not archive_X:
                # Should not happen, but safety
                x_parent = lo + (hi - lo) * np.random.random(self.dim)
            else:
                idx = np.random.randint(len(archive_X))
                x_parent = archive_X[idx]
                
            # Mutate
            # Adaptive sigma? Start with 0.1 * range
            sigma = 0.1 * (hi - lo)
            x_child = x_parent + sigma * np.random.standard_normal(self.dim)
            x_child = np.clip(x_child, lo, hi)
            
            f_child = problem(x_child)
            evals += 1
            
            if add_to_archive(x_child, f_child):
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                
            # If stuck, inject random point
            if stagnation_counter > 20:
                x_rand = lo + (hi - lo) * np.random.random(self.dim)
                f_rand = problem(x_rand)
                evals += 1
                add_to_archive(x_rand, f_rand)
                stagnation_counter = 0
                if evals >= self.budget: break

        return archive_X

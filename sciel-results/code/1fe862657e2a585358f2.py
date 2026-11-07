import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
    
    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        pop_size = 20
        n_gen = (self.budget - pop_size) // pop_size
        n_gen = max(n_gen, 1)
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Archive of non-dominated solutions
        archive_points = []
        archive_fvals = []
        
        def is_non_dominated(f_new, archive_fvals):
            """Check if f_new is non-dominated by any point in archive."""
            if len(archive_fvals) == 0:
                return True
            for f_a in archive_fvals:
                if np.all(f_a <= f_new) and np.any(f_a < f_new):
                    return False
            return True
        
        def update_archive(f_new, x_new):
            """Add f_new to archive if non-dominated, and remove dominated points."""
            if not is_non_dominated(f_new, archive_fvals):
                return
            
            # Remove points dominated by f_new
            new_archive_points = []
            new_archive_fvals = []
            for i in range(len(archive_fvals)):
                f_a = archive_fvals[i]
                # Check if f_new dominates f_a
                if np.all(f_new <= f_a) and np.any(f_new < f_a):
                    continue # f_a is dominated, remove it
                new_archive_points.append(archive_points[i])
                new_archive_fvals.append(f_a)
            
            new_archive_points.append(x_new)
            new_archive_fvals.append(f_new)
            
            archive_points[:] = new_archive_points
            archive_fvals[:] = new_archive_fvals

        # Initialize archive
        for i in range(pop_size):
            update_archive(F[i], X[i])

        # DE Parameters
        F_diff = 0.8  # Scale factor
        CR = 0.9      # Crossover rate

        for gen in range(n_gen):
            next_X = []
            next_F = []
            
            for i in range(pop_size):
                # Pick 3 distinct random indices
                while True:
                    idxs = rng.integers(0, pop_size, 3)
                    if len(set(idxs.tolist())) == 3 and i not in idxs:
                        break
                
                i1, i2, i3 = idxs.tolist()
                
                # DE/rand/1 mutation
                v = X[i1] + F_diff * (X[i2] - X[i3])
                
                # Crossover
                j_rand = rng.integers(0, self.dim)
                for j in range(self.dim):
                    if rng.random() < CR or j == j_rand:
                        pass # Use v
                    else:
                        v[j] = X[i][j]
                
                # Clip to bounds
                v = np.clip(v, lo, hi)
                
                # Evaluate
                f_v = problem(v)
                used += 1
                
                # Selection: If v dominates X[i], replace
                if np.all(f_v <= F[i]) and np.any(f_v < F[i]):
                    next_X.append(v)
                    next_F.append(f_v)
                    update_archive(f_v, v)
                else:
                    next_X.append(X[i])
                    next_F.append(F[i])
            
            X = np.array(next_X)
            F = np.array(next_F)
            
            # Optional: Inject diversity from archive if population stagnates
            if len(archive_points) > pop_size:
                # Find worst point in population (simple heuristic: sum of objectives)
                worst_idx = np.argmax(np.sum(F, axis=1))
                # Pick a random point from archive
                arch_idx = rng.integers(0, len(archive_points))
                # Replace to ensure exploration
                X[worst_idx] = archive_points[arch_idx]
                F[worst_idx] = archive_fvals[arch_idx]

        return X

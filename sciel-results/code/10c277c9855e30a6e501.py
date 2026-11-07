import numpy as np
import time

class Optimizer:
    """
    Optimizer for ZDT6 using parametric initialization and local search.
    Exploits the structure of ZDT6 where concentrating 'mass' in one variable
    minimizes the secondary objective g, leading to a better Pareto front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        
        # Archive to store non-dominated points
        # Each entry: (x_vector, f_vector)
        archive_x = []
        archive_f = []
        
        used = 0
        
        def is_dominated(f_new, archive_f):
            """Check if f_new is dominated by any point in archive_f."""
            if len(archive_f) == 0:
                return False
            # f_new is dominated if there exists f_old such that f_old <= f_new and f_old != f_new
            # Vectorized check:
            # less_equal = np.all(archive_f <= f_new, axis=1)
            # less = np.any(archive_f < f_new, axis=1)
            # return np.any(less_equal & less)
            
            # For small archives, loop is fine and clear
            for f_old in archive_f:
                if np.all(f_old <= f_new) and np.any(f_old < f_new):
                    return True
            return False

        def add_to_archive(x_new, f_new):
            """Add point to archive if non-dominated, and remove dominated points."""
            # Check if new point is dominated by existing
            if is_dominated(f_new, archive_f):
                return
            
            # Remove existing points dominated by new point
            new_archive_x = []
            new_archive_f = []
            for i, f_old in enumerate(archive_f):
                # If old is dominated by new, skip it
                if np.all(f_new <= f_old) and np.any(f_new < f_old):
                    continue
                new_archive_x.append(archive_x[i])
                new_archive_f.append(f_old)
            
            # Add new point
            new_archive_x.append(x_new)
            new_archive_f.append(f_new)
            
            archive_x[:] = new_archive_x
            archive_f[:] = new_archive_f

        # Phase 1: Parametric Initialization
        # ZDT6 structure: f1 and g depend on x_0...x_{dim-2}. x_{dim-1} is often irrelevant or weak.
        # To minimize g (and thus f2) for a given f1, we concentrate the sum S in x_0.
        # S = sum(x_0...x_{dim-2}). f1 = 1 + 10 * S / (dim-1).
        # We sample S using power law to bias towards low f1 (high hypervolume contribution).
        
        n_init = 100
        n_init = min(n_init, self.budget // 2)
        
        # Generate u in [0, 1]
        u = np.random.random(n_init)
        # Power law: t = u^1.5 biases towards 0
        # Map t to S in [0, dim-1]
        S = (dim - 1) * (u ** 1.5)
        
        for i in range(n_init):
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            # Concentrate mass in x[0]
            x[0] = S[i]
            # Set x[1...dim-2] to 0
            # Set x[dim-1] to 0.5 (middle) as it's likely irrelevant
            x[dim-1] = 0.5
            
            # Clip to bounds
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            used += 1
            
            add_to_archive(x, f)

        # Phase 2: Local Search / Refinement
        # Use remaining budget to perturb points in the archive.
        # We prioritize points with low f1 (high hypervolume contribution).
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            if not archive_x:
                # Fallback to random
                x = lo + (hi - lo) * np.random.random(dim)
                f = problem(x)
                used += 1
                add_to_archive(x, f)
                continue
            
            # Select a parent from the archive
            # Bias selection towards points with lower f1
            # Score for selection: 1 / (f1 + epsilon)
            f1_vals = np.array([f[0] for f in archive_f])
            weights = 1.0 / (f1_vals + 1e-6)
            weights /= np.sum(weights)
            
            idx = np.random.choice(len(archive_x), p=weights)
            parent_x = archive_x[idx]
            
            # Mutation: Gaussian noise
            # Scale noise based on distance to bounds or just fixed sigma
            sigma = 0.1
            noise = sigma * np.random.standard_normal(dim)
            child_x = parent_x + noise
            child_x = np.clip(child_x, lo, hi)
            
            f_child = problem(child_x)
            used += 1
            
            add_to_archive(child_x, f_child)

        return archive_x

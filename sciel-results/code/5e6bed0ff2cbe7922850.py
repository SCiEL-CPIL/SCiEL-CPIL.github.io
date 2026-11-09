import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Phase 1: Latin Hypercube Sampling
        N_init = 100
        if self.budget < 2 * N_init:
            N_init = self.budget // 2
            
        # Generate LHS points
        X_lhs = np.zeros((N_init, dim))
        for d in range(dim):
            # Divide range into N_init bins
            bin_width = (hi[d] - lo[d]) / N_init
            # Random permutation of bin indices
            perm = np.random.permutation(N_init)
            # Center of bins
            centers = lo[d] + (perm + 0.5) * bin_width
            X_lhs[:, d] = centers
            
        # Evaluate initial population
        F_lhs = np.array([problem(x) for x in X_lhs])
        
        # Maintain non-dominated archive
        # Archive stores indices into a global list of evaluated points?
        # Or just store the points and their fitness.
        # Let's store (x, f) pairs in a list `archive`.
        archive = []
        for i in range(N_init):
            x = X_lhs[i]
            f = F_lhs[i]
            # Check if f is dominated by any in archive
            dominated = False
            for _, f_arch in archive:
                if np.all(f_arch <= f) and np.any(f_arch < f):
                    dominated = True
                    break
            if not dominated:
                # Remove points in archive dominated by f
                new_archive = []
                for x_arch, f_arch in archive:
                    if not (np.all(f <= f_arch) and np.any(f < f_arch)):
                        new_archive.append((x_arch, f_arch))
                new_archive.append((x, f))
                archive = new_archive
        
        # Phase 2: Local Search
        N_search = self.budget - N_init
        sigma = 0.1 # Relative to range?
        
        for i in range(N_search):
            if time.time() > deadline:
                break
                
            if not archive:
                # Fallback to random
                x = lo + (hi - lo) * np.random.random(dim)
            else:
                # Pick random parent from archive
                parent_idx = np.random.randint(len(archive))
                x_parent = archive[parent_idx][0]
                
                # Gaussian perturbation
                # Scale noise by the range of each dimension
                noise = np.random.normal(0, 1, dim)
                x = x_parent + noise * (hi - lo) * sigma
                
                # Clip to bounds
                x = np.clip(x, lo, hi)
            
            f = problem(x)
            
            # Update archive
            dominated = False
            for _, f_arch in archive:
                if np.all(f_arch <= f) and np.any(f_arch < f):
                    dominated = True
                    break
            if not dominated:
                new_archive = []
                for x_arch, f_arch in archive:
                    if not (np.all(f <= f_arch) and np.any(f < f_arch)):
                        new_archive.append((x_arch, f_arch))
                new_archive.append((x, f))
                archive = new_archive
        
        return archive # Return value is ignored, but good practice

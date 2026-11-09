"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer combining archive-based search with LHS initialization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin before 330s limit
        
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Archive of non-dominated points: list of (x, f)
        archive = []
        
        def is_dominated(f_new, archive):
            for _, f_old in archive:
                if np.all(f_old <= f_new) and np.any(f_old < f_new):
                    return True
            return False
            
        def update_archive(x_new, f_new):
            # Check if new point is dominated
            if is_dominated(f_new, archive):
                return archive
            
            # Remove points dominated by the new point
            new_archive = []
            for x_old, f_old in archive:
                if np.all(f_new <= f_old) and np.any(f_new < f_old):
                    continue
                new_archive.append((x_old, f_old))
                
            new_archive.append((x_new, f_new))
            return new_archive

        # 1. Initialization: Latin Hypercube Sampling
        n_init = max(10, int(budget * 0.1))
        
        # Generate LHS points
        # For each dimension, divide [0,1] into n_init intervals, pick random point in each
        perm = np.argsort(np.argsort(np.random.rand(n_init, dim)), axis=0)
        u = (perm + np.random.rand(n_init, dim)) / n_init
        X_init = lo + (hi - lo) * u
        
        # Evaluate and populate archive
        used = 0
        for i in range(n_init):
            if time.time() > deadline:
                break
            x = X_init[i]
            f = problem(x)
            used += 1
            # Basic validity check
            if np.any(np.isnan(f)) or np.any(np.isinf(f)):
                continue
            archive = update_archive(x, f)
            
        # 2. Iterative Refinement
        # Strategy: Pick a point from archive, perturb, evaluate.
        # Use adaptive step size and random selection for exploration.
        
        while used < budget:
            if time.time() > deadline:
                break
            
            if not archive:
                # If archive is empty, try random
                x = lo + (hi - lo) * np.random.rand(dim)
                f = problem(x)
                used += 1
                if np.any(np.isnan(f)) or np.any(np.isinf(f)):
                    continue
                archive = update_archive(x, f)
                continue
            
            # Select a parent from the archive
            # Random selection promotes exploration
            idx = np.random.randint(0, len(archive))
            x_parent, f_parent = archive[idx]
            
            # Perturbation
            # Adaptive step size: smaller as we approach the budget end
            progress = used / budget
            step_size = 0.2 * (1.0 - 0.5 * progress)
            
            # Gaussian noise
            noise = np.random.randn(dim) * step_size
            x_child = x_parent + noise * (hi - lo)
            x_child = np.clip(x_child, lo, hi)
            
            f_child = problem(x_child)
            used += 1
            
            if np.any(np.isnan(f_child)) or np.any(np.isinf(f_child)):
                continue
                
            archive = update_archive(x_child, f_child)
            
        return archive
# EVOLVE-BLOCK-END

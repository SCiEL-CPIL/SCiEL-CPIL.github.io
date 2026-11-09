"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """
    Multi-objective optimizer using:
    1. Deterministic Sobol quasi-random sequence for uniform initial coverage.
    2. Local Gaussian perturbations around non-dominated points for refinement.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Phase 1: Sobol quasi-random initialization
        # Use half the budget for global exploration to ensure good coverage
        n_init = self.budget // 2
        if n_init < 1:
            n_init = 1
            
        sobol = qmc.Sobol(d=self.dim, scramble=False)
        points_unit = sobol.random(n_init)
        X_init = lo + points_unit * span
        F_init = np.array([problem(x) for x in X_init])
        used = n_init

        # Build initial non-dominated archive
        # Archive stores (x, f) pairs
        archive_x = []
        archive_f = []
        
        for x, f in zip(X_init, F_init):
            # Check if f is dominated by any in archive
            dominated = False
            for af in archive_f:
                if np.all(af <= f) and np.any(af < f):
                    dominated = True
                    break
            if not dominated:
                # Remove points in archive dominated by f
                new_archive_x = []
                new_archive_f = []
                for ax, af in zip(archive_x, archive_f):
                    if not (np.all(f <= af) and np.any(f < af)):
                        new_archive_x.append(ax)
                        new_archive_f.append(af)
                new_archive_x.append(x)
                new_archive_f.append(f)
                archive_x = new_archive_x
                archive_f = new_archive_f

        # Phase 2: Local refinement
        remaining = self.budget - used
        rng = np.random.default_rng(42)
        
        if remaining > 0 and len(archive_x) > 0:
            # Adaptive step size: start moderate, maybe decay
            # For [0,1]^4, 0.1 is a good start
            step_size = 0.1
            
            for i in range(remaining):
                # Select a parent from the archive
                # To avoid clustering, we can pick randomly. 
                # Alternatively, pick the one with highest "hypervolume contribution" but that's expensive.
                idx = rng.integers(len(archive_x))
                parent = archive_x[idx]
                
                # Gaussian perturbation
                perturbation = step_size * span * rng.standard_normal(self.dim)
                child = parent + perturbation
                child = np.clip(child, lo, hi)
                
                fc = problem(child)
                
                # Update archive
                dominated = False
                for af in archive_f:
                    if np.all(af <= fc) and np.any(af < fc):
                        dominated = True
                        break
                
                if not dominated:
                    # Remove dominated points from archive
                    new_archive_x = []
                    new_archive_f = []
                    for ax, af in zip(archive_x, archive_f):
                        if not (np.all(fc <= af) and np.any(fc < af)):
                            new_archive_x.append(ax)
                            new_archive_f.append(af)
                    
                    # Add new point
                    new_archive_x.append(child)
                    new_archive_f.append(fc)
                    
                    archive_x = new_archive_x
                    archive_f = new_archive_f
                    
                    # Optional: Decay step size slightly to refine
                    # step_size *= 0.995

        return X_init # Return value is ignored, but we must return something
# EVOLVE-BLOCK-END

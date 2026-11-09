"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc
from scipy.spatial.distance import cdist


class Optimizer:
    """
    Multi-objective optimizer using:
    1. Deterministic Sobol quasi-random sequence for uniform initial coverage.
    2. Local Gaussian perturbations around non-dominated points selected via maximin diversity.
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
        # Allocate 60% of the budget to initialization for a robust starting front
        n_init = int(self.budget * 0.6)
        if n_init < 1:
            n_init = 1
            
        sobol = qmc.Sobol(d=self.dim, scramble=False)
        points_unit = sobol.random(n_init)
        X_init = lo + points_unit * span
        F_init = np.array([problem(x) for x in X_init])
        used = n_init

        # Build initial non-dominated archive
        archive_x = []
        archive_f = []
        
        for x, f in zip(X_init, F_init):
            dominated = False
            for af in archive_f:
                if np.all(af <= f) and np.any(af < f):
                    dominated = True
                    break
            if not dominated:
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
        
        if remaining > 0 and len(archive_x) > 0:
            # Smaller step size for refinement
            step_size = 0.05
            
            for i in range(remaining):
                if len(archive_x) == 1:
                    idx = 0
                else:
                    # Select parent with maximin distance in objective space
                    # Compute pairwise distances
                    dists = cdist(archive_f, archive_f, metric='euclidean')
                    np.fill_diagonal(dists, np.inf)
                    min_dists = np.min(dists, axis=1)
                    idx = np.argmax(min_dists)
                
                parent = archive_x[idx]
                
                # Gaussian perturbation
                # Use global np.random to respect evaluator seed
                perturbation = step_size * span * np.random.standard_normal(self.dim)
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
                    new_archive_x = []
                    new_archive_f = []
                    for ax, af in zip(archive_x, archive_f):
                        if not (np.all(fc <= af) and np.any(fc < af)):
                            new_archive_x.append(ax)
                            new_archive_f.append(af)
                    
                    new_archive_x.append(child)
                    new_archive_f.append(fc)
                    
                    archive_x = new_archive_x
                    archive_f = new_archive_f

        return X_init
# EVOLVE-BLOCK-END

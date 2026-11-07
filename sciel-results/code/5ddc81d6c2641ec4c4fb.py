"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution optimizer for ZDT4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n = self.dim
        budget = self.budget

        archive = []
        
        def is_dominated(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def add_to_archive(x, f):
            # Check if x is dominated by any in archive
            for ax, af in archive:
                if is_dominated(af, f):
                    return False
            
            # Remove any points in archive dominated by x
            new_archive = []
            for ax, af in archive:
                if not is_dominated(f, af):
                    new_archive.append((ax, af))
            
            new_archive.append((x, f))
            
            # Prune archive if too large to maintain diversity and speed
            # Use a simple crowding distance heuristic
            if len(new_archive) > 40:
                # Calculate approximate crowding distance
                # For 2 objectives, we can sort by each objective and sum distances to neighbors
                # To keep it simple, we'll just keep the points that are most spread out
                # by sorting by f1 and keeping a subset?
                
                # Better: Keep the 40 points with the best "spread"
                # We can approximate this by keeping the points that are 
                # furthest from the centroid? No.
                
                # Let's use a simple grid-based pruning.
                # Divide the objective space into bins.
                f_vals = np.array([p[1] for p in new_archive])
                f_min = f_vals.min(axis=0)
                f_max = f_vals.max(axis=0)
                
                # If range is 0, just keep first 40
                if np.all(f_max == f_min):
                    new_archive = new_archive[:40]
                else:
                    # Normalize
                    f_norm = (f_vals - f_min) / (f_max - f_min)
                    
                    # Use a 5x5 grid
                    grid = {}
                    for i, p in enumerate(new_archive):
                        idx0 = int(f_norm[i][0] * 5)
                        idx1 = int(f_norm[i][1] * 5)
                        if idx0 > 4: idx0 = 4
                        if idx1 > 4: idx1 = 4
                        key = (idx0, idx1)
                        if key not in grid:
                            grid[key] = []
                        grid[key].append(p)
                    
                    # Keep best point in each grid cell (by sum of objectives)
                    pruned = []
                    for key, points in grid.items():
                        points.sort(key=lambda x: np.sum(x[1]))
                        pruned.append(points[0])
                    
                    # If we still have more than 40, keep the best 40 by sum
                    if len(pruned) > 40:
                        pruned.sort(key=lambda x: np.sum(x[1]))
                        pruned = pruned[:40]
                    
                    new_archive = pruned

            archive[:] = new_archive
            return True

        evaluations = 0
        
        # Phase 1: Initialize a diverse population
        # ZDT4 structure: x1 is the "shape" variable, x2..xn should be near 0.
        n_init = 20
        for _ in range(n_init):
            x = np.empty(n)
            # x1 uniform
            x[0] = np.random.uniform(lo[0], hi[0])
            # x2..xn Gaussian centered at 0, std 1.0 (wider to escape local minima)
            for i in range(1, n):
                x[i] = np.random.normal(0, 1.0)
                x[i] = np.clip(x[i], lo[i], hi[i])
            
            f = problem(x)
            evaluations += 1
            add_to_archive(x, f)
            
        # Phase 2: Differential Evolution
        F = 0.8
        CR = 0.9
        
        while evaluations < budget:
            if len(archive) < 3:
                # If archive is too small, add random points
                x = np.empty(n)
                x[0] = np.random.uniform(lo[0], hi[0])
                for i in range(1, n):
                    x[i] = np.random.normal(0, 1.0)
                    x[i] = np.clip(x[i], lo[i], hi[i])
                f = problem(x)
                evaluations += 1
                add_to_archive(x, f)
                continue

            # Select 3 distinct parents
            indices = np.random.choice(len(archive), 3, replace=False)
            x_r1 = archive[indices[0]][0]
            x_r2 = archive[indices[1]][0]
            x_r3 = archive[indices[2]][0]
            
            # Create mutant vector
            v = x_r1 + F * (x_r2 - x_r3)
            
            # Crossover: binomial crossover with probability CR
            j_rand = np.random.randint(0, n)
            trial = np.zeros(n)
            for i in range(n):
                if np.random.rand() < CR or i == j_rand:
                    trial[i] = v[i]
                else:
                    trial[i] = x_r1[i]
            
            trial = np.clip(trial, lo, hi)
            
            f_trial = problem(trial)
            evaluations += 1
            
            add_to_archive(trial, f_trial)

        return None

# EVOLVE-BLOCK-END

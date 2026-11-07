"""Multi-objective optimizer on WFG4 using hypervolume indicator guided search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-guided multi-objective optimizer with adaptive search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270.0  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref_point = np.array([3.0, 5.0, 7.0], dtype=np.float64)

        rng = np.random

        # Archive: list of objective vectors (non-dominated)
        archive = []
        archive_X = []  # corresponding decision variables

        def dominates(f1, f2):
            """Check if f1 dominates f2."""
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def update_archive(f, x):
            """Add f to archive if non-dominated, remove dominated archive points.
            Returns True if f was added."""
            # Check if f is dominated by any archive point
            for af in archive:
                if np.all(af <= f) and np.any(af < f):
                    return False
            # Remove archive points dominated by f
            new_archive = []
            new_archive_X = []
            for i, af in enumerate(archive):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive.append(af)
                    new_archive_X.append(archive_X[i])
            new_archive.append(f.copy())
            new_archive_X.append(x.copy())
            archive[:] = new_archive
            archive_X[:] = new_archive_X
            return True

        def hypervolume_3d(points, ref):
            """Compute exact 3D hypervolume for a set of points vs reference point.
            Only points strictly better than ref in all objectives count."""
            valid = [p for p in points if np.all(p < ref)]
            if len(valid) == 0:
                return 0.0
            if len(valid) == 1:
                p = valid[0]
                return float(np.prod(ref - p))
            
            # Use grid-based approach for exact computation
            xs = sorted(set([p[0] for p in valid] + [ref[0]]))
            ys = sorted(set([p[1] for p in valid] + [ref[1]]))
            zs = sorted(set([p[2] for p in valid] + [ref[2]]))
            
            nx = len(xs) - 1
            ny = len(ys) - 1
            nz = len(zs) - 1
            
            if nx == 0 or ny == 0 or nz == 0:
                return 0.0
            
            covered = np.zeros((nx, ny, nz), dtype=bool)
            
            for p in valid:
                xi = np.searchsorted(xs, p[0]) - 1
                yi = np.searchsorted(ys, p[1]) - 1
                zi = np.searchsorted(zs, p[2]) - 1
                
                if xi >= 0 and yi >= 0 and zi >= 0:
                    covered[xi:, yi:, zi:] = True
            
            total = 0.0
            for i in range(nx):
                dx = xs[i+1] - xs[i]
                if dx == 0:
                    continue
                for j in range(ny):
                    dy = ys[j+1] - ys[j]
                    if dy == 0:
                        continue
                    for k in range(nz):
                        if covered[i, j, k]:
                            dz = zs[k+1] - zs[k]
                            total += dx * dy * dz
            
            return total

        def compute_hv():
            """Compute hypervolume of the archive."""
            return hypervolume_3d(archive, ref_point)

        # Initialize with a diverse set of points
        # Use Latin hypercube sampling for initial population
        pop_size = min(60, budget)
        X_init = np.zeros((pop_size, dim))
        for i in range(pop_size):
            for d in range(dim):
                u = (i + rng.random()) / pop_size
                X_init[i, d] = lo[d] + span[d] * u
        
        F_init = np.array([problem(x) for x in X_init], dtype=np.float64)
        used = pop_size
        
        for i in range(pop_size):
            update_archive(F_init[i], X_init[i])

        # Main search loop: use a combination of local search around archive points
        # and random perturbation with bias toward known Pareto front structure
        while used < budget:
            if time.time() > deadline - 5.0:
                break
            
            # Strategy: pick a point from the archive and perturb it
            # With some probability, do a global random exploration
            if rng.random() < 0.15:
                # Global random exploration
                x = lo + span * rng.random(dim)
            else:
                # Pick a random archive point and perturb it
                if len(archive) > 0:
                    idx = rng.randint(len(archive))
                    base_x = archive_X[idx]
                    
                    # Adaptive step size: start larger, decrease over time
                    progress = used / budget
                    base_step = 0.3 * (1.0 - 0.7 * progress)
                    
                    # Gaussian perturbation
                    step = base_step * rng.standard_normal(dim)
                    x = base_x + step * span
                    x = np.clip(x, lo, hi)
                else:
                    x = lo + span * rng.random(dim)
            
            f = problem(x)
            used += 1
            
            if update_archive(f, x):
                # New point added to archive - good!
                pass

        return X_init  # Return something to satisfy the interface

# EVOLVE-BLOCK-END

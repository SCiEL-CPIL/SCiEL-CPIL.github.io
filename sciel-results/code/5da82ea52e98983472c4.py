"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using hypervolume indicator with efficient archive management."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hv_contribution(self, point, archive_points, ref):
        """Compute hypervolume contribution of a single point given archive and reference."""
        if len(archive_points) == 0:
            # Full hypervolume of the point
            if all(point[i] < ref[i] for i in range(self.n_obj)):
                return np.prod(ref - point)
            return 0.0
        
        # Check if point is dominated by any archive point
        for ap in archive_points:
            if all(ap[i] <= point[i] for i in range(self.n_obj)) and any(ap[i] < point[i] for i in range(self.n_obj)):
                return 0.0
        
        # Compute contribution: volume dominated by point but not by any archive point
        # For 3D, use inclusion-exclusion
        # This is a simplified version - for exactness we need more careful computation
        # For efficiency, approximate by checking against each archive point
        
        # Filter archive points that are relevant (not dominated by point)
        relevant = []
        for ap in archive_points:
            if not (all(point[i] <= ap[i] for i in range(self.n_obj)) and any(point[i] < ap[i] for i in range(self.n_obj))):
                relevant.append(ap)
        
        if len(relevant) == 0:
            if all(point[i] < ref[i] for i in range(self.n_obj)):
                return np.prod(ref - point)
            return 0.0
        
        # For 3D: contribution = volume of box from point to ref, minus overlaps with relevant archive points
        # Simplified: just return the volume of the point's box if it's non-dominated
        if all(point[i] < ref[i] for i in range(self.n_obj)):
            return np.prod(ref - point)
        return 0.0

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget
        ref = np.array([1705.0, 12.0, 0.3])
        
        rng = np.random.default_rng(42)

        # Initialize with a diverse set
        n_init = 30
        X = lo + (hi - lo) * rng.random((n_init, dim))
        F = np.array([problem(x) for x in X])
        used = n_init

        # Build initial archive
        archive = []
        for i in range(n_init):
            # Check if F[i] is dominated by any point in archive
            dominated = False
            for a in archive:
                if np.all(F[a] <= F[i]) and np.any(F[a] < F[i]):
                    dominated = True
                    break
            if dominated:
                continue
            # Remove archive points dominated by F[i]
            new_archive = []
            for a in archive:
                if not (np.all(F[i] <= F[a]) and np.any(F[i] < F[a])):
                    new_archive.append(a)
            new_archive.append(i)
            archive = new_archive

        # Main loop: use archive-based selection with adaptive mutation
        while used < budget:
            if time.time() > deadline:
                break

            # Select parent from archive using weighted random selection
            # Weight by inverse of objective values (lower is better)
            if len(archive) > 0:
                # Use tournament selection: pick 2 random, choose the one with lower sum of normalized objectives
                if len(archive) >= 2:
                    i1 = archive[rng.integers(len(archive))]
                    i2 = archive[rng.integers(len(archive))]
                    # Normalize and compare
                    s1 = np.sum(F[i1] / (hi - lo))
                    s2 = np.sum(F[i2] / (hi - lo))
                    parent_idx = i1 if s1 <= s2 else i2
                else:
                    parent_idx = archive[0]
            else:
                parent_idx = rng.integers(n_init)

            # Adaptive mutation: step size decreases over time
            progress = used / budget
            step_scale = 0.2 * (1.0 - 0.5 * progress)
            
            child = X[parent_idx].copy()
            # Gaussian mutation with adaptive step size
            child += step_scale * (hi - lo) * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Check if child is non-dominated
            dominated = False
            for a in archive:
                if np.all(F[a] <= fc) and np.any(F[a] < fc):
                    dominated = True
                    break
            
            if not dominated:
                # Remove archive points dominated by child
                new_archive = []
                for a in archive:
                    if not (np.all(fc <= F[a]) and np.any(fc < F[a])):
                        new_archive.append(a)
                new_archive.append(len(X))
                archive = new_archive
                
                # Add child to population
                X = np.vstack([X, child])
                F = np.vstack([F, fc])
            else:
                # Child is dominated; still add to population but replace a random point
                rep_idx = rng.integers(len(X))
                X[rep_idx] = child
                F[rep_idx] = fc

            # Periodic local search around best archive points
            if used % 15 == 0 and time.time() < deadline - 20 and len(archive) > 0:
                # Pick top 3 archive points by hypervolume contribution (approx)
                top_k = min(3, len(archive))
                # Sort archive by sum of objectives (lower is better)
                sorted_archive = sorted(archive, key=lambda i: np.sum(F[i]))
                for _ in range(top_k):
                    if used >= budget or time.time() > deadline:
                        break
                    local_parent = sorted_archive[_]
                    for _ls in range(2):
                        if used >= budget or time.time() > deadline:
                            break
                        lchild = X[local_parent].copy()
                        lchild += 0.05 * (hi - lo) * rng.standard_normal(dim)
                        lchild = np.clip(lchild, lo, hi)
                        lfc = problem(lchild)
                        used += 1
                        
                        # Check if local child is non-dominated
                        ldominated = False
                        for a in archive:
                            if np.all(F[a] <= lfc) and np.any(F[a] < lfc):
                                ldominated = True
                                break
                        
                        if not ldominated:
                            new_archive = []
                            for a in archive:
                                if not (np.all(lfc <= F[a]) and np.any(lfc < F[a])):
                                    new_archive.append(a)
                            new_archive.append(len(X))
                            archive = new_archive
                            X = np.vstack([X, lchild])
                            F = np.vstack([F, lfc])
                        else:
                            rep_idx = rng.integers(len(X))
                            X[rep_idx] = lchild
                            F[rep_idx] = lfc

        return X
# EVOLVE-BLOCK-END

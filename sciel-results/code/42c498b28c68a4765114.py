"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using an explicit Pareto front archive for DTLZ2."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Archive of non-dominated points
        archive_F = []  # list of objective vectors
        archive_X = []  # list of decision vectors

        used = 0

        def dominates(a, b):
            """Returns True if a dominates b (a <= b in all objectives and a < b in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def is_dominated(f, archive_F):
            """Returns True if f is dominated by any point in archive_F."""
            for af in archive_F:
                if dominates(af, f):
                    return True
            return False

        def update_archive(x, f):
            """Add x, f to archive if non-dominated; remove dominated points."""
            nonlocal archive_F, archive_X
            # Check if f is dominated by existing archive
            for i, af in enumerate(archive_F):
                if dominates(af, f):
                    return  # f is dominated, don't add
            # Remove points dominated by f
            new_archive_F = []
            new_archive_X = []
            for i, af in enumerate(archive_F):
                if not dominates(f, af):
                    new_archive_F.append(af)
                    new_archive_X.append(archive_X[i])
            new_archive_F.append(f)
            new_archive_X.append(x)
            archive_F = new_archive_F
            archive_X = new_archive_X

        # Initial random population
        n_init = min(40, budget // 5)
        for i in range(n_init):
            if used >= budget:
                break
            x = lo + (hi - lo) * rng.random(dim)
            f = np.array(problem(x))
            used += 1
            update_archive(x, f)

        # Main search loop
        while used < budget:
            if time.time() > deadline:
                break

            # Pick a random point from archive as parent
            idx = rng.integers(len(archive_X))
            parent = archive_X[idx].copy()

            # Mutation: Gaussian perturbation
            sigma = 0.1
            child = parent + sigma * (hi - lo) * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)

            fc = np.array(problem(child))
            used += 1
            update_archive(child, fc)

        return archive_X
# EVOLVE-BLOCK-END

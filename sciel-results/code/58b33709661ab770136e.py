"""Multi-objective optimizer for RE34 using LHS initialization and DE search with Pareto archive."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Latin Hypercube initialization with differential evolution and Pareto archive."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 315.0  # Safety margin before 330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        rng = np.random.default_rng(0)

        # ---------- Initialization: Latin Hypercube Sampling ----------
        # Use all budget for a structured LHS design to maximize initial coverage
        n_init = min(budget, 200)
        
        # Generate LHS points: for each dimension, divide into n_init strata
        # and sample one point from each stratum, then permute
        lhs_points = np.zeros((n_init, dim))
        for d in range(dim):
            # Create strata boundaries
            strata = np.linspace(0, 1, n_init + 1)
            # Sample one point per stratum
            for i in range(n_init):
                lhs_points[i, d] = strata[i] + (strata[i + 1] - strata[i]) * rng.random()
            # Permute the dimension to decorrelate
            perm = rng.permutation(n_init)
            lhs_points[:, d] = lhs_points[perm, d]
        
        # Map from [0,1]^dim to [lo, hi]^dim
        lhs_points = lo + (hi - lo) * lhs_points

        # Evaluate all LHS points
        F_init = np.array([problem(x) for x in lhs_points])
        used = n_init

        # ---------- Pareto archive ----------
        def dominates(a, b):
            """Return True if a dominates b (a <= b in all, a < b in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def is_dominated(f, archive_F):
            for af in archive_F:
                if dominates(af, f):
                    return True
            return False

        def crowding_distance(archive_F):
            """Compute crowding distance for each member of the archive."""
            n = len(archive_F)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(n_obj):
                obj = np.array([af[m] for af in archive_F])
                order = np.argsort(obj)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                span = obj[order[-1]] - obj[order[0]]
                if span > 0:
                    for i in range(1, n - 1):
                        cd[order[i]] += (obj[order[i + 1]] - obj[order[i - 1]]) / span
            return cd

        def update_archive(x, f, archive_X, archive_F):
            """Add x/f to archive if non-dominated, removing dominated members."""
            if is_dominated(f, archive_F):
                return archive_X, archive_F
            for j, af in enumerate(archive_F):
                if np.allclose(f, af, atol=1e-12):
                    return archive_X, archive_F
            new_archive_X = []
            new_archive_F = []
            for ax, af in zip(archive_X, archive_F):
                if not dominates(f, af):
                    new_archive_X.append(ax)
                    new_archive_F.append(af)
            new_archive_X.append(x.copy())
            new_archive_F.append(f.copy())
            return new_archive_X, new_archive_F

        # Build initial archive from LHS points
        archive_X = []
        archive_F = []
        for i in range(n_init):
            archive_X, archive_F = update_archive(lhs_points[i], F_init[i], archive_X, archive_F)

        # Use non-dominated points as the population for DE
        # If archive is too large, downsample to a manageable population size
        pop_size = min(len(archive_X), 50)
        if len(archive_X) > pop_size:
            # Select subset based on crowding distance
            cd = crowding_distance(archive_F)
            # Sort by crowding distance descending, take top pop_size
            idx_sorted = np.argsort(cd)[::-1]
            X = np.array([archive_X[i] for i in idx_sorted[:pop_size]])
            F = np.array([archive_F[i] for i in idx_sorted[:pop_size]])
        else:
            X = np.array(archive_X)
            F = np.array(archive_F)

        # ---------- Main search loop ----------
        F_factor = 0.8
        CR = 0.9
        pop_size = len(X)

        while used < budget:
            if time.time() - start_time > deadline:
                break

            # Select parent from archive using crowding distance
            if len(archive_X) > 1:
                cd = crowding_distance(archive_F)
                i1 = rng.integers(len(archive_X))
                i2 = rng.integers(len(archive_X))
                parent_idx = i1 if cd[i1] >= cd[i2] else i2
                parent = archive_X[parent_idx]
            elif len(archive_X) == 1:
                parent = archive_X[0]
            else:
                parent = X[rng.integers(pop_size)]

            # DE mutation: select 3 distinct random indices from population
            if pop_size >= 3:
                idx1, idx2, idx3 = rng.choice(pop_size, size=3, replace=False)
                trial = X[idx1] + F_factor * (X[idx2] - X[idx3])
            else:
                # Fallback: random perturbation
                trial = parent + 0.5 * (hi - lo) * (rng.random(dim) - 0.5)

            # Binary crossover with parent
            mask = rng.random(dim) < CR
            trial = np.where(mask, trial, parent)
            trial = np.clip(trial, lo, hi)

            ft = problem(trial)
            used += 1

            # Update archive
            archive_X, archive_F = update_archive(trial, ft, archive_X, archive_F)

            # Update population: replace worst member
            worst_idx = np.argmax(np.sum(F, axis=1))
            if not dominates(F[worst_idx], ft):
                X[worst_idx] = trial
                F[worst_idx] = ft
            else:
                if rng.random() < 0.1:
                    X[worst_idx] = trial
                    F[worst_idx] = ft

            # Periodically refresh population with new LHS-style random points
            if used % 40 == 0 and used < budget - 10:
                idx = rng.integers(pop_size)
                new_x = lo + (hi - lo) * rng.random(dim)
                new_f = problem(new_x)
                used += 1
                X[idx] = new_x
                F[idx] = new_f
                archive_X, archive_F = update_archive(new_x, new_f, archive_X, archive_F)

        return X
# EVOLVE-BLOCK-END

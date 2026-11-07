"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential-evolution based multi-objective search with archive and time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random.default_rng(42)

        # Population size: use a meaningful fraction of budget
        # For 200 budget, pop=20 gives 10 generations of DE
        pop_size = max(10, min(40, budget // 5))
        # Ensure we don't exceed budget
        pop_size = min(pop_size, budget - 1)

        # Initialize population
        X = lo + span * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        evals = pop_size

        # Archive of non-dominated solutions (external)
        archive_X = [X[0].copy()]
        archive_F = [F[0].copy()]

        def dominates(a, b):
            """a dominates b if a <= b in all objectives and < in at least one."""
            return np.all(a <= b) and np.any(a < b)

        def is_nondominated(f, archive_F_list):
            """Check if f is non-dominated w.r.t. archive."""
            for af in archive_F_list:
                if dominates(af, f):
                    return False
            return True

        def update_archive(x, f, archive_X, archive_F):
            """Add to archive if non-dominated, remove dominated points."""
            if not is_nondominated(f, archive_F):
                return
            # Remove points dominated by f
            new_X = []
            new_F = []
            for ax, af in zip(archive_X, archive_F):
                if not dominates(f, af):
                    new_X.append(ax)
                    new_F.append(af)
            new_X.append(x.copy())
            new_F.append(f.copy())
            archive_X[:] = new_X
            archive_F[:] = new_F

        # Initial archive
        for i in range(pop_size):
            update_archive(X[i], F[i], archive_X, archive_F)

        # Differential Evolution parameters
        CR = 0.9
        F_mut = 0.8

        while evals < budget:
            if time.time() > deadline - 2.0:
                break

            # DE/rand/1/bin
            i = rng.integers(pop_size)
            # Pick three distinct individuals
            a = b = c = i
            while a == i:
                a = rng.integers(pop_size)
            while b == i or b == a:
                b = rng.integers(pop_size)
            while c == i or c == a or c == b:
                c = rng.integers(pop_size)

            # Mutation
            v = X[a] + F_mut * (X[b] - X[c])
            v = np.clip(v, lo, hi)

            # Crossover
            jrand = rng.integers(dim)
            trial = X[i].copy()
            for j in range(dim):
                if rng.random() < CR or j == jrand:
                    trial[j] = v[j]

            trial = np.clip(trial, lo, hi)
            ft = problem(trial)
            evals += 1

            # Selection
            if np.any(ft < F[i]) and np.all(ft <= F[i]):
                X[i] = trial
                F[i] = ft
                update_archive(trial, ft, archive_X, archive_F)

            # Occasionally inject diversity: replace worst with random or archive point
            if evals % 20 == 0 and len(archive_X) > 0:
                # Replace worst individual with a perturbed archive point
                worst_idx = int(np.argmax(np.sum(F, axis=1)))
                arch_idx = rng.integers(len(archive_X))
                new_x = archive_X[arch_idx] + 0.15 * span * rng.standard_normal(dim)
                new_x = np.clip(new_x, lo, hi)
                if evals < budget:
                    new_f = problem(new_x)
                    evals += 1
                    X[worst_idx] = new_x
                    F[worst_idx] = new_f
                    update_archive(new_x, new_f, archive_X, archive_F)

        # Final: ensure we return something (return value is ignored by evaluator)
        return X
# EVOLVE-BLOCK-END

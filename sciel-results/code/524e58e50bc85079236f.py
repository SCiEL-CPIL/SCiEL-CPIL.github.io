"""Multi-objective optimizer on RE37 using Differential Evolution with an epsilon-dominated archive."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DE-based multi-objective search with an epsilon-dominance archive for spread."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270  # 30s safety margin before 300s/330s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Archive: list of (x, f) that are non-dominated under epsilon-dominance
        # Epsilon box size for dominance check (relative to span)
        eps = 0.05 * span

        def dominates(a, b):
            """True if a dominates b (a <= b in all, a < b in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def eps_dominates(a, b):
            """True if a epsilon-dominates b: a <= b + eps in all, a < b in at least one."""
            return np.all(a <= b + eps) and np.any(a < b)

        archive_X = []
        archive_F = []
        used = 0

        def add_to_archive(x, f):
            """Try to add (x,f) to archive. Returns True if added."""
            nonlocal archive_X, archive_F
            # Check if f is dominated (or epsilon-dominated) by any archive member
            for af in archive_F:
                if eps_dominates(af, f):
                    return False
            # Remove archive members that are epsilon-dominated by f
            keep = []
            for i, af in enumerate(archive_F):
                if eps_dominates(f, af):
                    continue  # drop this one
                keep.append(i)
            archive_X = [archive_X[i] for i in keep]
            archive_F = [archive_F[i] for i in keep]
            archive_X.append(x)
            archive_F.append(f)
            return True

        # Initialize population with random points
        pop_size = min(40, budget // 5)
        pop_size = max(pop_size, 10)
        X = lo + span * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used += pop_size

        # Seed archive with initial population
        for i in range(pop_size):
            add_to_archive(X[i], F[i])

        # DE parameters
        F_mut = 0.8
        CR = 0.9
        max_iter = budget - used

        while used < budget:
            if time.time() > deadline:
                break

            # For each individual, generate an offspring
            for i in range(pop_size):
                if used >= budget or time.time() > deadline:
                    break

                # Select three distinct random individuals
                while True:
                    r1 = np.random.randint(pop_size)
                    r2 = np.random.randint(pop_size)
                    r3 = np.random.randint(pop_size)
                    if r1 != r2 and r2 != r3 and r1 != r3:
                        break

                # DE/1/bin mutation
                v = X[r1] + F_mut * (X[r2] - X[r3])
                # Crossover
                jrand = np.random.randint(dim)
                trial = X[i].copy()
                for j in range(dim):
                    if np.random.random() < CR or j == jrand:
                        trial[j] = v[j]
                trial = np.clip(trial, lo, hi)

                f_trial = problem(trial)
                used += 1

                # Archive the trial point regardless (it's a valid evaluation)
                add_to_archive(trial, f_trial)

                # Replace if trial dominates or is non-dominated with parent
                if np.any(np.all(f_trial <= F[i], axis=1) & np.any(f_trial < F[i], axis=1)):
                    X[i] = trial
                    F[i] = f_trial
                elif np.all(f_trial >= F[i]) and np.any(f_trial > F[i]):
                    # parent dominates trial, keep parent
                    pass
                else:
                    # non-dominated with parent: keep the one with better spread
                    # Use a simple rule: keep trial if archive is larger or random
                    if np.random.random() < 0.5:
                        X[i] = trial
                        F[i] = f_trial

        return X
# EVOLVE-BLOCK-END

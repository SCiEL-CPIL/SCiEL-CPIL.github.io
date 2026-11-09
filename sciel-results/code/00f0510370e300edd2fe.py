"""Multi-objective optimizer for RE34 using differential evolution with archive."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using differential evolution with an archive."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        # Structured initialization: corners, midpoints, and random
        # Corners of the hypercube
        corners = np.array([[lo[i] if (k >> i) & 1 else hi[i] for i in range(dim)]
                           for k in range(2 ** dim)])
        # Midpoints
        mid = (lo + hi) / 2.0
        # Structured points
        structured = np.vstack([corners, mid[None, :]])

        # Population size
        pop_size = min(30, max(10, budget // 6))

        # Fill remaining with random
        n_structured = min(pop_size, len(structured))
        n_random = pop_size - n_structured
        random_pts = lo + span * np.random.rand(n_random, dim)
        X = np.vstack([structured[:n_structured], random_pts])

        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Archive: keep non-dominated points
        archive_X = [X[0].copy()]
        archive_F = [F[0].copy()]

        def is_dominated(f_new, archive_F):
            """Check if f_new is dominated by any point in archive."""
            for af in archive_F:
                if np.all(af <= f_new) and np.any(af < f_new):
                    return True
            return False

        def dominates(f1, f2):
            """Check if f1 dominates f2."""
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def update_archive(x_new, f_new, archive_X, archive_F):
            """Add point to archive if non-dominated, remove dominated points."""
            if is_dominated(f_new, archive_F):
                return
            # Remove archive points dominated by new point
            keep = []
            for i in range(len(archive_X)):
                if not dominates(f_new, archive_F[i]):
                    keep.append(i)
            for i in range(len(archive_X)):
                if i not in keep:
                    archive_X.pop(i)
                    archive_F.pop(i)
            archive_X.append(x_new.copy())
            archive_F.append(f_new.copy())

        # Update archive with initial population
        for i in range(pop_size):
            update_archive(X[i], F[i], archive_X, archive_F)

        # Differential evolution parameters
        CR = 0.7  # Crossover rate
        F_factor = 0.8  # Scale factor

        def time_ok():
            return time.time() < deadline

        # Main optimization loop
        while used < budget and time_ok():
            # Select parents via tournament selection (non-dominated preferred)
            # Get non-dominated indices
            non_dom_idx = []
            for i in range(len(archive_X)):
                non_dom_idx.append(i)

            # Differential evolution: pick 3 distinct random parents
            i1 = np.random.randint(pop_size)
            i2 = np.random.randint(pop_size)
            i3 = np.random.randint(pop_size)
            # Ensure distinct
            attempts = 0
            while (i1 == i2 or i1 == i3 or i2 == i3) and attempts < 10:
                i2 = np.random.randint(pop_size)
                i3 = np.random.randint(pop_size)
                attempts += 1

            # Mutation: v = x[i1] + F_factor * (x[i2] - x[i3])
            v = X[i1] + F_factor * (X[i2] - X[i3])
            v = np.clip(v, lo, hi)

            # Binomial crossover
            jrand = np.random.randint(dim)
            trial = X[i1].copy()
            for j in range(dim):
                if np.random.rand() < CR or j == jrand:
                    trial[j] = v[j]
            trial = np.clip(trial, lo, hi)

            # Evaluate trial
            f_trial = problem(trial)
            used += 1

            # Replace worst individual in population (by sum of objectives)
            # Use the worst in terms of archive dominance or just random replacement
            # Replace the individual that is most dominated
            worst_idx = np.argmax(np.sum(F, axis=1))
            X[worst_idx] = trial
            F[worst_idx] = f_trial

            # Update archive
            update_archive(trial, f_trial, archive_X, archive_F)

            # Periodically refresh population from archive to maintain diversity
            if used % 20 == 0 and len(archive_X) > 0:
                # Replace some random population members with archive points
                n_replace = min(5, pop_size // 6)
                for _ in range(n_replace):
                    if not time_ok():
                        break
                    idx = np.random.randint(pop_size)
                    arch_idx = np.random.randint(len(archive_X))
                    X[idx] = archive_X[arch_idx].copy()
                    F[idx] = archive_F[arch_idx].copy()

        return X
# EVOLVE-BLOCK-END

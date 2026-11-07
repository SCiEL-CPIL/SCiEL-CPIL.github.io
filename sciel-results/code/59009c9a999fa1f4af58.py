# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-objective optimizer using MOPSO with external archive for ZDT4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_vars = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # MOPSO parameters
        swarm_size = 30
        archive_size = 30
        c1 = 1.5  # cognitive coefficient
        c2 = 1.5  # social coefficient
        w_min = 0.4  # minimum inertia weight
        w_max = 0.9  # maximum inertia weight

        # Initialize swarm
        X = lo + (hi - lo) * rng.random((swarm_size, n_vars))
        V = (hi - lo) * 0.1 * rng.standard_normal((swarm_size, n_vars))
        F = np.array([problem(x) for x in X])
        remaining_budget = budget - swarm_size

        # Initialize personal bests
        pbest_X = X.copy()
        pbest_F = F.copy()

        # Initialize external archive
        archive_X = []
        archive_F = []

        def dominates(a, b):
            """Check if solution a dominates b (minimization)."""
            return np.all(a <= b) and np.any(a < b)

        def add_to_archive(x, f, archive_X, archive_F):
            """Add solution to archive if non-dominated, remove dominated."""
            new_idx = len(archive_X)
            # Check if any archive member dominates the new solution
            for i in range(len(archive_X)):
                if dominates(archive_F[i], f):
                    return archive_X, archive_F

            # Remove archive members dominated by the new solution
            keep = []
            for i in range(len(archive_X)):
                if not dominates(f, archive_F[i]):
                    keep.append(i)

            new_archive_X = [archive_X[i] for i in keep]
            new_archive_F = [archive_F[i] for i in keep]

            # Add new solution
            new_archive_X.append(x)
            new_archive_F.append(f)

            # If archive exceeds capacity, remove least diverse
            if len(new_archive_X) > archive_size:
                # Compute crowding distance
                n = len(new_archive_F)
                if n < 3:
                    return np.array(new_archive_X), np.array(new_archive_F)

                crowding = np.zeros(n)
                for obj in range(n_obj):
                    sorted_idx = np.argsort(new_archive_F[:, obj])
                    crowding[sorted_idx[0]] = np.inf
                    crowding[sorted_idx[-1]] = np.inf
                    for k in range(1, n - 1):
                        if new_archive_F[sorted_idx[k + 1], obj] - new_archive_F[sorted_idx[k - 1], obj] > 1e-10:
                            crowding[sorted_idx[k]] += (
                                new_archive_F[sorted_idx[k + 1], obj] - new_archive_F[sorted_idx[k - 1], obj]
                            )

                # Remove the one with minimum crowding distance
                min_idx = np.argmin(crowding)
                new_archive_X.pop(min_idx)
                new_archive_F.pop(min_idx)

            return np.array(new_archive_X), np.array(new_archive_F)

        # Initialize archive with initial population
        for i in range(swarm_size):
            archive_X, archive_F = add_to_archive(X[i], F[i], archive_X, archive_F)
        archive_X = np.array(archive_X)
        archive_F = np.array(archive_F)

        # Compute ideal and nadir from archive for normalization
        def get_ideal_nadir(archive_F):
            if len(archive_F) == 0:
                return np.zeros(n_obj), np.ones(n_obj)
            ideal = np.min(archive_F, axis=0)
            nadir = np.max(archive_F, axis=0)
            return ideal, nadir

        # Select global best from archive using crowding distance
        def select_global_best(archive_F):
            if len(archive_F) == 0:
                return None
            if len(archive_F) == 1:
                return 0

            n = len(archive_F)
            crowding = np.zeros(n)
            for obj in range(n_obj):
                sorted_idx = np.argsort(archive_F[:, obj])
                crowding[sorted_idx[0]] = np.inf
                crowding[sorted_idx[-1]] = np.inf
                for k in range(1, n - 1):
                    if archive_F[sorted_idx[k + 1], obj] - archive_F[sorted_idx[k - 1], obj] > 1e-10:
                        crowding[sorted_idx[k]] += (
                            archive_F[sorted_idx[k + 1], obj] - archive_F[sorted_idx[k - 1], obj]
                        )

            # Pick randomly among top 3 by crowding distance
            top_k = min(3, n)
            top_indices = np.argsort(crowding)[-top_k:]
            return rng.choice(top_indices)

        # Main loop
        while remaining_budget >= swarm_size:
            # Update inertia weight
            w = w_min + (w_max - w_min) * (remaining_budget / max(1, budget - swarm_size))

            ideal, nadir = get_ideal_nadir(archive_F)
            gbest_idx = select_global_best(archive_F)
            gbest_X = archive_X[gbest_idx] if gbest_idx is not None else X[0]

            new_X = []
            new_V = []
            new_F = []

            for i in range(swarm_size):
                if remaining_budget < 1:
                    break

                r1 = rng.random(n_vars)
                r2 = rng.random(n_vars)

                # Update velocity
                V[i] = w * V[i] + c1 * r1 * (pbest_X[i] - X[i]) + c2 * r2 * (gbest_X - X[i])

                # Clip velocity
                V[i] = np.clip(V[i], -0.5 * (hi - lo), 0.5 * (hi - lo))

                # Update position
                X[i] = X[i] + V[i]
                X[i] = np.clip(X[i], lo, hi)

                # Evaluate
                f_i = problem(X[i])
                remaining_budget -= 1

                # Update personal best
                if dominates(f_i, pbest_F[i]):
                    pbest_F[i] = f_i
                    pbest_X[i] = X[i].copy()

                new_X.append(X[i])
                new_V.append(V[i])
                new_F.append(f_i)

            X = np.array(new_X)
            V = np.array(new_V)
            F = np.array(new_F)

            # Update archive with new solutions
            for i in range(len(new_F)):
                archive_X, archive_F = add_to_archive(X[i], F[i], archive_X, archive_F)

            # Ensure archive is not empty
            if len(archive_X) == 0:
                archive_X = X[:1]
                archive_F = F[:1]

        return X
# EVOLVE-BLOCK-END

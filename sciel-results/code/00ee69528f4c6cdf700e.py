# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE34 with deadline guard and efficient search."""

    def __init__(self, dim, n_obj, budget):
        self.dim = dim
        self.n_obj = n_obj
        self.budget = budget

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 300.0  # 300s per eval; safety margin below
        dim, n_obj, budget = self.dim, self.n_obj, self.budget
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        def time_ok():
            return time.time() < deadline - 5.0  # 5s safety margin

        def fast_dom(F):
            """Return boolean mask of dominated points (True = dominated)."""
            n = len(F)
            if n == 0:
                return np.zeros(0, dtype=bool)
            dom = np.zeros(n, dtype=bool)
            for i in range(n):
                if dom[i]:
                    continue
                # Does any j dominate i?
                # j dominates i if F[j] <= F[i] for all obj and F[j] < F[i] for at least one
                diff = F - F[i]  # shape (n, n_obj); diff[j] = F[j] - F[i]
                le = np.all(diff <= 0, axis=1)   # F[j] <= F[i] for all obj
                lt = np.any(diff < 0, axis=1)    # F[j] < F[i] for at least one
                d = le & lt
                d[i] = False
                if np.any(d):
                    dom[i] = True
            return dom

        # Adaptive population: keep a good archive of non-dominated solutions
        # Use a larger initial population for better coverage, then refine
        init_pop = min(40, max(10, budget // 5))
        used = 0

        # Phase 1: Latin-hypercube-like spread for initial diversity
        X = np.empty((init_pop, dim))
        for i in range(init_pop):
            # Stratified sampling per dimension
            X[i] = lo + span * ((np.arange(dim) + 1) / (dim + 1) * (i + 1) / (init_pop + 1))
            X[i] = lo + span * np.sort(np.linspace(0, 1, dim) + (i / init_pop) * (1 / init_pop))
        # Simpler: use stratified grid
        X = np.empty((init_pop, dim))
        idx = np.linspace(0, 1, init_pop + 2)[1:-1]
        for i in range(init_pop):
            X[i] = lo + span * idx[i]
        # Add some random perturbation for diversity
        X += span * np.random.randn(init_pop, dim) * 0.05
        X = np.clip(X, lo, hi)
        F = np.array([problem(x) for x in X])
        used = init_pop

        # Maintain archive of non-dominated solutions
        def update_archive(X, F):
            """Return indices of non-dominated solutions."""
            n = len(F)
            if n == 0:
                return np.array([], dtype=int)
            nd = ~fast_dom(F)
            return np.where(nd)[0]

        archive_idx = update_archive(X, F)

        # Phase 2: Iterative improvement
        # Mix of: (a) perturb best non-dominated, (b) crossover, (c) random restarts
        step_sizes = [0.15, 0.10, 0.05, 0.02, 0.01]
        ss_idx = 0

        while used < budget and time_ok():
            # Check time frequently
            if not time_ok():
                break

            # Decide action based on iteration progress
            progress = used / budget

            # Decay step size as we progress
            if progress > 0.7:
                step_sizes = [0.02, 0.01, 0.005]
            elif progress > 0.4:
                step_sizes = [0.05, 0.03, 0.01]
            else:
                step_sizes = [0.15, 0.10, 0.05]

            ss = step_sizes[ss_idx % len(step_sizes)]
            ss_idx += 1

            # Choose parent: prefer non-dominated, with some diversity
            if len(archive_idx) > 0:
                # Pick from archive with some randomness
                if np.random.rand() < 0.7:
                    parent_i = archive_idx[np.random.randint(len(archive_idx))]
                else:
                    parent_i = np.random.randint(len(X))
            else:
                parent_i = np.random.randint(len(X))

            # Generate child
            r = np.random.rand()
            if r < 0.5:
                # Mutation
                child = X[parent_i].copy()
                # Random direction mutation
                child += ss * span * np.random.randn(dim)
                child = np.clip(child, lo, hi)
            elif r < 0.8:
                # Crossover with another random parent
                parent2_i = np.random.randint(len(X))
                alpha = np.random.rand()
                child = alpha * X[parent_i] + (1 - alpha) * X[parent2_i]
                child += ss * 0.5 * span * np.random.randn(dim)
                child = np.clip(child, lo, hi)
            else:
                # Random restart (exploration)
                child = lo + span * np.random.rand(dim)

            fc = problem(child)
            used += 1

            # Update population: replace worst dominated or worst by sum
            # Find worst point to replace
            if len(F) > 1:
                # Compute dominance
                dom = fast_dom(F)
                dominated_mask = dom
                if np.any(dominated_mask):
                    # Replace a dominated point (pick worst by sum)
                    dom_indices = np.where(dominated_mask)[0]
                    worst = dom_indices[np.argmax(F[dom_indices].sum(axis=1))]
                else:
                    # All non-dominated; replace worst by sum
                    worst = np.argmax(F.sum(axis=1))
                X[worst] = child
                F[worst] = fc
            else:
                X[0] = child
                F[0] = fc

            # Update archive
            archive_idx = update_archive(X, F)

        return X
# EVOLVE-BLOCK-END

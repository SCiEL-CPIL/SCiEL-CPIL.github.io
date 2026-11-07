"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE21 with origin-focused exploration."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Initial population: mix of origin-focused and random
        pop_size = 25
        X = np.zeros((pop_size, dim))
        # 40% near origin (Rastrigin optimum)
        n_origin = max(5, pop_size // 3)
        for i in range(n_origin):
            X[i] = 0.05 * span * np.random.standard_normal(dim)
        # Rest: uniform random
        for i in range(n_origin, pop_size):
            X[i] = lo + span * np.random.random(dim)

        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Archive of all non-dominated points seen
        archive_X = X.copy()
        archive_F = F.copy()

        def update_archive(x, f):
            nonlocal archive_X, archive_F
            # Check if dominated by archive
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return  # dominated, skip
            # Remove archive points dominated by new point
            keep = []
            for j in range(len(archive_F)):
                if not (np.all(f <= archive_F[j]) and np.any(f < archive_F[j])):
                    keep.append(j)
            archive_X = np.array([archive_X[j] for j in keep])
            archive_F = np.array([archive_F[j] for j in keep])
            archive_X = np.vstack([archive_X, x.reshape(1, -1)])
            archive_F = np.vstack([archive_F, f.reshape(1, -1)])

        # Update archive with initial points
        for i in range(pop_size):
            update_archive(X[i], F[i])

        # Adaptive mutation scale
        base_scale = 0.25
        while used < budget:
            if time.time() > deadline:
                break

            progress = used / budget
            # Scale decays with progress, but maintains some exploration
            scale = base_scale * (0.3 + 0.7 * (1.0 - progress))

            # Select parent from non-dominated archive
            if len(archive_F) > 0:
                # Find non-dominated in archive
                dom_mask = np.zeros(len(archive_F), dtype=bool)
                for i in range(len(archive_F)):
                    for j in range(len(archive_F)):
                        if i != j and np.all(archive_F[j] <= archive_F[i]) and np.any(archive_F[j] < archive_F[i]):
                            dom_mask[i] = True
                            break
                nd_idx = np.where(~dom_mask)[0]
                if len(nd_idx) == 0:
                    nd_idx = np.array([np.random.randint(len(archive_F))])
                parent_idx = nd_idx[np.random.randint(len(nd_idx))]
                parent = archive_X[parent_idx]
            else:
                parent = np.random.uniform(lo, hi)

            # Mutation
            child = parent + scale * span * np.random.standard_normal(dim)

            # 15% chance: jump toward origin (exploit RE21 structure)
            if np.random.random() < 0.15:
                origin_jump = 0.02 * span * np.random.standard_normal(dim)
                child = 0.5 * child + 0.5 * origin_jump

            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1

            # Update population: replace worst by sum
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            X[worst_idx] = child
            F[worst_idx] = fc

            # Update archive
            update_archive(child, fc)

        return X
# EVOLVE-BLOCK-END

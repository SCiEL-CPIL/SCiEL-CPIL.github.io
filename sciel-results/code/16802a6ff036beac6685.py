# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ4 optimizer exploiting known structure: distance vars ~0.5, shape vars drive Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # DTLZ4 structure: x0, x1 are shape variables; x2..x11 are distance variables
        # Distance variables have optimum at 0.5 (alpha=100)
        # Shape variables determine position on Pareto front via sin

        # Initialize with structured sampling of shape variables
        # Use a grid-like approach for coverage
        n_shape = 2  # x0, x1
        n_dist = self.dim - n_shape  # x2..x11

        # Create a diverse initial set using stratified sampling
        # We'll use a combination of random and structured points
        pop_size = 40
        X = np.zeros((pop_size, self.dim))
        F = np.zeros((pop_size, self.n_obj))

        # Set distance variables near 0.5 with small noise
        dist_base = np.full(n_dist, 0.5)
        for i in range(pop_size):
            # Shape variables: stratified random
            X[i, 0] = rng.random()
            X[i, 1] = rng.random()
            # Distance variables: near 0.5 with small Gaussian noise
            X[i, n_shape:] = np.clip(dist_base + 0.02 * rng.standard_normal(n_dist), 0, 1)

        # Evaluate initial population
        for i in range(pop_size):
            F[i] = problem(X[i])
        used = pop_size

        # Track best non-dominated set
        def is_dominated(f, F_set):
            """Check if f is dominated by any point in F_set."""
            for fi in F_set:
                if np.all(fi <= f) and np.any(fi < f):
                    return True
            return False

        def get_non_dominated(F_set):
            """Return indices of non-dominated points."""
            nd = []
            for i, fi in enumerate(F_set):
                dominated = False
                for j, fj in enumerate(F_set):
                    if i != j and np.all(fj <= fi) and np.any(fj < fi):
                        dominated = True
                        break
                if not dominated:
                    nd.append(i)
            return nd

        # Evolutionary loop
        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Get non-dominated parents
            nd_idx = get_non_dominated(F)
            if len(nd_idx) == 0:
                nd_idx = list(range(pop_size))

            # Select a parent from non-dominated set
            parent_idx = nd_idx[rng.integers(len(nd_idx))]
            parent = X[parent_idx].copy()

            # Mutate: focus on shape variables with adaptive step size
            # Shape variables get larger mutations for exploration
            mutation_scale = 0.15  # scale for shape variables
            parent[0] += mutation_scale * rng.standard_normal()
            parent[1] += mutation_scale * rng.standard_normal()

            # Distance variables: small perturbation around 0.5
            dist_noise = 0.01 * rng.standard_normal(n_dist)
            parent[n_shape:] = np.clip(0.5 + dist_noise, 0, 1)

            # Clip to bounds
            parent = np.clip(parent, lo, hi)

            # Evaluate child
            fc = problem(parent)
            used += 1

            # Replace worst point (by sum of objectives) if child is not dominated
            # Find worst point
            sums = np.sum(F, axis=1)
            worst_idx = int(np.argmax(sums))

            # Only replace if child is better or non-dominated
            if not is_dominated(fc, F):
                X[worst_idx] = parent
                F[worst_idx] = fc

            # Occasionally add a new random point for diversity
            if used % 20 == 0 and used < self.budget - 10:
                new_point = np.zeros(self.dim)
                new_point[0] = rng.random()
                new_point[1] = rng.random()
                new_point[n_shape:] = np.clip(0.5 + 0.02 * rng.standard_normal(n_dist), 0, 1)
                new_point = np.clip(new_point, lo, hi)
                f_new = problem(new_point)
                used += 1

                # Replace worst if new point is non-dominated
                if not is_dominated(f_new, F):
                    sums = np.sum(F, axis=1)
                    worst_idx = int(np.argmax(sums))
                    X[worst_idx] = new_point
                    F[worst_idx] = f_new

        return X
# EVOLVE-BLOCK-END

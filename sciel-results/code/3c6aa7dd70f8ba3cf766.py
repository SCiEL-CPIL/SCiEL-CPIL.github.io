# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT3 using population-based search with non-dominated sorting."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # ZDT3: first variable determines position on Pareto front
        # Pareto front is discontinuous - only certain x1 values are on the front
        # f1 = x1, f2 = g(x)*(1 - sqrt(f1/g) - f1*sin(10*pi*f1))
        # For minimization, we want x1 in [0,1] and g minimized

        pop_size = min(30, self.budget // 5)  # population size
        if pop_size < 5:
            pop_size = 5

        # Initialize population - spread x1 across [0,1] to cover front
        X = np.random.uniform(lo, hi, size=(pop_size, self.dim))
        # Ensure x1 is spread to cover the front
        X[:, 0] = np.linspace(0.01, 0.99, pop_size) + np.random.uniform(-0.02, 0.02, pop_size)
        X[:, 0] = np.clip(X[:, 0], 0.0, 1.0)

        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track non-dominated solutions
        def is_dominated(f, F_set):
            """Check if f is dominated by any point in F_set."""
            for fp in F_set:
                if np.all(fp <= f) and np.any(fp < f):
                    return True
            return False

        def get_non_dominated(F_set):
            """Return indices of non-dominated points."""
            n = len(F_set)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    if np.all(F_set[j] <= F_set[i]) and np.any(F_set[j] < F_set[i]):
                        dominated[i] = True
                        break
            return np.where(~dominated)[0]

        # Main loop
        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break

            # Get non-dominated parents
            nd_idx = get_non_dominated(F)
            if len(nd_idx) == 0:
                nd_idx = np.arange(len(F))

            # Select parent from non-dominated set
            parent_idx = nd_idx[rng.integers(len(nd_idx))]
            parent = X[parent_idx].copy()

            # Mutation: perturb all variables, with emphasis on x1 for front coverage
            # Use adaptive mutation strength
            if used < self.budget * 0.3:
                # Exploration phase: larger mutations
                mut_scale = 0.15
            elif used < self.budget * 0.7:
                # Transition phase
                mut_scale = 0.08
            else:
                # Exploitation phase: smaller mutations
                mut_scale = 0.03

            child = parent + mut_scale * (hi - lo) * rng.standard_normal(self.dim)

            # Special handling for x1: try to stay on or near Pareto front
            # ZDT3 front has discontinuities, so we want good x1 values
            if rng.random() < 0.3:
                # Occasionally set x1 to a value likely on the front
                child[0] = rng.uniform(0.01, 0.99)

            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace worst solution in population
            # Use a simple crowding distance approximation: replace the most dominated
            # or the one with worst sum of objectives
            worst_idx = int(np.argmax(np.sum(F, axis=1)))

            # But if child is non-dominated and worst is dominated, prefer replacement
            child_dominated = is_dominated(fc, F)
            worst_dominated = is_dominated(F[worst_idx], np.delete(F, worst_idx, axis=0))

            if not child_dominated or (child_dominated and worst_dominated):
                X[worst_idx] = child
                F[worst_idx] = fc
            elif not worst_dominated and child_dominated:
                # Keep worst if it's non-dominated and child is dominated
                pass
            else:
                # Both dominated or both non-dominated: use hypervolume contribution heuristic
                # Replace if child has better sum
                if np.sum(fc) < np.sum(F[worst_idx]):
                    X[worst_idx] = child
                    F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END

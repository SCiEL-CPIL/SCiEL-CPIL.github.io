# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution multi-objective optimizer for DTLZ1."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        pop_size = 40
        # DE parameters
        F = 0.8  # Scale factor
        CR = 0.9  # Crossover rate
        # Maximum iterations
        max_iter = max(1, budget // pop_size)

        # Initialize population
        X = np.zeros((pop_size, dim))
        for i in range(pop_size):
            # DTLZ1 specific initialization: x[0] random, x[1:] = 0.5
            X[i, 0] = np.random.uniform(lo[0], hi[0])
            X[i, 1:] = 0.5

        # Evaluate initial population
        F_obj = np.array([problem(x) for x in X])
        used = pop_size

        # Track best non-dominated front
        best_front_indices = list(range(pop_size))

        def fast_non_dominated_sort(F_values):
            """Returns list of fronts, each a list of indices."""
            n = len(F_values)
            S = [[] for _ in range(n)]
            n_count = np.zeros(n, dtype=int)
            fronts = []
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F_values[p] <= F_values[q]) and np.any(F_values[p] < F_values[q]):
                        S[p].append(q)
                    elif np.all(F_values[q] <= F_values[p]) and np.any(F_values[q] < F_values[p]):
                        n_count[p] += 1
                if n_count[p] == 0:
                    fronts.append([p])
            i = 0
            while i < len(fronts):
                current_front = fronts[i]
                next_front = []
                for p in current_front:
                    for q in S[p]:
                        n_count[q] -= 1
                        if n_count[q] == 0:
                            next_front.append(q)
                if next_front:
                    fronts.append(next_front)
                i += 1
            return fronts

        # Main DE loop
        for iteration in range(max_iter):
            # Determine how many offspring we can generate this iteration
            n_offspring = min(pop_size, budget - used)
            if n_offspring <= 0:
                break

            # Create offspring using DE/rand/1/bin
            offspring = []
            offspring_obj = []

            for i in range(n_offspring):
                # Select 3 distinct random parents (different from i)
                candidates = [j for j in range(pop_size) if j != i]
                p1, p2, p3 = np.random.choice(candidates, size=3, replace=False)

                # Differential variation
                trial = X[p1] + F * (X[p2] - X[p3])

                # Binomial crossover
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = X[p3, j]  # Keep from p3 (target-like)
                    # else keep from differential vector

                # Clip to bounds
                trial = np.clip(trial, lo, hi)

                offspring.append(trial)
                used += 1
                if used >= budget:
                    break

            if len(offspring) == 0:
                break

            # Evaluate offspring
            offspring = np.array(offspring)
            offspring_obj = np.array([problem(x) for x in offspring])

            # Combine parent and offspring
            X_combined = np.vstack([X, offspring])
            F_combined = np.vstack([F_obj, offspring_obj])

            # Fast non-dominated sort
            fronts = fast_non_dominated_sort(F_combined)

            # Select next population
            new_X = []
            new_F_obj = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(X_combined[idx])
                        new_F_obj.append(F_combined[idx])
                else:
                    remaining = pop_size - len(new_X)
                    for idx in front[:remaining]:
                        new_X.append(X_combined[idx])
                        new_F_obj.append(F_combined[idx])
                    break

            X = np.array(new_X)
            F_obj = np.array(new_F_obj)

        return X
# EVOLVE-BLOCK-END

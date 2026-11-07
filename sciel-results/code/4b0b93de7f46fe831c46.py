"""Multi-objective optimizer on WFG4 using NSGA-II with adaptive mutation and local polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II inspired multi-objective optimizer for WFG4 with local search polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 315.0  # Safety margin for 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # Population size
        pop_size = min(50, self.budget // 4)
        if pop_size < 15:
            pop_size = 15

        # Initialize population with stratified sampling for better coverage
        X = np.zeros((pop_size, self.dim))
        for d in range(self.dim):
            # Use stratified sampling per dimension
            for i in range(pop_size):
                X[i, d] = lo[d] + (hi[d] - lo[d]) * ((i + rng.random()) / pop_size)
        # Shuffle rows
        rng.shuffle(X)
        F = np.array([problem(x) for x in X])
        evaluations = pop_size

        # Archive of all non-dominated points found so far
        archive_X = list(X)
        archive_F = list(F)

        def fast_non_dominated_sort(F_arr):
            """Fast non-dominated sort. Returns list of fronts (list of indices)."""
            n = len(F_arr)
            # domination count and set of dominated indices
            S = [[] for _ in range(n)]  # S[i] = set of indices dominated by i
            nd = np.zeros(n, dtype=int)  # number of solutions dominating i
            fronts = [[]]

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Check if i dominates j
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        S[i].append(j)
                        nd[j] += 1
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        nd[i] += 1

            for i in range(n):
                if nd[i] == 0:
                    fronts[0].append(i)

            i = 0
            while len(fronts[i]) > 0:
                Q = []
                for p in fronts[i]:
                    for q in S[p]:
                        nd[q] -= 1
                        if nd[q] == 0:
                            Q.append(q)
                i += 1
                if len(Q) > 0:
                    fronts.append(Q)
                else:
                    break

            return fronts[:-1] if len(fronts) > 1 else fronts

        def crowding_distance(F_arr, indices):
            """Compute crowding distance for a set of indices."""
            if len(indices) <= 2:
                return {i: float('inf') for i in indices}

            distances = {i: 0.0 for i in indices}
            for m in range(self.n_obj):
                values = [F_arr[i][m] for i in indices]
                min_val = min(values)
                max_val = max(values)
                if max_val - min_val < 1e-12:
                    continue
                sorted_idx = sorted(indices, key=lambda i: F_arr[i][m])
                distances[sorted_idx[0]] = float('inf')
                distances[sorted_idx[-1]] = float('inf')
                for k in range(1, len(sorted_idx) - 1):
                    distances[sorted_idx[k]] += (
                        F_arr[sorted_idx[k + 1]][m] - F_arr[sorted_idx[k - 1]][m]
                    ) / (max_val - min_val)
            return distances

        def select_parents(pop_X, pop_F, n_parents):
            """Select parents using non-dominated sorting and crowding distance."""
            fronts = fast_non_dominated_sort(pop_F)
            selected = []
            for front in fronts:
                if len(selected) + len(front) <= n_parents:
                    selected.extend(front)
                else:
                    cd = crowding_distance(pop_F, front)
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    selected.extend(sorted_front[:n_parents - len(selected)])
                    break
            return selected

        def crossover(x1, x2):
            """Simulated binary crossover with vectorized operations."""
            eta = 20.0
            u = rng.random(self.dim)
            beta = np.where(
                u <= 0.5,
                (2 * u) ** (1.0 / (eta + 1)),
                (1 / (2 * (1 - u))) ** (1.0 / (eta + 1))
            )
            child1 = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
            child2 = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
            return child1, child2

        def mutate(x, sigma=0.15):
            """Gaussian mutation with adaptive sigma."""
            return x + sigma * (hi - lo) * rng.standard_normal(self.dim)

        # Main loop with adaptive mutation rate
        gen = 0
        base_sigma = 0.2
        min_sigma = 0.02
        while evaluations < self.budget:
            if time.time() > deadline - 10.0:
                break

            gen += 1
            # Adaptive mutation: start large, decay over generations
            progress = min(1.0, gen / max(1, (self.budget - pop_size) // pop_size))
            sigma = base_sigma * (1.0 - progress) + min_sigma * progress

            # Select parents
            n_parents = min(pop_size, len(archive_X))
            if n_parents < 2:
                n_parents = 2
            parent_indices = select_parents(
                np.array(archive_X[:n_parents]),
                np.array(archive_F[:n_parents]),
                n_parents
            )
            # Ensure we have enough parents
            while len(parent_indices) < n_parents:
                parent_indices.append(rng.integers(n_parents))

            # Create offspring
            offspring_X = []
            offspring_F = []
            n_offspring = min(pop_size // 2, self.budget - evaluations)

            for _ in range(n_offspring):
                i1 = parent_indices[rng.integers(len(parent_indices))]
                i2 = parent_indices[rng.integers(len(parent_indices))]
                x1 = archive_X[i1]
                x2 = archive_X[i2]

                child1, child2 = crossover(x1, x2)
                child1 = mutate(child1, sigma)
                child2 = mutate(child2, sigma)

                child1 = np.clip(child1, lo, hi)
                child2 = np.clip(child2, lo, hi)

                if evaluations < self.budget:
                    f1 = problem(child1)
                    evaluations += 1
                    offspring_X.append(child1)
                    offspring_F.append(f1)

                if evaluations < self.budget:
                    f2 = problem(child2)
                    evaluations += 1
                    offspring_X.append(child2)
                    offspring_F.append(f2)

            if len(offspring_X) == 0:
                break

            # Combine archive and offspring
            all_X = archive_X + offspring_X
            all_F = archive_F + offspring_F

            # Select next generation using non-dominated sorting
            n_select = min(pop_size, len(all_X))
            all_F_arr = np.array(all_F)
            fronts = fast_non_dominated_sort(all_F_arr)
            selected = []
            for front in fronts:
                if len(selected) + len(front) <= n_select:
                    selected.extend(front)
                else:
                    cd = crowding_distance(all_F_arr, front)
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    selected.extend(sorted_front[:n_select - len(selected)])
                    break

            # Update archive with selected
            archive_X = [all_X[i] for i in selected]
            archive_F = [all_F[i] for i in selected]

        # Local polish: refine non-dominated points with small perturbations
        if evaluations < self.budget and time.time() < deadline - 20.0:
            # Get non-dominated points from current archive
            archive_F_arr = np.array(archive_F)
            fronts = fast_non_dominated_sort(archive_F_arr)
            if fronts:
                nd_indices = fronts[0]
                # Try local improvements on each non-dominated point
                for idx in nd_indices:
                    if evaluations >= self.budget or time.time() > deadline - 15.0:
                        break
                    base_x = archive_X[idx]
                    base_f = archive_F[idx]
                    # Try small perturbations
                    for _ in range(3):
                        if evaluations >= self.budget or time.time() > deadline - 15.0:
                            break
                        perturbation = 0.01 * (hi - lo) * rng.standard_normal(self.dim)
                        candidate_x = np.clip(base_x + perturbation, lo, hi)
                        candidate_f = problem(candidate_x)
                        evaluations += 1
                        # Check if candidate dominates base or is non-dominated
                        if np.all(candidate_f <= base_f) and np.any(candidate_f < base_f):
                            # Candidate is better, replace
                            archive_X[idx] = candidate_x
                            archive_F[idx] = candidate_f
                            base_x = candidate_x
                            base_f = candidate_f

        return archive_X
# EVOLVE-BLOCK-END

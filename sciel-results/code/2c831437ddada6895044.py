# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT6 using NSGA-II with time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # 300s limit, keep safety margin
        safety_margin = 5.0  # seconds before deadline to stop

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo

        # Population size
        pop_size = min(100, self.budget // 2)
        if pop_size < 10:
            pop_size = 10

        # Initial population
        rng = np.random.default_rng(42)
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Crossover and mutation parameters
        cx_prob = 0.9
        mut_prob = 0.3
        eta_mut = 20.0
        eta_cx = 15.0

        def fast_non_dominated_sort(F):
            """Fast non-dominated sorting."""
            n = len(F)
            S = [[] for _ in range(n)]  # domination relations
            n_count = [0] * n  # number of points that dominate i
            front = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_count[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_count[i] += 1

            # First front
            current_front = [i for i in range(n) if n_count[i] == 0]
            if not current_front:
                current_front = [0]
            front.append(current_front)
            idx = 0
            while current_front and idx < len(front):
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_count[j] -= 1
                        if n_count[j] == 0:
                            next_front.append(j)
                if next_front:
                    front.append(next_front)
                    idx += 1
            return front

        def crowding_distance(F, indices):
            """Calculate crowding distance for a set of indices."""
            if len(indices) == 0:
                return {}
            if len(indices) == 1:
                return {indices[0]: np.inf}

            distances = {i: 0.0 for i in indices}
            m = self.n_obj
            for k in range(m):
                sorted_idx = sorted(indices, key=lambda i: F[i, k])
                distances[sorted_idx[0]] = np.inf
                distances[sorted_idx[-1]] = np.inf
                f_min = F[sorted_idx[0], k]
                f_max = F[sorted_idx[-1], k]
                if f_max - f_min < 1e-12:
                    continue
                for i in range(1, len(sorted_idx) - 1):
                    distances[sorted_idx[i]] += (F[sorted_idx[i + 1], k] - F[sorted_idx[i - 1], k]) / (f_max - f_min)
            return distances

        def sbx_crossover(x1, x2):
            """Simulated binary crossover."""
            if rng.random() > cx_prob:
                return x1.copy(), x2.copy()
            child1 = x1.copy()
            child2 = x2.copy()
            for j in range(self.dim):
                if rng.random() < 0.5:
                    if x1[j] == x2[j]:
                        continue
                    if x1[j] > x2[j]:
                        x1[j], x2[j] = x2[j], x1[j]
                    u = rng.random()
                    beta = 1.0 + 2.0 * (x1[j] - lo[j]) / (x2[j] - x1[j] + 1e-12)
                    if u <= 1.0 / (eta_cx + 1.0):
                        alpha = 2.0 * u + (1.0 - 2.0 * u) * (beta ** (eta_cx + 1.0))
                    else:
                        alpha = (1.0 / (2.0 * (1.0 - u) + (1.0 - 2.0 * u) * ((2.0 - beta) ** (eta_cx + 1.0))))
                    c1 = 0.5 * ((1.0 + 1.0 / alpha) * x1[j] + (1.0 - 1.0 / alpha) * x2[j])
                    c2 = 0.5 * ((1.0 + alpha) * x1[j] + (1.0 - alpha) * x2[j])
                    child1[j] = np.clip(c1, lo[j], hi[j])
                    child2[j] = np.clip(c2, lo[j], hi[j])
            return child1, child2

        def polynomial_mutation(x):
            """Polynomial mutation."""
            for j in range(self.dim):
                if rng.random() < mut_prob:
                    y = np.clip(x[j], lo[j], hi[j])
                    u = rng.random()
                    if u < 0.5:
                        xy = (2.0 * u + (y - lo[j]) / (hi[j] - lo[j] + 1e-12)) ** (eta_mut + 1.0)
                        delta = (xy - 1.0)
                    else:
                        xy = (2.0 * (1.0 - u) + (hi[j] - y) / (hi[j] - lo[j] + 1e-12)) ** (eta_mut + 1.0)
                        delta = 1.0 - xy
                    x[j] = np.clip(y + delta * (hi[j] - lo[j]), lo[j], hi[j])
            return x

        # Main evolutionary loop
        while used < self.budget:
            if time.time() > deadline - safety_margin:
                break

            # Fast non-dominated sorting
            fronts = fast_non_dominated_sort(F)

            # Assign ranks
            rank = np.zeros(len(F), dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = i + 1

            # Create offspring
            offspring_X = []
            offspring_F = []
            while len(offspring_X) < pop_size:
                # Binary tournament selection
                i = rng.integers(len(F))
                j = rng.integers(len(F))
                if rank[i] < rank[j] or (rank[i] == rank[j] and F[i].sum() < F[j].sum()):
                    p1 = X[i]
                else:
                    p1 = X[j]

                i = rng.integers(len(F))
                j = rng.integers(len(F))
                if rank[i] < rank[j] or (rank[i] == rank[j] and F[i].sum() < F[j].sum()):
                    p2 = X[i]
                else:
                    p2 = X[j]

                c1, c2 = sbx_crossover(p1, p2)
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)

                if len(offspring_X) < pop_size:
                    offspring_X.append(c1)
                    offspring_F.append(problem(c1))
                    used += 1
                    if used >= self.budget:
                        break
                if len(offspring_X) < pop_size:
                    offspring_X.append(c2)
                    offspring_F.append(problem(c2))
                    used += 1
                    if used >= self.budget:
                        break

            if not offspring_X:
                break

            # Combine parent and offspring
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, np.array(offspring_F)])

            # Fast non-dominated sorting on combined
            fronts = fast_non_dominated_sort(combined_F)

            # Select next generation
            next_X = []
            next_F = []
            for front in fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                else:
                    # Need crowding distance
                    remaining = pop_size - len(next_X)
                    cd = crowding_distance(combined_F, front)
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    for idx in sorted_front[:remaining]:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                    break

            X = np.array(next_X)
            F = np.array(next_F)

        return X
# EVOLVE-BLOCK-END

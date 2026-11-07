"""Multi-objective optimizer on WFG7: NSGA-II-inspired with deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """NSGA-II-inspired multi-objective optimizer with wall-clock deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 295.0  # 5s safety margin

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(0)

        # Population size
        pop_size = min(30, self.budget // 5)
        pop_size = max(10, pop_size)

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Track all evaluated points for hypervolume (deduplicated)
        all_F = F.copy()

        # Fast non-dominated sort
        def fast_non_dominated_sort(F_mat):
            n = len(F_mat)
            S = [[] for _ in range(n)]
            n_count = [0] * n
            fronts = []
            current_front = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # i dominates j
                    if np.all(F_mat[i] <= F_mat[j]) and np.any(F_mat[i] < F_mat[j]):
                        S[i].append(j)
                        n_count[i] += 1
                    elif np.all(F_mat[j] <= F_mat[i]) and np.any(F_mat[j] < F_mat[i]):
                        n_count[j] += 1
                if n_count[i] == 0:
                    current_front.append(i)
            fronts.append(current_front)
            while current_front:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_count[j] -= 1
                        if n_count[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                current_front = next_front
            return fronts

        # Crowding distance
        def crowding_distance(F_mat, front_indices):
            if len(front_indices) <= 2:
                return {i: float('inf') for i in front_indices}
            dist = {i: 0.0 for i in front_indices}
            for m in range(self.n_obj):
                vals = [(F_mat[i][m], i) for i in front_indices]
                vals.sort(key=lambda x: x[0])
                dist[vals[0][1]] = float('inf')
                dist[vals[-1][1]] = float('inf')
                if len(vals) > 2:
                    min_val = vals[0][0]
                    max_val = vals[-1][0]
                    range_val = max_val - min_val
                    if range_val > 1e-15:
                        for k in range(1, len(vals) - 1):
                            dist[vals[k][1]] += (vals[k + 1][0] - vals[k - 1][0]) / range_val
            return dist

        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Get fronts
            fronts = fast_non_dominated_sort(F)

            # Select parents from first front (or first two if needed)
            if len(fronts) > 0 and len(fronts[0]) >= 2:
                parent_front = fronts[0]
            elif len(fronts) > 1:
                parent_front = fronts[0] + fronts[1]
            else:
                parent_front = list(range(len(F)))

            # Crowding distance on parent front
            cd = crowding_distance(F, parent_front)

            # Select two parents (prefer high crowding distance)
            n_parents = min(2, len(parent_front))
            # Weighted random selection favoring high crowding
            weights = np.array([cd[i] + 1e-6 for i in parent_front])
            weights /= weights.sum()
            parent_indices = parent_front[rng.choice(len(parent_front), size=n_parents, replace=False, p=weights)]

            p1 = X[parent_indices[0]]
            p2 = X[parent_indices[1]] if n_parents > 1 else p1

            # SBX crossover
            if rng.random() < 0.7:
                beta = 15.0
                child = np.zeros_like(p1)
                for d in range(self.dim):
                    if abs(p1[d] - p2[d]) < 1e-15:
                        child[d] = p1[d]
                    else:
                        u = rng.random()
                        if u <= 0.5:
                            alpha = 2.0 * u
                            beta_val = (1.0 + 2.0 * (p1[d] - p2[d]) / (hi[d] - lo[d] + 1e-15)) ** (beta + 1)
                            child[d] = (p1[d] * beta_val + p2[d]) / (1.0 + beta_val)
                        else:
                            alpha = 2.0 * (1.0 - u)
                            beta_val = (1.0 + 2.0 * (p2[d] - p1[d]) / (hi[d] - lo[d] + 1e-15)) ** (beta + 1)
                            child[d] = (p1[d] + p2[d] * beta_val) / (1.0 + beta_val)
            else:
                child = p1.copy()

            # Adaptive mutation: try different step sizes
            step_size = rng.choice([0.05, 0.1, 0.2, 0.3, 0.5])
            mutation = step_size * (hi - lo) * rng.standard_normal(self.dim)
            child = child + mutation
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1
            all_F = np.vstack([all_F, fc.reshape(1, -1)])

            # Replace worst point in population
            # Find worst point: dominated by most others, or highest sum if all non-dominated
            dominated_count = np.zeros(len(F))
            for i in range(len(F)):
                for j in range(len(F)):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated_count[i] += 1
            # Among most dominated, pick the one with highest objective sum
            max_dom = dominated_count.max()
            candidates = np.where(dominated_count == max_dom)[0]
            if len(candidates) > 0:
                worst = candidates[np.argmax(np.sum(F[candidates], axis=1))]
            else:
                worst = np.argmax(np.sum(F, axis=1))

            X[worst] = child
            F[worst] = fc

        return X
# EVOLVE-BLOCK-END

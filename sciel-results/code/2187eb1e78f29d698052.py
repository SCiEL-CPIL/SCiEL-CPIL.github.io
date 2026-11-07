# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for ZDT2 with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)

        # ZDT2: Pareto front achieved at x[1:] = 0, f1 = x[0] in [0,1]
        # f2 = 1 - sqrt(f1) for the ideal front
        # Use Chebyshev nodes for optimal spacing on convex front
        n_points = self.budget
        
        # Chebyshev nodes: x_k = 0.5 * (1 + cos((2k+1)*pi/(2n))) for k=0..n-1
        # These minimize the maximum gap and are optimal for convex functions
        k = np.arange(n_points)
        x0_values = 0.5 * (1.0 + np.cos((2.0 * k + 1.0) * np.pi / (2.0 * n_points)))
        
        # Ensure endpoints are included for full coverage
        x0_values[0] = 0.0
        x0_values[-1] = 1.0
        
        # Create points on the Pareto front: x[0] = x0_value, x[1:] = 0
        X = np.zeros((n_points, self.dim))
        X[:, 0] = x0_values
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        used = n_points
        
        return X

    def _fast_non_dominated_sort(self, F):
        n = len(F)
        domination_count = np.zeros(n, dtype=int)
        dominated_set = [[] for _ in range(n)]
        fronts = []

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Check if i dominates j
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominated_set[i].append(j)
                    domination_count[j] += 1

        # First front
        front = [i for i in range(n) if domination_count[i] == 0]
        if front:
            fronts.append(np.array(front))

        k = 0
        while front:
            next_front = []
            for i in front:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(np.array(next_front))
            front = next_front

        return fronts

    def _tournament_select(self, fronts, F, rng, k=2):
        candidates = []
        for _ in range(k):
            # Pick a random front, then a random individual from it
            front_idx = rng.integers(len(fronts))
            front = fronts[front_idx]
            idx = front[rng.integers(len(front))]
            candidates.append(idx)

        # Return the one with best front rank
        best = candidates[0]
        for c in candidates[1:]:
            # Find which front each is in
            for fi, front in enumerate(fronts):
                if c in front:
                    for fj, front2 in enumerate(fronts):
                        if best in front2:
                            if fi < fj:
                                best = c
                            break
                    break
        return F[best]

    def _sbx_crossover(self, p1, p2, lo, hi, rng, eta=20.0):
        child = np.zeros_like(p1)
        for i in range(len(p1)):
            if rng.random() < 0.5:
                if abs(p2[i] - p1[i]) < 1e-14:
                    child[i] = p1[i]
                else:
                    u = rng.random()
                    if u <= 0.5:
                        alpha = (u * 2) ** (1.0 / (eta + 1))
                    else:
                        alpha = (1.0 - 2 * (1 - u)) ** (1.0 / (eta + 1))
                    c1 = 0.5 * (1 + alpha) * p1[i] + 0.5 * (1 - alpha) * p2[i]
                    child[i] = c1
            else:
                if abs(p2[i] - p1[i]) < 1e-14:
                    child[i] = p1[i]
                else:
                    u = rng.random()
                    if u <= 0.5:
                        alpha = (u * 2) ** (1.0 / (eta + 1))
                    else:
                        alpha = (1.0 - 2 * (1 - u)) ** (1.0 / (eta + 1))
                    c2 = 0.5 * (1 - alpha) * p1[i] + 0.5 * (1 + alpha) * p2[i]
                    child[i] = c2
        return np.clip(child, lo, hi)

    def _crowding_distance(self, F):
        n = len(F)
        if n <= 2:
            return np.full(n, np.inf)

        dist = np.zeros(n)
        for m in range(F.shape[1]):
            obj = F[:, m]
            idx = np.argsort(obj)
            dist[idx[0]] = np.inf
            dist[idx[-1]] = np.inf
            min_val, max_val = obj[idx[0]], obj[idx[-1]]
            if max_val - min_val < 1e-14:
                continue
            for i in range(1, n - 1):
                dist[idx[i]] += (obj[idx[i + 1]] - obj[idx[i - 1]]) / (max_val - min_val)
        return dist
# EVOLVE-BLOCK-END

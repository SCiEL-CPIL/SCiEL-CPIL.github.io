"""Multi-objective optimizer on WFG7 using NSGA-II with adaptive mutation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II multi-objective optimizer with fast non-dominated sort and crowding distance."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _non_dominated_sort(self, F):
        """Fast non-dominated sort. Returns list of fronts (each a list of indices)."""
        n = len(F)
        S = [[] for _ in range(n)]   # S[i] = indices dominated by i
        n_count = np.zeros(n, dtype=int)  # number of solutions that dominate i
        fronts = [[]]

        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_count[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_count[i] += 1
            if n_count[i] == 0:
                fronts[0].append(i)

        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in S[i]:
                    n_count[j] -= 1
                    if n_count[j] == 0:
                        next_front.append(j)
            k += 1
            fronts.append(next_front)

        # Remove trailing empty fronts
        while fronts and not fronts[-1]:
            fronts.pop()

        return fronts

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for a set of indices in F."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)

        dist = np.zeros(n)
        F_sub = F[indices]
        n_obj = F_sub.shape[1]

        for m in range(n_obj):
            sorted_idx = np.argsort(F_sub[:, m])
            dist[sorted_idx[0]] = np.inf
            dist[sorted_idx[-1]] = np.inf
            f_range = F_sub[sorted_idx[-1], m] - F_sub[sorted_idx[0], m]
            if f_range > 0:
                for i in range(1, n - 1):
                    dist[sorted_idx[i]] += (F_sub[sorted_idx[i + 1], m] - F_sub[sorted_idx[i - 1], m]) / f_range

        return dist

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Population size: balance exploration with budget
        pop_size = min(20, max(10, self.budget // 10))
        budget = self.budget

        # Initialize population spread across bounds
        X = np.zeros((pop_size, self.dim))
        for i in range(pop_size):
            # Use stratified initialization for better spread
            for d in range(self.dim):
                X[i, d] = lo[d] + span[d] * ((i + 0.5) / pop_size + 0.1 * rng.random() - 0.05)
                X[i, d] = np.clip(X[i, d], lo[d], hi[d])

        F = np.array([problem(x) for x in X])
        used = pop_size

        # Adaptive mutation step
        mu = 0.15

        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Fast non-dominated sort
            fronts = self._non_dominated_sort(F)

            # Crowding distance for the first front (most important)
            if fronts[0]:
                cd = self._crowding_distance(F, fronts[0])
            else:
                cd = np.zeros(pop_size)

            # Build selection pool: prefer first front, then second, etc.
            # Create a ranking for each individual
            rank = np.full(pop_size, pop_size)
            for front_idx, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = front_idx

            # Select parents using tournament selection (size 2)
            def tournament_select():
                i1 = rng.integers(pop_size)
                i2 = rng.integers(pop_size)
                # Prefer lower rank, then higher crowding distance
                if rank[i1] < rank[i2]:
                    return i1
                elif rank[i2] < rank[i1]:
                    return i2
                else:
                    # Same rank: use crowding distance
                    cd1 = cd[i1] if i1 in fronts[0] else 0.0
                    cd2 = cd[i2] if i2 in fronts[0] else 0.0
                    return i1 if cd1 >= cd2 else i2

            p1 = tournament_select()
            p2 = tournament_select()

            # SBX-like crossover (blend)
            if rng.random() < 0.8:
                alpha = rng.random()
                child = (1 - alpha) * X[p1] + alpha * X[p2]
            else:
                child = X[p1].copy()

            # Adaptive Gaussian mutation
            noise = rng.standard_normal(self.dim)
            child = child + mu * span * noise
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace the worst individual
            # Use a combined metric: rank + inverse crowding
            scores = rank.astype(float)
            # Penalize low crowding distance in first front
            for i in range(pop_size):
                if i in fronts[0]:
                    scores[i] -= 0.1 * cd[i]

            worst = int(np.argmax(scores))
            X[worst] = child
            F[worst] = fc

            # Adapt mutation step: reduce if we're making progress
            if used % 20 == 0:
                mu *= 0.95
                mu = max(mu, 0.02)

        return X
# EVOLVE-BLOCK-END

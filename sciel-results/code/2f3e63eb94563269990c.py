# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated_sort(self, F):
        """Fast non-dominated sort. Returns list of fronts (lists of indices) and rank array."""
        n = len(F)
        S = [[] for _ in range(n)]  # S[i] = indices dominated by i
        n_dominated = np.zeros(n, dtype=int)  # number of solutions dominated by i
        fronts = []
        rank = np.full(n, -1, dtype=int)

        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dominated[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dominated[i] += 1

        # First front
        front = [i for i in range(n) if n_dominated[i] == 0]
        for i in front:
            rank[i] = 0
        fronts.append(front)

        k = 0
        while front:
            next_front = []
            for i in front:
                for j in S[i]:
                    n_dominated[j] -= 1
                    if n_dominated[j] == 0:
                        next_front.append(j)
                        rank[j] = k + 1
            k += 1
            front = next_front
            if front:
                fronts.append(front)

        return fronts, rank

    def _crowding_distance(self, F, front_indices):
        """Compute crowding distance for solutions in front_indices."""
        m = len(front_indices)
        if m <= 2:
            return np.full(m, np.inf)

        cd = np.zeros(m, dtype=float)
        F_front = F[front_indices]

        for d in range(self.n_obj):
            vals = F_front[:, d]
            min_val, max_val = vals.min(), vals.max()
            if max_val - min_val < 1e-15:
                continue
            order = np.argsort(vals)
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            for i in range(1, m - 1):
                cd[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / (max_val - min_val)

        return cd

    def _selection_tournament(self, rank, cd, k=2):
        """Tournament selection based on rank and crowding distance."""
        n = len(rank)
        idxs = np.random.choice(n, size=k, replace=False)
        best = idxs[0]
        for i in range(1, k):
            if rank[idxs[i]] < rank[best]:
                best = idxs[i]
            elif rank[idxs[i]] == rank[best]:
                if cd[idxs[i]] > cd[best]:
                    best = idxs[i]
        return best

    def _crossover(self, x1, x2, lo, hi, rng):
        """Simulated binary crossover."""
        if rng.random() < 0.9:
            beta = 1.0
            eta_c = 20.0
            child1 = np.zeros(self.dim)
            child2 = np.zeros(self.dim)
            for i in range(self.dim):
                if rng.random() < 0.5:
                    u = rng.random()
                    if u <= 1.0 / (beta + 1.0):
                        bc = (2 * u) ** (1.0 / (beta + 1.0))
                    elif u >= 1.0 - 1.0 / (beta + 1.0):
                        bc = (1.0 / (2 * (1 - u))) ** (1.0 / (beta + 1.0))
                    else:
                        bc = 1.0
                    child1[i] = 0.5 * ((1 + bc) * x1[i] + (1 - bc) * x2[i])
                    child2[i] = 0.5 * ((1 - bc) * x1[i] + (1 + bc) * x2[i])
                else:
                    child1[i] = x1[i]
                    child2[i] = x2[i]
            return np.clip(child1, lo, hi), np.clip(child2, lo, hi)
        else:
            return x1.copy(), x2.copy()

    def _mutation(self, x, lo, hi, rng, rate=0.2, sigma=0.1):
        """Polynomial mutation."""
        eta_m = 20.0
        x_mut = x.copy()
        for i in range(self.dim):
            if rng.random() < rate:
                u = rng.random()
                if u < 0.5:
                    delta1 = (2 * u) ** (1.0 / (eta_m + 1.0)) - 1.0
                else:
                    delta1 = 1.0 - (2 * (1 - u)) ** (1.0 / (eta_m + 1.0))
                x_mut[i] += delta1 * (hi[i] - lo[i])
        return np.clip(x_mut, lo, hi)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(0)

        # Initial population
        pop_size = min(20, budget // 4)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Record all evaluated points
        all_F = [F.copy()]

        # NSGA-II main loop
        while used < budget:
            if time.time() > deadline:
                break

            # Non-dominated sort
            fronts, rank = self._fast_non_dominated_sort(F)
            cd = np.zeros(len(F))
            for front in fronts:
                cd[front] = self._crowding_distance(F, front)

            # Selection and reproduction
            offspring = []
            while len(offspring) < pop_size and used + len(offspring) < budget:
                p1 = self._selection_tournament(rank, cd)
                p2 = self._selection_tournament(rank, cd)
                c1, c2 = self._crossover(X[p1], X[p2], lo, hi, rng)
                c1 = self._mutation(c1, lo, hi, rng)
                c2 = self._mutation(c2, lo, hi, rng)
                offspring.append(c1)
                offspring.append(c2)

            # Evaluate offspring
            for child in offspring:
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                fc = problem(child)
                used += 1
                # Add to population (replace worst if needed)
                if len(X) < pop_size:
                    X = np.vstack([X, child.reshape(1, -1)])
                    F = np.vstack([F, fc.reshape(1, -1)])
                else:
                    # Replace worst solution
                    worst_idx = np.argmax(np.sum(F, axis=1))
                    X[worst_idx] = child
                    F[worst_idx] = fc

            # Record
            all_F.append(F.copy())

        # Return final population
        return X
# EVOLVE-BLOCK-END

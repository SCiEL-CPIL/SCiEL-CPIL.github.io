"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT2 with NSGA-II-like selection and time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _dominance_mask(self, F):
        """Return boolean mask: True if row i is dominated by any other row."""
        n = len(F)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        return dominated

    def _fast_nondominated_sort(self, F):
        """Return list of fronts, each a list of indices."""
        n = len(F)
        S = [[] for _ in range(n)]
        nd = np.zeros(n, dtype=int)
        fronts = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    nd[i] += 1
            if nd[i] == 0:
                fronts.append([i])
        k = 0
        while k < len(fronts):
            current_front = fronts[k]
            next_front = []
            for i in current_front:
                for j in S[i]:
                    nd[j] -= 1
                    if nd[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            k += 1
        return fronts

    def _crowding_distance(self, F, indices):
        """Return crowding distance array for given indices."""
        n = len(indices)
        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([np.inf])
        dist = np.zeros(n)
        F_sub = F[indices]
        for m in range(self.n_obj):
            min_val = np.min(F_sub[:, m])
            max_val = np.max(F_sub[:, m])
            if max_val - min_val < 1e-12:
                continue
            order = np.argsort(F_sub[:, m])
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            for i in range(1, n - 1):
                dist[order[i]] += (F_sub[order[i + 1], m] - F_sub[order[i - 1], m]) / (max_val - min_val)
        return dist

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # Initial population: mix of random and structured points
        pop_size = 50
        X = np.empty((pop_size, dim))
        X[:pop_size // 2] = lo + (hi - lo) * rng.random((pop_size // 2, dim))
        # Structured initialization: vary x[0] systematically
        for i in range(pop_size // 2, pop_size):
            X[i, :] = 0.5
            X[i, 0] = (i - pop_size // 2) / (pop_size // 2)
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Main loop: NSGA-II style
        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break

            # Get non-dominated fronts
            fronts = self._fast_nondominated_sort(F)
            if not fronts:
                break

            # Select parents from first front using crowding distance
            front0 = fronts[0]
            cd = self._crowding_distance(F, front0)
            # Tournament selection: pick 2, choose one with better crowding distance
            parent1_idx = front0[rng.integers(len(front0))]
            parent2_idx = front0[rng.integers(len(front0))]
            # Ensure diversity in parent selection
            if parent1_idx == parent2_idx and len(front0) > 1:
                parent2_idx = front0[(front0.index(parent1_idx) + 1) % len(front0)]

            p1 = X[parent1_idx]
            p2 = X[parent2_idx]

            # DE-style mutation
            if rng.random() < 0.7:
                # Differential evolution mutation
                if len(front0) >= 3:
                    idxs = rng.choice(len(front0), size=3, replace=False)
                    base = X[front0[idxs[0]]]
                    diff1 = X[front0[idxs[1]]] - X[front0[idxs[2]]]
                    F_de = 0.5 + 0.5 * rng.random()
                    child = base + F_de * diff1
                else:
                    child = p1 + 0.2 * (hi - lo) * rng.standard_normal(dim)
            else:
                # Gaussian mutation
                sigma = 0.1 * (1 - 0.5 * used / self.budget)
                child = p1 + sigma * (hi - lo) * rng.standard_normal(dim)

            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1

            # Replace worst point using hypervolume contribution approximation
            # Use simple sum as proxy
            worst = int(np.argmax(np.sum(F, axis=1)))
            X[worst], F[worst] = child, fc

        return X
# EVOLVE-BLOCK-END

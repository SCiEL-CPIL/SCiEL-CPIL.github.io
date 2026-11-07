"""Multi-objective optimizer on WFG4 using NSGA-II with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 280.0  # safety margin from 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo

        # Population size: use a reasonable fraction of budget
        pop_size = min(self.budget, 100)
        # Ensure at least 10 for meaningful evolution
        pop_size = max(pop_size, 10)

        # LHS-inspired initialization for better spread
        X = self._lhs_init(pop_size, lo, hi)
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Check if we can continue
        if used >= self.budget:
            return F

        # NSGA-II main loop
        while used < self.budget:
            # Time check
            if time.time() - start_time > deadline:
                break

            # Fast non-dominated sort
            fronts = self._fast_non_dominated_sort(F)

            # Create mating pool via tournament selection on fronts
            parents = self._tournament_select(fronts, X, F, 2)

            # Crossover (SBX) and Mutation (Polynomial)
            child1, child2 = self._sbx_crossover(parents[0], parents[1], lo, hi)
            child1 = self._poly_mutation(child1, lo, hi)
            child2 = self._poly_mutation(child2, lo, hi)

            # Evaluate children
            f1 = problem(child1)
            used += 1
            if used >= self.budget:
                # Only one child fits; combine and select
                X_new = np.vstack([X, child1.reshape(1, -1)])
                F_new = np.vstack([F, f1.reshape(1, -1)])
                X, F = self._environmental_select(X_new, F_new, pop_size)
                break

            if time.time() - start_time > deadline:
                break

            f2 = problem(child2)
            used += 1

            # Combine parent and child populations
            X_new = np.vstack([X, child1.reshape(1, -1), child2.reshape(1, -1)])
            F_new = np.vstack([F, f1.reshape(1, -1), f2.reshape(1, -1)])

            # Environmental selection to get next population of size pop_size
            X, F = self._environmental_select(X_new, F_new, pop_size)

        return F

    def _lhs_init(self, n, lo, hi):
        """Latin Hypercube Sampling for initial population."""
        dim = self.dim
        X = np.zeros((n, dim))
        for j in range(dim):
            perm = np.random.permutation(n)
            X[:, j] = (perm + np.random.random(n)) / n
        X = lo + X * (hi - lo)
        return X

    def _fast_non_dominated_sort(self, F):
        """
        Fast non-dominated sort. Returns list of fronts (each a list of indices).
        """
        n = len(F)
        S = [[] for _ in range(n)]  # S[i] = indices dominated by i
        n_dominates = np.zeros(n, dtype=int)  # number of solutions dominating i
        fronts = [[]]

        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(F[i], F[j]):
                    S[i].append(j)
                    n_dominates[j] += 1
                elif self._dominates(F[j], F[i]):
                    S[j].append(i)
                    n_dominates[i] += 1

        for i in range(n):
            if n_dominates[i] == 0:
                fronts[0].append(i)

        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in S[i]:
                    n_dominates[j] -= 1
                    if n_dominates[j] == 0:
                        next_front.append(j)
            k += 1
            fronts.append(next_front)

        # Remove last empty front
        if fronts and not fronts[-1]:
            fronts.pop()

        return fronts

    def _dominates(self, a, b):
        """Check if a dominates b (minimization)."""
        return np.all(a <= b) and np.any(a < b)

    def _crowding_distance(self, F, front_indices):
        """Compute crowding distance for solutions in a front."""
        n = len(front_indices)
        if n <= 2:
            return np.full(n, np.inf)

        cd = np.zeros(n)
        m = self.n_obj

        for k in range(m):
            # Sort by k-th objective
            obj_vals = F[front_indices, k]
            sorted_order = np.argsort(obj_vals)
            # Map sorted positions back to front indices
            front_sorted = [front_indices[i] for i in sorted_order]

            # Distance for extremes
            cd[0] = np.inf
            cd[-1] = np.inf

            # Range
            obj_range = obj_vals[sorted_order[-1]] - obj_vals[sorted_order[0]]
            if obj_range > 0:
                for i in range(1, n - 1):
                    cd[i] += (obj_vals[sorted_order[i + 1]] - obj_vals[sorted_order[i - 1]]) / obj_range

        return cd

    def _environmental_select(self, X, F, pop_size):
        """Select pop_size individuals from X,F using NSGA-II environmental selection."""
        n = len(F)
        if n <= pop_size:
            return X, F

        fronts = self._fast_non_dominated_sort(F)

        X_sel = []
        F_sel = []
        total = 0

        for front in fronts:
            if total + len(front) <= pop_size:
                for idx in front:
                    X_sel.append(X[idx])
                    F_sel.append(F[idx])
                total += len(front)
            else:
                # Need to pick from this front using crowding distance
                needed = pop_size - total
                cd = self._crowding_distance(F, front)
                # Sort by crowding distance descending
                cd_sorted = np.argsort(-cd)
                for i in range(needed):
                    idx = front[cd_sorted[i]]
                    X_sel.append(X[idx])
                    F_sel.append(F[idx])
                break

        return np.array(X_sel), np.array(F_sel)

    def _tournament_select(self, fronts, X, F, k=2):
        """Tournament selection based on non-dominated rank and crowding distance."""
        n = len(X)

        # Assign ranks
        ranks = np.zeros(n, dtype=int)
        for rank, front in enumerate(fronts):
            for idx in front:
                ranks[idx] = rank

        # Compute crowding distances per front
        cd = np.zeros(n)
        for front in fronts:
            if len(front) > 2:
                front_cd = self._crowding_distance(F, front)
                for i, idx in enumerate(front):
                    cd[idx] = front_cd[i]
            elif len(front) == 1:
                cd[front[0]] = np.inf
            else:
                for idx in front:
                    cd[idx] = np.inf

        # Tournament
        def compare(i, j):
            """Return True if i is better than j."""
            if ranks[i] != ranks[j]:
                return ranks[i] < ranks[j]
            return cd[i] > cd[j]

        # Select two parents
        parent_indices = []
        for _ in range(2):
            candidates = np.random.choice(n, size=k, replace=False)
            best = candidates[0]
            for c in candidates[1:]:
                if compare(c, best):
                    best = c
            parent_indices.append(best)

        return X[parent_indices[0]], X[parent_indices[1]]

    def _sbx_crossover(self, p1, p2, lo, hi):
        """Simulated Binary Crossover."""
        eta = 20.0
        child1 = np.zeros_like(p1)
        child2 = np.zeros_like(p2)

        for j in range(len(p1)):
            if np.random.random() < 0.5:
                x1, x2 = p1[j], p2[j]
            else:
                x1, x2 = p2[j], p1[j]

            if x1 == x2:
                child1[j] = x1
                child2[j] = x2
                continue

            if x1 > x2:
                x1, x2 = x2, x1

            if np.random.random() <= 1.0 / (eta + 1.0):
                beta = (2.0 * np.random.random()) ** (1.0 / (eta + 1.0))
            else:
                beta = (1.0 / (2.0 - 2.0 * np.random.random())) ** (1.0 / (eta + 1.0))

            child1[j] = 0.5 * ((1.0 + beta) * x1 + (1.0 - beta) * x2)
            child2[j] = 0.5 * ((1.0 - beta) * x1 + (1.0 + beta) * x2)

            # Clip to bounds
            child1[j] = np.clip(child1[j], lo[j], hi[j])
            child2[j] = np.clip(child2[j], lo[j], hi[j])

        return child1, child2

    def _poly_mutation(self, x, lo, hi, eta=20.0, prob=None):
        """Polynomial mutation."""
        if prob is None:
            prob = 1.0 / self.dim

        x_mut = x.copy()
        span = hi - lo

        for j in range(self.dim):
            if np.random.random() < prob:
                x_j = x_mut[j]
                lo_j, hi_j = lo[j], hi[j]

                if x_j == lo_j:
                    delta1 = 1.0 - (x_j - lo_j) / span[j] if span[j] > 0 else 1.0
                    delta2 = 1.0
                elif x_j == hi_j:
                    delta1 = 1.0
                    delta2 = 1.0 - (hi_j - x_j) / span[j] if span[j] > 0 else 1.0
                else:
                    delta1 = (x_j - lo_j) / span[j] if span[j] > 0 else 0.5
                    delta2 = (hi_j - x_j) / span[j] if span[j] > 0 else 0.5

                if np.random.random() < 0.5:
                    xy = 2.0 * np.random.random() - delta1
                    val = (1.0 + xy) ** (1.0 / (eta + 1.0))
                    deltaq = 0.5 * (val - 1.0)
                else:
                    xy = 2.0 * (1.0 - np.random.random() - delta2)
                    val = (1.0 + xy) ** (1.0 / (eta + 1.0))
                    deltaq = 0.5 * (val - 1.0)

                x_mut[j] = x_j + deltaq * span[j]
                x_mut[j] = np.clip(x_mut[j], lo_j, hi_j)

        return x_mut
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _nondom_rank(self, F):
        """Return non-dominated ranks for each point in F (lower rank = better)."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)

        ranks = np.full(n, -1, dtype=int)
        remaining = list(range(n))
        current_rank = 0

        while remaining:
            n_rem = len(remaining)
            front = []
            for i in remaining:
                dominated = False
                for j in remaining:
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
                if not dominated:
                    front.append(i)
            for i in front:
                ranks[i] = current_rank
            remaining = [i for i in remaining if i not in front]
            current_rank += 1

        return ranks

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for a set of indices within F."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)

        cd = np.zeros(n)
        m = self.n_obj
        for k in range(m):
            vals = F[indices, k]
            order = np.argsort(vals)
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            val_range = vals[order[-1]] - vals[order[0]]
            if val_range > 1e-12:
                for i in range(1, n - 1):
                    cd[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / val_range
        return cd

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, 10s safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        budget = self.budget
        dim = self.dim

        # Initial population
        pop_size = min(50, budget // 2)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track all evaluated points for hypervolume
        all_X = list(X)
        all_F = list(F)

        # Main NSGA-II loop
        while used < budget:
            # Check time
            if time.time() > deadline:
                break

            # Combine current population with offspring
            # Create offspring
            n_off = pop_size
            children_X = np.zeros((n_off, dim))
            children_F = np.zeros((n_off, self.n_obj))

            # Rank and crowding for selection
            ranks = self._nondom_rank(F)
            # Group by rank
            max_rank = ranks.max() + 1
            fronts = [[] for _ in range(max_rank)]
            for i in range(len(F)):
                fronts[ranks[i]].append(i)

            # Tournament selection
            for ci in range(n_off):
                # Select two parents via tournament
                p1 = self._tournament(F, ranks, rng)
                p2 = self._tournament(F, ranks, rng)

                # SBX crossover
                if rng.random() < 0.9:
                    child1, child2 = self._sbx(X[p1], X[p2], lo, hi, rng)
                else:
                    child1 = X[p1].copy()
                    child2 = X[p2].copy()

                # Mutation
                if rng.random() < 0.9:
                    child1 = self._polynomial_mutation(child1, lo, hi, rng)
                if rng.random() < 0.9:
                    child2 = self._polynomial_mutation(child2, lo, hi, rng)

                children_X[ci] = child1
                if ci + 1 < n_off and rng.random() < 0.5:
                    children_X[ci + 1] = child2
                    ci += 1

            # Evaluate children
            for ci in range(n_off):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                fc = problem(children_X[ci])
                children_F[ci] = fc
                used += 1
                all_X.append(children_X[ci])
                all_F.append(fc)

            # Combine and select
            X_comb = np.vstack([X, children_X[:used - pop_size] if used > pop_size else children_X])
            F_comb = np.vstack([F, children_F[:used - pop_size] if used > pop_size else children_F])

            # Non-dominated sorting
            ranks_comb = self._nondom_rank(F_comb)
            max_rank = ranks_comb.max() + 1
            fronts = [[] for _ in range(max_rank)]
            for i in range(len(F_comb)):
                fronts[ranks_comb[i]].append(i)

            # Select next population
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(X_comb[idx])
                        new_F.append(F_comb[idx])
                else:
                    # Need to select from this front based on crowding
                    need = pop_size - len(new_X)
                    cd = self._crowding_distance(F_comb, front)
                    order = np.argsort(-cd)
                    for oi in order[:need]:
                        idx = front[oi]
                        new_X.append(X_comb[idx])
                        new_F.append(F_comb[idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

        return X

    def _tournament(self, F, ranks, rng, k=2):
        """Tournament selection based on rank."""
        n = len(F)
        if k == 1:
            return int(rng.integers(n))
        idxs = rng.integers(0, n, size=k)
        best = idxs[0]
        for i in range(1, k):
            if ranks[idxs[i]] < ranks[best]:
                best = idxs[i]
        return best

    def _sbx(self, x1, x2, lo, hi, rng, eta=20.0):
        """Simulated binary crossover."""
        dim = len(x1)
        child1 = np.zeros(dim)
        child2 = np.zeros(dim)
        for i in range(dim):
            if rng.random() < 0.5:
                if abs(x1[i] - x2[i]) > 1e-14:
                    if rng.random() < 0.5:
                        beta = 1 + 2 * (x1[i] - lo[i]) / (x2[i] - x1[i])
                    else:
                        beta = 1 + 2 * (hi[i] - x2[i]) / (x1[i] - x2[i])
                    alpha = 2 - beta ** (-(eta + 1))
                    rand = rng.random()
                    if rand <= 1 / alpha:
                        beta_q = (rand * alpha) ** (1 / (eta + 1))
                    else:
                        beta_q = (1 / (2 - rand * alpha)) ** (1 / (eta + 1))
                    child1[i] = 0.5 * ((1 + beta_q) * x1[i] + (1 - beta_q) * x2[i])
                    child2[i] = 0.5 * ((1 - beta_q) * x1[i] + (1 + beta_q) * x2[i])
                else:
                    child1[i] = x1[i]
                    child2[i] = x2[i]
            else:
                child1[i] = x2[i]
                child2[i] = x1[i]
        child1 = np.clip(child1, lo, hi)
        child2 = np.clip(child2, lo, hi)
        return child1, child2

    def _polynomial_mutation(self, x, lo, hi, rng, eta=20.0):
        """Polynomial mutation."""
        dim = len(x)
        for i in range(dim):
            if rng.random() < 1.0 / dim:
                if x[i] > lo[i] and x[i] < hi[i]:
                    rand = rng.random()
                    if rand < 0.5:
                        xy = (x[i] - lo[i]) / (hi[i] - lo[i])
                        mut_pow = 2 + (1 - xy) * (eta + 1)
                        delta = (2 * rand) ** (1 / mut_pow) - 1
                    else:
                        xy = (hi[i] - x[i]) / (hi[i] - lo[i])
                        mut_pow = 2 + xy * (eta + 1)
                        delta = 1 - (2 * (1 - rand)) ** (1 / mut_pow)
                    x[i] = x[i] + delta * (hi[i] - lo[i])
                    x[i] = np.clip(x[i], lo[i], hi[i])
        return x
# EVOLVE-BLOCK-END

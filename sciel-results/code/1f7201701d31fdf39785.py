# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated_sort(self, F):
        """Fast non-dominated sorting. Returns list of fronts (lists of indices)."""
        n = len(F)
        S = [[] for _ in range(n)]
        ncount = np.zeros(n, dtype=int)
        fronts = []

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # i dominates j?
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    ncount[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    ncount[i] += 1

        current_front = [i for i in range(n) if ncount[i] == 0]
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in S[i]:
                    ncount[j] -= 1
                    if ncount[j] == 0:
                        next_front.append(j)
            current_front = next_front

        # Assign rank based on front index
        rank = np.full(n, -1, dtype=int)
        for r, front in enumerate(fronts):
            for idx in front:
                rank[idx] = r
        return rank

    def _crowding_distance(self, F, idx):
        """Crowding distance for a set of indices."""
        m = len(idx)
        cd = np.zeros(m, dtype=float)
        if m <= 2:
            cd[:] = np.inf
            return cd

        for k in range(self.n_obj):
            vals = F[idx, k]
            lo = vals.min()
            hi = vals.max()
            rng = hi - lo
            if rng < 1e-30:
                continue
            order = np.argsort(vals)
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            for t in range(1, m - 1):
                cd[order[t]] += (vals[order[t + 1]] - vals[order[t - 1]]) / rng
        return cd

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 275  # safety margin under 300s

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size: balance exploration and exploitation
        # For WFG4 with 12 dims, 200 budget, a population of ~20-30 is reasonable
        pop_size = 20

        # Initial population
        X = lo + span * np.random.rand(pop_size, dim)
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size

        # Adaptive sigma
        sigma = 0.3
        sigma_min = 0.01
        sigma_decay = 0.995

        # Keep track of best points found (for final return if needed, though return is ignored)
        # We just need to maximize the set of recorded non-dominated points.

        while used < budget:
            if time.time() > deadline:
                break

            # Non-dominated sorting
            rank = self._fast_non_dominated_sort(F)

            # Crowding distance per front
            cd = np.zeros(len(F), dtype=float)
            for r in np.unique(rank):
                if r < 0:
                    continue
                idx = np.where(rank == r)[0]
                cd[idx] = self._crowding_distance(F, idx)

            # Selection: tournament selection based on (rank, crowding)
            def select():
                a = int(np.random.randint(len(F)))
                b = int(np.random.randint(len(F)))
                if rank[a] != rank[b]:
                    return a if rank[a] < rank[b] else b
                return a if cd[a] > cd[b] else b

            p1 = select()
            p2 = select()

            # Crossover: Simulated Binary Crossover (SBX)
            # For simplicity and speed, we use a blend of parents with mutation
            # But SBX is better for continuous variables. Let's implement a simple blend.
            # Actually, for high dimensions, simple mutation of a single parent or blend is fine.
            # Let's use a blend crossover with high probability, otherwise mutation.
            
            if np.random.rand() < 0.8:
                # Blend crossover
                alpha = np.random.rand(dim)
                child = alpha * X[p1] + (1 - alpha) * X[p2]
            else:
                # Just pick one parent
                if np.random.rand() < 0.5:
                    child = X[p1].copy()
                else:
                    child = X[p2].copy()

            # Mutation: Gaussian noise
            # Adaptive sigma
            sigma *= sigma_decay
            if sigma < sigma_min:
                sigma = sigma_min
            
            child = child + sigma * span * np.random.randn(dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replacement: replace the worst individual
            # Worst is highest rank, then lowest crowding
            score = rank.astype(float) * 1000.0 - cd
            worst = int(np.argmax(score))
            X[worst] = child
            F[worst] = fc

        return X
# EVOLVE-BLOCK-END

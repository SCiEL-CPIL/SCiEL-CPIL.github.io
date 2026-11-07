# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II-style multi-objective optimizer with LHS init, crowding distance, and deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        pop_size = min(50, max(10, budget // 4))

        # LHS initialization for good initial spread
        rng = np.random.default_rng(0)
        X = self._lhs_init(pop_size, dim, lo, hi, rng)
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Reference point for hypervolume-based selection
        ref = np.array([1705.0, 12.0, 0.3])

        while used < budget:
            if time.time() > deadline:
                break

            # Combine parent and offspring
            # Create offspring via crossover + mutation
            n_off = pop_size
            children_X = np.zeros((n_off, dim))
            children_F = np.zeros((n_off, n_obj))

            for i in range(n_off):
                if time.time() > deadline:
                    break
                # Tournament selection on non-dominated fronts
                p1, p2 = self._tournament_select(X, F, pop_size, rng)
                # Simulated binary crossover (SBX)
                c1, c2 = self._sbx(p1, p2, lo, hi, rng)
                # Polynomial mutation
                c1 = self._mutate(c1, lo, hi, rng)
                c2 = self._mutate(c2, lo, hi, rng)
                children_X[i] = c1
                children_F[i] = problem(c1)
                used += 1
                if used >= budget:
                    break
                if used + 1 <= budget:
                    children_X[i + 1] = c2
                    children_F[i + 1] = problem(c2)
                    used += 1
                else:
                    i += 1
                    break

            if used >= budget:
                break

            # Truncate children to actual number generated
            actual_children = min(used - pop_size, n_off * 2)
            if actual_children <= 0:
                break
            children_X = children_X[:actual_children]
            children_F = children_F[:actual_children]

            # Combine parents and children
            all_X = np.vstack([X, children_X])
            all_F = np.vstack([F, children_F])
            all_size = len(all_X)

            # Fast non-dominated sorting
            fronts = self._fast_non_dominated_sort(all_F)

            # Select next generation
            next_X = np.zeros((pop_size, dim))
            next_F = np.zeros((pop_size, n_obj))
            next_count = 0
            for front_idx in fronts:
                if next_count + len(front_idx) <= pop_size:
                    idxs = front_idx
                    next_X[next_count:next_count + len(idxs)] = all_X[idxs]
                    next_F[next_count:next_count + len(idxs)] = all_F[idxs]
                    next_count += len(idxs)
                else:
                    # Need crowding distance to fill the last front
                    remaining = pop_size - next_count
                    if remaining > 0:
                        cd = self._crowding_distance(all_F[front_idx])
                        sorted_idx = np.argsort(-cd)
                        selected = front_idx[sorted_idx[:remaining]]
                        next_X[next_count:next_count + remaining] = all_X[selected]
                        next_F[next_count:next_count + remaining] = all_F[selected]
                        next_count += remaining
                    break

            X = next_X
            F = next_F

        # Return (return value is ignored by evaluator, but keep interface)
        return X

    def _lhs_init(self, n, dim, lo, hi, rng):
        """Latin Hypercube Sampling initialization."""
        X = np.zeros((n, dim))
        for d in range(dim):
            perm = rng.permutation(n)
            X[:, d] = lo[d] + (hi[d] - lo[d]) * (perm + rng.random(n)) / n
        return X

    def _tournament_select(self, X, F, k, rng):
        """Tournament selection based on non-dominated rank."""
        fronts = self._fast_non_dominated_sort(F)
        rank = np.zeros(len(F), dtype=int)
        for i, front in enumerate(fronts):
            rank[front] = i

        def pick():
            idx = rng.integers(len(F))
            return idx

        i1 = pick()
        i2 = pick()
        if rank[i1] < rank[i2]:
            return X[i1], X[i2]
        elif rank[i2] < rank[i1]:
            return X[i2], X[i1]
        else:
            # Same rank, use crowding distance
            front = fronts[rank[i1]]
            cd = self._crowding_distance(F[front])
            # Map global indices to local
            local_pos1 = np.where(front == i1)[0][0]
            local_pos2 = np.where(front == i2)[0][0]
            if cd[local_pos1] > cd[local_pos2]:
                return X[i1], X[i2]
            else:
                return X[i2], X[i1]

    def _sbx(self, p1, p2, lo, hi, rng, eta=20.0):
        """Simulated Binary Crossover."""
        c1 = p1.copy()
        c2 = p2.copy()
        for i in range(len(p1)):
            if rng.random() < 0.5:
                u = rng.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (eta + 1))
                else:
                    beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        return c1, c2

    def _mutate(self, x, lo, hi, rng, eta=20.0):
        """Polynomial mutation."""
        x = x.copy()
        for i in range(len(x)):
            if rng.random() < 0.5:
                span = hi[i] - lo[i]
                if span <= 0:
                    continue
                u = rng.random()
                if u < 0.5:
                    delta = (2 * u) ** (1.0 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                x[i] += delta * span
        return np.clip(x, lo, hi)

    def _fast_non_dominated_sort(self, F):
        """NSGA-II fast non-dominated sorting. Returns list of fronts (arrays of indices)."""
        n = len(F)
        S = [[] for _ in range(n)]  # domination count
        n_dom = np.zeros(n, dtype=int)  # number of solutions dominated by i
        fronts = []

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(F[i], F[j]):
                    S[i].append(j)
                elif self._dominates(F[j], F[i]):
                    n_dom[i] += 1

        front = []
        for i in range(n):
            if n_dom[i] == 0:
                front.append(i)

        k = 0
        while front:
            fronts.append(np.array(front))
            next_front = []
            for i in front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            front = next_front
            k += 1

        return fronts

    def _dominates(self, a, b):
        """Check if a dominates b (minimization)."""
        return np.all(a <= b) and np.any(a < b)

    def _crowding_distance(self, F):
        """Compute crowding distance for a set of solutions."""
        n = len(F)
        if n <= 2:
            return np.full(n, np.inf)

        cd = np.zeros(n)
        n_obj = F.shape[1]
        for m in range(n_obj):
            obj = F[:, m]
            min_val = obj.min()
            max_val = obj.max()
            if max_val - min_val < 1e-14:
                continue
            sorted_idx = np.argsort(obj)
            cd[sorted_idx[0]] = np.inf
            cd[sorted_idx[-1]] = np.inf
            for i in range(1, n - 1):
                cd[sorted_idx[i]] += (obj[sorted_idx[i + 1]] - obj[sorted_idx[i - 1]]) / (max_val - min_val)
        return cd
# EVOLVE-BLOCK-END

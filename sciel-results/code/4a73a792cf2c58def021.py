"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for RE34 (3 objectives, 5 variables).
    Strategy:
      - Latin Hypercube Sampling for initial population to spread coverage.
      - NSGA-II style non-dominated sorting + crowding distance for selection.
      - Adaptive mutation: starts large, shrinks over time.
      - Deadline guard to stay within wall-clock limits.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _lhs(self, n, dim, rng):
        """Latin Hypercube Sampling: n points in [0,1]^dim."""
        pts = np.zeros((n, dim))
        for j in range(dim):
            perm = rng.permutation(n)
            pts[:, j] = (perm + rng.random(n)) / n
        return pts

    def _fast_non_dominated_sort(self, F):
        """
        Fast non-dominated sorting.
        Returns list of fronts, each a list of indices.
        """
        n = len(F)
        S = [[] for _ in range(n)]  # S[i] = indices dominated by i
        n_dom = np.zeros(n, dtype=int)  # number of points dominating i
        fronts = []

        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dom[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dom[i] += 1

        # First front
        front = [i for i in range(n) if n_dom[i] == 0]
        while front:
            fronts.append(front)
            next_front = []
            for i in front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            front = next_front

        return fronts

    def _crowding_distance(self, F, indices):
        """
        Compute crowding distance for a set of indices into F.
        Returns array of distances for those indices.
        """
        if len(indices) <= 2:
            return np.full(len(indices), np.inf)

        dist = np.zeros(len(indices))
        F_sub = F[indices]
        m = self.n_obj

        for k in range(m):
            order = np.argsort(F_sub[:, k])
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            f_range = F_sub[order[-1], k] - F_sub[order[0], k]
            if f_range == 0:
                continue
            for i in range(1, len(order) - 1):
                dist[order[i]] += (F_sub[order[i + 1], k] - F_sub[order[i - 1], k]) / f_range

        return dist

    def _select_parents(self, F, n_parents):
        """
        Select n_parents indices using non-dominated sorting + crowding.
        """
        fronts = self._fast_non_dominated_sort(F)
        selected = []
        for front in fronts:
            if len(selected) + len(front) <= n_parents:
                selected.extend(front)
            else:
                need = n_parents - len(selected)
                cd = self._crowding_distance(F, front)
                order = np.argsort(-cd)
                selected.extend(front[order[:need]].tolist())
                break
        # If still not enough (shouldn't happen), pad randomly
        while len(selected) < n_parents:
            selected.append(np.random.randint(len(F)))
        return selected[:n_parents]

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Initial population via LHS
        pop_size = min(20, self.budget // 5)
        X = lo + span * self._lhs(pop_size, self.dim, rng)
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track all evaluated points for hypervolume
        all_F = F.copy()
        all_X = X.copy()

        # Adaptive mutation scale
        mut_scale = 0.3  # fraction of range

        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Select parents
            n_sel = max(2, pop_size // 2)
            parent_idx = self._select_parents(F, n_sel)

            # Generate children with adaptive mutation
            n_children = min(pop_size, self.budget - used)
            children_X = np.zeros((n_children, self.dim))
            for i in range(n_children):
                p = parent_idx[rng.integers(len(parent_idx))]
                # Adaptive mutation: Gaussian with shrinking scale
                noise = rng.standard_normal(self.dim) * mut_scale * span
                child = X[p] + noise
                child = np.clip(child, lo, hi)
                children_X[i] = child

            # Evaluate children
            children_F = np.array([problem(x) for x in children_X])
            used += n_children

            # Combine parents and children, select next population
            combined_X = np.vstack([X, children_X])
            combined_F = np.vstack([F, children_F])

            # Update all evaluated points
            all_F = np.vstack([all_F, children_F])
            all_X = np.vstack([all_X, children_X])

            # Selection: keep best pop_size from combined
            fronts = self._fast_non_dominated_sort(combined_F)
            next_idx = []
            for front in fronts:
                if len(next_idx) + len(front) <= pop_size:
                    next_idx.extend(front)
                else:
                    need = pop_size - len(next_idx)
                    cd = self._crowding_distance(combined_F, front)
                    order = np.argsort(-cd)
                    next_idx.extend(front[order[:need]].tolist())
                    break

            X = combined_X[next_idx]
            F = combined_F[next_idx]

            # Adapt mutation scale: shrink over time
            progress = used / self.budget
            mut_scale = 0.3 * (1.0 - 0.5 * progress) + 0.05

        return X
# EVOLVE-BLOCK-END

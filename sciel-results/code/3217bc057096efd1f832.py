"""Combined: population-based MOEA with exact-hypervolume-driven truncation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based multi-objective optimizer with HV-driven selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _dominates(self, a, b):
        return np.all(a <= b) and np.any(a < b)

    def _is_nondominated(self, point, archive):
        for a in archive:
            if self._dominates(a, point):
                return False
        return True

    def _add_to_archive(self, point, archive):
        if not self._is_nondominated(point, archive):
            return archive
        new_archive = []
        for a in archive:
            if not self._dominates(point, a):
                new_archive.append(a)
        new_archive.append(point)
        return new_archive

    def _hypervolume_3d(self, points, ref):
        """Exact 3D hypervolume via slicing (same as scoring uses)."""
        valid = [p for p in points if all(p[i] < ref[i] for i in range(3))]
        if len(valid) == 0:
            return 0.0
        if len(valid) == 1:
            p = valid[0]
            return (ref[0] - p[0]) * (ref[1] - p[1]) * (ref[2] - p[2])

        valid.sort(key=lambda x: x[0])
        hv = 0.0
        for i in range(len(valid)):
            p = valid[i]
            x0_lo = p[0]
            x0_hi = valid[i - 1][0] if i > 0 else ref[0]
            if x0_hi <= x0_lo:
                continue
            w = x0_hi - x0_lo
            hv += w * (ref[1] - p[1]) * (ref[2] - p[2])
        return max(0.0, hv)

    def _fast_non_dominated_sort(self, F_arr):
        """Return list of fronts (each a list of indices)."""
        n = len(F_arr)
        domination_count = np.zeros(n, dtype=int)
        dominated = [[] for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                    dominated[i].append(j)
                    domination_count[j] += 1
                elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                    dominated[j].append(i)
                    domination_count[i] += 1

        fronts = []
        current_front = [i for i in range(n) if domination_count[i] == 0]
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in dominated[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front = next_front
        return fronts

    def _crowding_distance(self, F_arr, front_indices):
        n = len(front_indices)
        if n <= 2:
            return np.full(n, np.inf)
        distances = np.zeros(n)
        for m in range(self.n_obj):
            f_vals = F_arr[front_indices, m]
            min_val = f_vals.min()
            max_val = f_vals.max()
            if max_val - min_val < 1e-15:
                continue
            sorted_idx = np.argsort(f_vals)
            distances[sorted_idx[0]] = np.inf
            distances[sorted_idx[-1]] = np.inf
            for k in range(1, n - 1):
                distances[sorted_idx[k]] += (
                    f_vals[sorted_idx[k + 1]] - f_vals[sorted_idx[k - 1]]
                ) / (max_val - min_val)
        return distances

    def _sbx_crossover(self, x1, x2, lo, hi, eta=15.0, rng=None):
        child = x1.copy()
        if rng is None:
            rng = np.random
        if rng.random() < 0.5:
            for d in range(self.dim):
                if rng.random() < 0.5:
                    if abs(x2[d] - x1[d]) > 1e-15:
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (eta + 1.0))
                        else:
                            beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                        child[d] = 0.5 * (
                            (1.0 + beta) * x1[d] + (1.0 - beta) * x2[d]
                        )
        return np.clip(child, lo, hi)

    def _hv_driven_select(self, X_new, F_new, pop_size, ref, rng):
        """Select next population maximizing HV of non-dominated subset."""
        fronts = self._fast_non_dominated_sort(F_new)
        X_next = []
        F_next = []

        for i, front in enumerate(fronts):
            if len(X_next) + len(front) <= pop_size:
                for idx in front:
                    X_next.append(X_new[idx])
                    F_next.append(F_new[idx])
            else:
                remaining = pop_size - len(X_next)
                cd_front = self._crowding_distance(F_new, front)
                sorted_indices = np.argsort(-cd_front)
                for k in range(remaining):
                    idx = front[sorted_indices[k]]
                    X_next.append(X_new[idx])
                    F_next.append(F_new[idx])
                break

        X_sel = np.array(X_next) if X_next else np.empty((0, self.dim))
        F_sel = np.array(F_next) if F_next else np.empty((0, self.n_obj))
        return X_sel, F_sel

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 315.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        ref = np.array([3.0, 5.0, 7.0])

        total_budget = self.budget
        used = 0

        def evaluate(x):
            nonlocal used
            if used >= total_budget:
                return None
            f = problem(np.clip(x, lo, hi))
            used += 1
            return f

        # Stratified initialization for good initial coverage
        pop_size = 25
        n_init = min(pop_size, total_budget)
        X = np.zeros((n_init, self.dim))
        for i in range(n_init):
            for d in range(self.dim):
                X[i, d] = lo[d] + range_[d] * ((i + np.random.rand()) / n_init)

        F = np.array([evaluate(x) for x in X])

        # Keep track of best archive for final return
        best_archive = []
        for f in F:
            best_archive = self._add_to_archive(f, best_archive)

        # Main evolutionary loop
        while used < total_budget:
            if time.time() > deadline:
                break

            # Compute fronts and crowding for current population
            fronts = self._fast_non_dominated_sort(F)
            crowding_dists = []
            for front in fronts:
                crowding_dists.append(self._crowding_distance(F, front))

            # Build rank and crowding arrays for tournament
            n_pop = len(F)
            rank = np.zeros(n_pop, dtype=int)
            cd = np.zeros(n_pop)
            for r, front in enumerate(fronts):
                cd_front = crowding_dists[r]
                for i, idx in enumerate(front):
                    rank[idx] = r
                    cd[idx] = cd_front[i]

            def tournament_select():
                idx1 = np.random.randint(n_pop)
                idx2 = np.random.randint(n_pop)
                if rank[idx1] < rank[idx2]:
                    return idx1
                elif rank[idx2] < rank[idx1]:
                    return idx2
                else:
                    return idx1 if cd[idx1] > cd[idx2] else idx2

            # Generate offspring (one at a time, re-evaluate each)
            child = None
            fc = None
            while used < total_budget:
                if time.time() > deadline:
                    break

                p1_idx = tournament_select()
                p2_idx = tournament_select()
                parent1 = X[p1_idx]
                parent2 = X[p2_idx]

                if np.random.rand() < 0.9:
                    child = self._sbx_crossover(parent1, parent2, lo, hi, eta=15.0)
                else:
                    child = parent1.copy()

                # Progressive mutation: wide early, narrow later
                progress = used / total_budget
                step = 0.25 * (1.0 - 0.7 * progress)
                step *= np.random.uniform(0.5, 1.5)
                child = child + step * range_ * np.random.randn(self.dim)
                child = np.clip(child, lo, hi)

                fc = evaluate(child)
                if fc is None:
                    break
                break  # one offspring per generation

            if fc is None:
                break

            # Combine and select
            X_new = np.vstack([X, child.reshape(1, -1)])
            F_new = np.vstack([F, fc.reshape(1, -1)])

            X, F = self._hv_driven_select(X_new, F_new, pop_size, ref, np.random)

            # Update best archive
            for f in F:
                best_archive = self._add_to_archive(f, best_archive)

        return best_archive
# EVOLVE-BLOCK-END

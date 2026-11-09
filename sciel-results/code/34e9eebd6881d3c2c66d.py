"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II-like multi-objective optimizer with DE mutation and crowding distance."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        # Initial population
        pop_size = min(30, self.budget // 3)
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size

        # Archive of all evaluated points
        archive_X = [tuple(x) for x in X]
        archive_F = [tuple(f) for f in F]

        def _dominance_matrix(Farr):
            """Return matrix M[i,j]=True if i dominates j."""
            n = len(Farr)
            M = np.zeros((n, n), dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(Farr[i] <= Farr[j]) and np.any(Farr[i] < Farr[j]):
                        M[i, j] = True
            return M

        def _non_dominated_sort(Farr):
            """Fast non-dominated sort. Returns list of fronts (lists of indices)."""
            n = len(Farr)
            M = _dominance_matrix(Farr)
            dominance_count = np.sum(M, axis=1)  # number of points dominated by i
            dominated_by = [np.where(M[:, j])[0] for j in range(n)]  # points that dominate j

            fronts = []
            current_front = [i for i in range(n) if dominance_count[i] == 0]
            if not current_front:
                # All dominated? shouldn't happen with at least one point
                current_front = [0]

            while current_front:
                fronts.append(current_front)
                next_front = set()
                for i in current_front:
                    for j in dominated_by[i]:
                        dominance_count[j] -= 1
                        if dominance_count[j] == 0:
                            next_front.add(j)
                current_front = sorted(next_front)

            return fronts

        def _crowding_distance(Farr, indices):
            """Compute crowding distance for given indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            dist = np.zeros(n)
            for m in range(self.n_obj):
                vals = Farr[indices, m]
                vmin, vmax = vals.min(), vals.max()
                if vmax - vmin < 1e-30:
                    dist[:] = np.inf
                    continue
                # Sort indices by this objective
                order = np.argsort(vals)
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                for k in range(1, n - 1):
                    dist[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / (vmax - vmin)
            return dist

        def _selection(Farr, Xarr, n_select):
            """Select n_select parents using non-dominated sort + crowding."""
            fronts = _non_dominated_sort(Farr)
            selected = []
            for front in fronts:
                if len(selected) + len(front) <= n_select:
                    selected.extend(front)
                else:
                    # Need to fill from this front using crowding
                    need = n_select - len(selected)
                    cd = _crowding_distance(Farr, front)
                    order = np.argsort(-cd)
                    selected.extend([front[i] for i in order[:need]])
                    break
            return selected

        def _de_mutation(parents_X, parents_F, rng):
            """Differential evolution mutation using vector differences."""
            n_parents = len(parents_X)
            if n_parents < 3:
                # Fallback to random mutation
                p = parents_X[rng.integers(n_parents)]
                return p + 0.1 * span * rng.standard_normal(self.dim)

            i = rng.integers(n_parents)
            a, b, c = rng.choice(n_parents, size=3, replace=False)
            while a == i:
                a = rng.integers(n_parents)
            while b == i or b == a:
                b = rng.integers(n_parents)
            while c == i or c == a or c == b:
                c = rng.integers(n_parents)

            F_de = 0.8
            child = parents_X[a] + F_de * (parents_X[b] - parents_X[c])
            # Crossover
            cross_rate = 0.5
            mask = rng.random(self.dim) < cross_rate
            if not mask.any():
                mask[rng.integers(self.dim)] = True
            child = np.where(mask, child, parents_X[i])
            return np.clip(child, lo, hi)

        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Get non-dominated front for selection
            fronts = _non_dominated_sort(F)
            n_front = len(fronts[0])

            # Generate one child at a time to stay within budget
            parents_idx = _selection(F, X, max(3, n_front))
            if len(parents_idx) < 3:
                parents_idx = list(range(len(X)))

            parent_X = X[parents_idx]
            parent_F = F[parents_idx]

            child = _de_mutation(parent_X, parent_F, rng)
            fc = problem(child)
            used += 1

            # Update population: replace worst point (by sum of objectives)
            # But only if child is not dominated by all existing points
            dominated = np.all(F <= fc, axis=1) & np.any(F < fc, axis=1)
            if not dominated.all():
                # Find worst point to replace
                worst = int(np.argmax(np.sum(F, axis=1)))
                X[worst] = child
                F[worst] = fc
            else:
                # Child is dominated by everything; still record but maybe skip replacement
                # For hypervolume, we want diverse points, so replace worst anyway
                worst = int(np.argmax(np.sum(F, axis=1)))
                X[worst] = child
                F[worst] = fc

            # Add to archive
            archive_X.append(tuple(child))
            archive_F.append(tuple(fc))

        return X
# EVOLVE-BLOCK-END

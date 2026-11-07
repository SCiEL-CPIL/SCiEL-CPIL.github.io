"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective evolutionary strategy with DE-style mutation and deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random.default_rng(0)

        # Population size
        pop_size = min(40, max(10, budget // 5))

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def dominates(a, b):
            """Return True if a dominates b (a <= b in all objectives, a < b in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_indices(F):
            """Return boolean mask of non-dominated points."""
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    if dominates(F[j], F[i]):
                        dominated[i] = True
                        break
            return ~dominated

        def crowding_distance(F, indices):
            """Compute crowding distance for given indices."""
            n = len(indices)
            if n <= 2:
                return np.ones(n) * np.inf
            cd = np.zeros(n)
            for m in range(n_obj):
                fvals = F[indices, m]
                min_f = fvals.min()
                max_f = fvals.max()
                if max_f - min_f < 1e-12:
                    cd[:] = np.inf
                    continue
                order = np.argsort(fvals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for k in range(1, n - 1):
                    cd[order[k]] += (fvals[order[k + 1]] - fvals[order[k - 1]]) / (max_f - min_f)
            return cd

        def env_selection(F, indices, k):
            """Select k points from indices using environmental selection."""
            if len(indices) <= k:
                return list(indices)
            # Non-dominated sort
            remaining = list(indices)
            fronts = []
            while remaining:
                nd = non_dominated_indices(F[remaining])
                front = [remaining[i] for i in range(len(remaining)) if nd[i]]
                fronts.append(front)
                remaining = [remaining[i] for i in range(len(remaining)) if not nd[i]]
            
            selected = []
            for front in fronts:
                if len(selected) + len(front) <= k:
                    selected.extend(front)
                else:
                    # Need to trim this front by crowding distance
                    need = k - len(selected)
                    cd = crowding_distance(F, front)
                    order = np.argsort(-cd)  # descending crowding distance
                    selected.extend([front[i] for i in order[:need]])
                    break
            return selected

        # Main loop
        while used < budget:
            # Check deadline
            if time.time() > deadline - 2.0:
                break

            # Find non-dominated points
            nd_mask = non_dominated_indices(F)
            if np.any(nd_mask):
                nd_indices = np.where(nd_mask)[0]
            else:
                nd_indices = np.arange(len(F))

            # DE-style mutation: pick random parent and two other random parents
            n_nd = len(nd_indices)
            parent_idx = nd_indices[rng.integers(n_nd)]
            
            # Pick two other distinct parents from non-dominated set
            other_indices = np.delete(nd_indices, np.where(nd_indices == parent_idx)[0])
            if len(other_indices) >= 2:
                a_idx, b_idx = other_indices[rng.integers(len(other_indices), size=2)]
            else:
                # Fallback: pick any two other points
                all_other = np.delete(np.arange(len(F)), parent_idx)
                a_idx, b_idx = all_other[rng.integers(len(all_other), size=2)]

            # DE mutation
            F1 = F[parent_idx]
            F2 = F[a_idx]
            F3 = F[b_idx]
            child = F1 + 0.8 * (F2 - F3)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Combine with population
            X_new = np.vstack([X, child.reshape(1, -1)])
            F_new = np.vstack([F, fc.reshape(1, -1)])

            # Environmental selection: keep pop_size best
            all_indices = np.arange(len(X_new))
            selected = env_selection(F_new, all_indices, pop_size)
            X = X_new[selected]
            F = F_new[selected]

        return X
# EVOLVE-BLOCK-END

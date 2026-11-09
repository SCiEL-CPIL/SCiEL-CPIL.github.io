"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II-inspired multi-objective optimizer with adaptive mutation and deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _dominance_matrix(self, F):
        """Return n x n boolean matrix where D[i,j] means i dominates j."""
        n = len(F)
        D = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(n):
                if i != j and np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    D[i, j] = True
        return D

    def _non_dominated_sort(self, F):
        """Fast non-dominated sort. Returns list of fronts (lists of indices)."""
        n = len(F)
        D = self._dominance_matrix(F)
        S = [set() for _ in range(n)]  # S[i] = set of indices dominated by i
        nd = np.zeros(n, dtype=int)    # nd[i] = number of points that dominate i
        fronts = []
        for i in range(n):
            for j in range(n):
                if D[i, j]:
                    S[i].add(j)
                elif D[j, i]:
                    nd[i] += 1
            if nd[i] == 0:
                fronts.append([i])
        k = 0
        while k < len(fronts) and fronts[k]:
            Q = fronts[k]
            fronts[k] = []
            for i in Q:
                for j in S[i]:
                    nd[j] -= 1
                    if nd[j] == 0:
                        if k + 1 >= len(fronts):
                            fronts.append([])
                        fronts[k + 1].append(j)
            k += 1
        # Remove empty fronts at the end
        while fronts and not fronts[-1]:
            fronts.pop()
        return fronts

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for a set of indices."""
        if not indices:
            return np.zeros(len(F))
        cd = np.zeros(len(F))
        n_obj = self.n_obj
        for m in range(n_obj):
            vals = F[indices, m]
            order = np.argsort(vals)
            cd[indices[order[0]]] = np.inf
            cd[indices[order[-1]]] = np.inf
            fmin, fmax = vals[order[0]], vals[order[-1]]
            if fmax - fmin > 1e-15:
                for i in range(1, len(order) - 1):
                    cd[indices[order[i]]] += (vals[order[i + 1]] - vals[order[i - 1]]) / (fmax - fmin)
        return cd

    def _tournament_select(self, F, X, fronts, cd):
        """Binary tournament selection based on front rank then crowding distance."""
        n = len(F)
        # Assign ranks
        ranks = np.zeros(n, dtype=int)
        for rank, front in enumerate(fronts):
            for idx in front:
                ranks[idx] = rank

        def better(i, j):
            if ranks[i] < ranks[j]:
                return True
            if ranks[i] > ranks[j]:
                return False
            return cd[i] > cd[j]

        i1, i2 = rng.integers(n), rng.integers(n)
        return i1 if better(i1, i2) else i2

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim, n_obj = self.dim, self.n_obj

        # Population size: balance between exploration and evaluation budget
        pop_size = min(30, max(10, self.budget // 6))

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Adaptive mutation parameters
        base_mutation = 0.2
        jump_mutation = 0.4
        jump_prob = 0.2
        mutation_decay = 0.995

        # Track best hypervolume proxy for adaptive control
        best_sum = np.sum(F, axis=1).min()

        while used < self.budget:
            # Deadline guard
            if time.time() > deadline:
                break

            # Non-dominated sort
            fronts = self._non_dominated_sort(F)

            # Crowding distance for first front (most important)
            if fronts:
                cd = self._crowding_distance(F, fronts[0])
                # For simplicity, use first front's cd for all
                full_cd = np.zeros(len(F))
                full_cd[fronts[0]] = cd[fronts[0]]
                # Assign decreasing cd to later fronts
                for k, front in enumerate(fronts[1:], 1):
                    full_cd[front] = np.inf - k * 1e-6
            else:
                full_cd = np.zeros(len(F))

            # Generate one child per iteration (single-child NSGA-II variant)
            parent_idx = self._tournament_select(F, X, fronts, full_cd)
            parent = X[parent_idx]

            # Adaptive mutation
            current_mutation = base_mutation * (mutation_decay ** (used // 20))
            if rng.random() < jump_prob:
                scale = jump_mutation
            else:
                scale = current_mutation

            child = parent + scale * (hi - lo) * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace worst individual based on front rank and crowding
            # Find the individual to replace: worst front, then lowest crowding
            ranks = np.zeros(len(F), dtype=int)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank

            # Add child to population temporarily for ranking
            F_new = np.vstack([F, fc.reshape(1, -1)])
            fronts_new = self._non_dominated_sort(F_new)

            # Find worst individual in the combined population
            ranks_new = np.zeros(len(F_new), dtype=int)
            for rank, front in enumerate(fronts_new):
                for idx in front:
                    ranks_new[idx] = rank

            # Crowding distance for the worst front
            max_rank = ranks_new.max()
            worst_front_indices = [i for i in range(len(F_new)) if ranks_new[i] == max_rank]
            cd_new = self._crowding_distance(F_new, worst_front_indices)

            # Replace the individual with lowest crowding distance in worst front
            worst_in_front = worst_front_indices[int(np.argmin(cd_new[worst_front_indices]))]

            if worst_in_front < len(F):
                X[worst_in_front] = child
                F[worst_in_front] = fc
            else:
                # This shouldn't happen, but keep as safety
                X[-1] = child
                F[-1] = fc

            # Occasionally inject a random solution to maintain diversity
            if used % 15 == 0 and used < self.budget:
                rand_idx = int(np.argmin(full_cd))  # Replace lowest crowding
                X[rand_idx] = lo + (hi - lo) * rng.random(dim)
                F[rand_idx] = problem(X[rand_idx])
                used += 1

        return X
# EVOLVE-BLOCK-END

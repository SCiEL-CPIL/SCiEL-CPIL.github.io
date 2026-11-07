"""Multi-objective optimizer on ZDT3 with NSGA-II style evolution."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for ZDT3."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated_sort(self, F):
        """Assign non-dominated fronts. Returns list of lists of indices."""
        n = len(F)
        S = [[] for _ in range(n)]
        nd = [0] * n
        fronts = []

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    nd[i] += 0  # i dominates j
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    nd[i] += 1

        front0 = [i for i in range(n) if nd[i] == 0]
        fronts.append(front0)
        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in S[i]:
                    nd[j] -= 1
                    if nd[j] == 0:
                        next_front.append(j)
            k += 1
            if next_front:
                fronts.append(next_front)
        return fronts

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for a set of indices in F."""
        n = len(indices)
        if n == 0:
            return []
        if n == 1:
            return [float('inf')]
        dist = np.zeros(n)
        for m in range(self.n_obj):
            vals = F[indices, m]
            sorted_idx = np.argsort(vals)
            dist[sorted_idx[0]] = float('inf')
            dist[sorted_idx[-1]] = float('inf')
            fmin = vals[sorted_idx[0]]
            fmax = vals[sorted_idx[-1]]
            if fmax - fmin > 1e-12:
                for i in range(1, n - 1):
                    dist[sorted_idx[i]] += (vals[sorted_idx[i + 1]] - vals[sorted_idx[i - 1]]) / (fmax - fmin)
        return dist

    def _select_next_gen(self, F, X, budget_remaining, lo, hi, rng, deadline):
        """Select next generation using tournament + crowding distance."""
        n = len(F)
        fronts = self._fast_non_dominated_sort(F)

        # Assign rank to each individual
        rank = np.zeros(n, dtype=int)
        for r, front in enumerate(fronts):
            for idx in front:
                rank[idx] = r

        # Compute crowding distance per front
        cd = np.zeros(n)
        for front in fronts:
            if len(front) > 1:
                d = self._crowding_distance(F, front)
                for i, idx in enumerate(front):
                    cd[idx] = d[i]
            else:
                cd[front[0]] = float('inf')

        # Tournament selection
        pop_size = max(10, min(100, self.budget // 2))
        while pop_size > 1 and self.budget - (len(F) - pop_size) < 2:
            pop_size -= 1

        new_X = []
        new_F = []
        evals_used = 0

        while len(new_X) < pop_size and evals_used < budget_remaining:
            if time.time() > deadline:
                break

            # Tournament of size 3
            tsize = min(3, len(new_X) + len(F))
            candidates = list(np.random.choice(n, size=tsize, replace=False))

            # Rank candidates by (rank, -crowding_distance)
            def sort_key(i):
                return (rank[i], -cd[i])

            candidates.sort(key=sort_key)
            parent_idx = candidates[0]

            # Crossover: SBX-like blend
            other_idx = np.random.randint(n)
            if parent_idx == other_idx:
                other_idx = (parent_idx + 1) % n

            p1 = X[parent_idx]
            p2 = X[other_idx]

            # Blend crossover
            beta = 0.1
            child = (1 - beta) * p1 + beta * p2
            # Add some random perturbation
            child += 0.05 * (hi - lo) * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            new_X.append(child)
            new_F.append(fc)
            evals_used += 1

            # Also add a pure mutation of the best parent sometimes
            if evals_used < budget_remaining and len(new_X) < pop_size and rng.random() < 0.3:
                mut = parent_idx
                child2 = X[mut] + 0.1 * (hi - lo) * rng.standard_normal(self.dim)
                child2 = np.clip(child2, lo, hi)
                fc2 = problem(child2)
                new_X.append(child2)
                new_F.append(fc2)
                evals_used += 1

        return np.array(new_X), np.array(new_F), evals_used

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Initial population
        pop_size = min(100, max(20, self.budget // 2))
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Main evolution loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Fast non-dominated sort
            fronts = self._fast_non_dominated_sort(F)

            # Assign rank
            n = len(F)
            rank = np.zeros(n, dtype=int)
            for r, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = r

            # Crowding distance
            cd = np.zeros(n)
            for front in fronts:
                if len(front) > 1:
                    d = self._crowding_distance(F, front)
                    for i, idx in enumerate(front):
                        cd[idx] = d[i]
                else:
                    cd[front[0]] = float('inf')

            # Generate children to replace worst individuals
            # Select parents via tournament
            n_children = min(pop_size, self.budget - used)
            if n_children <= 0:
                break

            children_X = []
            children_F = []

            for _ in range(n_children):
                if time.time() > deadline:
                    break

                # Tournament selection
                tsize = min(3, n)
                candidates = list(rng.integers(n, size=tsize))
                # Remove duplicates
                candidates = list(set(candidates))
                while len(candidates) < tsize:
                    candidates.append(int(rng.integers(n)))

                def sort_key(i):
                    return (rank[i], -cd[i])

                candidates.sort(key=sort_key)
                p1_idx = candidates[0]
                p2_idx = candidates[1] if len(candidates) > 1 else candidates[0]

                p1 = X[p1_idx]
                p2 = X[p2_idx]

                # Blend crossover + mutation
                beta = 0.1
                child = (1 - beta) * p1 + beta * p2
                child += 0.08 * (hi - lo) * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)

                fc = problem(child)
                children_X.append(child)
                children_F.append(fc)
                used += 1

                if used >= self.budget:
                    break

            if not children_X:
                break

            children_X = np.array(children_X)
            children_F = np.array(children_F)

            # Combine parent and child populations
            all_X = np.vstack([X, children_X])
            all_F = np.vstack([F, children_F])

            # Select next generation
            all_fronts = self._fast_non_dominated_sort(all_F)
            all_n = len(all_F)
            all_rank = np.zeros(all_n, dtype=int)
            for r, front in enumerate(all_fronts):
                for idx in front:
                    all_rank[idx] = r

            all_cd = np.zeros(all_n)
            for front in all_fronts:
                if len(front) > 1:
                    d = self._crowding_distance(all_F, front)
                    for i, idx in enumerate(front):
                        all_cd[idx] = d[i]
                else:
                    all_cd[front[0]] = float('inf')

            # Select best pop_size individuals
            def individual_key(i):
                return (all_rank[i], -all_cd[i])

            all_indices = list(range(all_n))
            all_indices.sort(key=individual_key)
            selected = all_indices[:pop_size]

            X = all_X[selected]
            F = all_F[selected]

        return X
# EVOLVE-BLOCK-END

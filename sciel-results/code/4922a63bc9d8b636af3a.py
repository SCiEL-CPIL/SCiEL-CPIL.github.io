"""Rewritten program for WFG7 multi-objective optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using NSGA-II-style approach with adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        # Population size: balance exploration and exploitation
        pop_size = min(30, max(10, self.budget // 10))
        # Reserve some budget for local refinement
        main_budget = int(self.budget * 0.85)
        polish_budget = self.budget - main_budget

        # Latin Hypercube-style initialization for better coverage
        X = np.zeros((pop_size, self.dim))
        for d in range(self.dim):
            perm = rng.permutation(pop_size)
            X[perm, d] = np.linspace(0, 1, pop_size) + rng.random(pop_size) * (1.0 / pop_size)
            X[:, d] = np.clip(X[:, d], 0, 1)
        X = lo + X * range_

        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(F):
            """Return fronts as list of index arrays, and rank array."""
            n = len(F)
            S = [[] for _ in range(n)]  # Dominated sets
            n_dom = np.zeros(n, dtype=int)  # Number of dominators
            fronts = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    fronts.append([i])
            k = 0
            while k < len(fronts):
                current_front = fronts[k]
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                k += 1
            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for given indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(self.n_obj):
                vals = F[indices, m]
                min_v, max_v = vals.min(), vals.max()
                if max_v - min_v < 1e-15:
                    continue
                order = np.argsort(vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for k in range(1, n - 1):
                    cd[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / (max_v - min_v)
            return cd

        def hypervolume_contribution(F, ref, idx):
            """Approximate hypervolume contribution of point idx."""
            # Simple approximation: product of improvements over neighbors
            if len(F) == 0:
                return 0
            f = F[idx]
            if np.any(f >= ref):
                return 0
            # Compute volume of box where this point is better than all others in each dim
            contrib = 1.0
            for m in range(self.n_obj):
                others = np.delete(F, idx, axis=0)
                better_mask = others[:, m] >= f[m]
                if np.any(better_mask):
                    min_other = np.min(others[better_mask, m])
                    contrib *= (ref[m] - min_other) if min_other > f[m] else (ref[m] - f[m])
                else:
                    contrib *= (ref[m] - f[m])
            return contrib

        # Main evolutionary loop
        while used < main_budget:
            if time.time() > deadline - 5:
                break

            fronts = fast_non_dominated_sort(F)
            if not fronts:
                break

            # Select parents from best front(s)
            parent_pool = []
            for front in fronts:
                if len(parent_pool) + len(front) >= pop_size // 2:
                    parent_pool.extend(front[:pop_size // 2 - len(parent_pool)])
                    break
                parent_pool.extend(front)

            if not parent_pool:
                parent_pool = list(range(len(F)))

            # Generate children
            n_children = min(pop_size // 2, main_budget - used)
            if n_children <= 0:
                break

            children_X = np.zeros((n_children, self.dim))
            children_F = np.zeros((n_children, self.n_obj))

            # Adaptive mutation rate based on convergence
            front_size = len(fronts[0])
            convergence = front_size / max(1, len(F))
            base_rate = 0.05 + 0.15 * (1 - convergence)

            for i in range(n_children):
                # Tournament selection from parent pool
                p1_idx = parent_pool[rng.integers(len(parent_pool))]
                p2_idx = parent_pool[rng.integers(len(parent_pool))]

                # Blend
                alpha = rng.random()
                child = alpha * X[p1_idx] + (1 - alpha) * X[p2_idx]

                # Adaptive mutation
                rate = base_rate * rng.exponential(1.0)
                child = child + rate * range_ * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)

                children_X[i] = child
                children_F[i] = problem(child)
                used += 1

                if used >= main_budget:
                    n_children = i + 1
                    break

            # Combine and select
            all_X = np.vstack([X, children_X[:n_children]])
            all_F = np.vstack([F, children_F[:n_children]])

            # Truncate to pop_size using non-dominated sort + crowding distance
            fronts = fast_non_dominated_sort(all_F)
            new_X = np.zeros((pop_size, self.dim))
            new_F = np.zeros((pop_size, self.n_obj))
            count = 0
            for front in fronts:
                if count + len(front) <= pop_size:
                    new_X[count:count + len(front)] = all_X[front]
                    new_F[count:count + len(front)] = all_F[front]
                    count += len(front)
                else:
                    # Need to select from this front using crowding distance
                    need = pop_size - count
                    cd = crowding_distance(all_F, front)
                    order = np.argsort(cd)[::-1]
                    selected = front[order[:need]]
                    new_X[count:count + need] = all_X[selected]
                    new_F[count:count + need] = all_F[selected]
                    count += need
                    break

            X = new_X
            F = new_F

        # Polish phase: local refinement around best points
        if used < self.budget:
            if time.time() > deadline - 3:
                return X

            # Get non-dominated points
            fronts = fast_non_dominated_sort(F)
            nd_indices = fronts[0] if fronts else [0]

            # Local search around each non-dominated point
            for idx in nd_indices:
                if used >= self.budget or time.time() > deadline - 3:
                    break
                # Try several small perturbations
                for _ in range(3):
                    if used >= self.budget or time.time() > deadline - 3:
                        break
                    # Adaptive step size
                    step = 0.02 * range_ * rng.standard_normal(self.dim)
                    candidate = np.clip(X[idx] + step, lo, hi)
                    fc = problem(candidate)
                    used += 1
                    # Replace if it dominates or is non-dominated and improves crowding
                    dominated = np.any(np.all(F <= fc, axis=1) & np.any(F < fc, axis=1))
                    if not dominated:
                        # Replace worst point in terms of hypervolume contribution
                        worst = int(np.argmax(np.sum(F, axis=1)))
                        if np.sum(fc) < np.sum(F[worst]):
                            X[worst] = candidate
                            F[worst] = fc

        return X
# EVOLVE-BLOCK-END

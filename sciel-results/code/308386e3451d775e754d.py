# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, keep 20s margin

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget

        # Latin Hypercube Sampling for initial population
        n_init = min(40, budget // 4)
        X = np.zeros((n_init, dim))
        for d in range(dim):
            X[:, d] = lo[d] + (hi[d] - lo[d]) * (np.arange(n_init) + np.random.uniform(0, 1, n_init)) / n_init
        # Shuffle rows to decorrelate
        idx = np.random.permutation(n_init)
        X = X[idx]
        X = np.clip(X, lo, hi)

        F = np.array([problem(x) for x in X])
        used = n_init

        def dominates(a, b):
            """Return True if a dominates b (a is better in all objectives, strictly better in at least one)"""
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_mask(F):
            """Return boolean mask of non-dominated points in F"""
            n = len(F)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]:
                    continue
                for j in range(n):
                    if i == j or not mask[j]:
                        continue
                    if dominates(F[j], F[i]):
                        mask[i] = False
                        break
            return mask

        def crowding_distance(F, nd_indices):
            """Compute crowding distance for non-dominated points"""
            n = len(nd_indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for k in range(F.shape[1]):
                obj_vals = F[nd_indices, k]
                sorted_idx = np.argsort(obj_vals)
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                obj_range = obj_vals[sorted_idx[-1]] - obj_vals[sorted_idx[0]]
                if obj_range > 1e-15:
                    for i in range(1, n - 1):
                        cd[sorted_idx[i]] += (obj_vals[sorted_idx[i + 1]] - obj_vals[sorted_idx[i - 1]]) / obj_range
            return cd

        # Main evolutionary loop
        while used < budget:
            if time.time() > deadline:
                break

            # Find non-dominated set
            nd_mask = non_dominated_mask(F)
            nd_indices = np.where(nd_mask)[0]

            # Compute crowding distance for non-dominated points
            cd = crowding_distance(F, nd_indices)

            # Selection: pick parents from non-dominated set with best crowding distance
            # Use tournament-like selection: pick 2 random non-dominated, take better crowding
            parent_indices = []
            for _ in range(2):
                c1 = nd_indices[np.random.randint(len(nd_indices))]
                c2 = nd_indices[np.random.randint(len(nd_indices))]
                # Get crowding distance for these two
                pos1 = np.where(nd_indices == c1)[0][0]
                pos2 = np.where(nd_indices == c2)[0][0]
                if cd[pos1] >= cd[pos2]:
                    parent_indices.append(c1)
                else:
                    parent_indices.append(c2)

            p1 = X[parent_indices[0]]
            p2 = X[parent_indices[1]]

            # SBX crossover
            if np.random.rand() < 0.8:
                eta = 20.0
                child1 = np.zeros(dim)
                child2 = np.zeros(dim)
                for d in range(dim):
                    u = np.random.rand()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (eta + 1))
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                    child1[d] = 0.5 * ((1 + beta) * p1[d] + (1 - beta) * p2[d])
                    child2[d] = 0.5 * ((1 - beta) * p1[d] + (1 + beta) * p2[d])
                children = [child1, child2]
            else:
                children = [p1.copy(), p2.copy()]

            # Adaptive mutation: larger early, smaller late
            progress = used / budget
            mut_rate = 0.5 * (1 - progress) + 0.1
            mut_scale = 0.3 * (1 - progress) + 0.05

            for i, child in enumerate(children):
                if used >= budget or time.time() > deadline:
                    break
                for d in range(dim):
                    if np.random.rand() < mut_rate:
                        child[d] += mut_scale * (hi[d] - lo[d]) * np.random.randn()
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1

                # Replace worst point in population
                # Use sum of normalized objectives as proxy
                F_min = F.min(axis=0)
                F_max = F.max(axis=0)
                F_range = F_max - F_min
                F_range[F_range < 1e-15] = 1.0
                F_norm = (F - F_min) / F_range
                worst = int(np.argmax(F_norm.sum(axis=1)))
                X[worst] = child
                F[worst] = fc

        return X
# EVOLVE-BLOCK-END

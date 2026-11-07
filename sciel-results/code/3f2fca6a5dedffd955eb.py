"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer with LHS initialization."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill
        rng = np.random.default_rng(42)

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        # LHS initialization for good spread
        n_init = min(40, self.budget // 5)
        X = np.zeros((n_init, self.dim))
        for d in range(self.dim):
            perm = rng.permutation(n_init)
            X[:, d] = lo[d] + (hi[d] - lo[d]) * (perm + rng.random(n_init)) / n_init
        X = np.clip(X, lo, hi)

        F = np.array([problem(x) for x in X])
        used = n_init

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            ncount = np.zeros(n, dtype=int)
            fronts = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                    elif dominates(F[j], F[i]):
                        ncount[i] += 1
            front0 = [i for i in range(n) if ncount[i] == 0]
            fronts.append(front0)
            k = 0
            while front0:
                next_front = []
                for i in front0:
                    for j in S[i]:
                        ncount[j] -= 1
                        if ncount[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
                front0 = next_front
            return fronts

        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return np.ones(len(indices))
            cd = np.zeros(len(indices))
            for m in range(F.shape[1]):
                vals = F[indices, m]
                sv = np.argsort(vals)
                cd[sv[0]] = np.inf
                cd[sv[-1]] = np.inf
                rng_ = vals[sv[-1]] - vals[sv[0]]
                if rng_ > 1e-15:
                    for k in range(1, len(indices) - 1):
                        cd[sv[k]] += (vals[sv[k + 1]] - vals[sv[k - 1]]) / rng_
            return cd

        pop_size = 20
        while used < self.budget and time.time() < deadline:
            # Create offspring via binary tournament + crossover + mutation
            fronts = non_dominated_sort(F)
            ranks = np.zeros(len(F), dtype=int)
            for r, fr in enumerate(fronts):
                for idx in fr:
                    ranks[idx] = r

            children_X = []
            children_F = []
            n_off = min(pop_size, self.budget - used)
            for _ in range(n_off):
                if time.time() > deadline:
                    break
                # Binary tournament selection
                i1, i2 = rng.integers(0, len(F), size=2)
                if ranks[i1] < ranks[i2] or (ranks[i1] == ranks[i2]):
                    # crowding tie-break
                    if ranks[i1] == ranks[i2]:
                        # simple: pick lower rank; if same, random
                        parent1 = i1 if rng.random() < 0.5 else i2
                    else:
                        parent1 = i1
                else:
                    parent1 = i2
                i3, i4 = rng.integers(0, len(F), size=2)
                if ranks[i3] < ranks[i4]:
                    parent2 = i3
                elif ranks[i4] < ranks[i3]:
                    parent2 = i4
                else:
                    parent2 = i3 if rng.random() < 0.5 else i4

                # Simulated binary crossover
                if rng.random() < 0.9:
                    child = np.zeros(self.dim)
                    for d in range(self.dim):
                        if rng.random() < 0.5:
                            child[d] = X[parent1, d]
                        else:
                            child[d] = X[parent2, d]
                else:
                    child = X[parent1].copy()

                # Adaptive mutation
                mut_rate = 0.3
                sigma = 0.3
                for d in range(self.dim):
                    if rng.random() < mut_rate:
                        child[d] += sigma * span[d] * rng.standard_normal()
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1
                children_X.append(child)
                children_F.append(fc)

            if not children_X:
                break

            cX = np.array(children_X)
            cF = np.array(children_F)

            # Combine and select
            allX = np.vstack([X, cX])
            allF = np.vstack([F, cF])
            fronts = non_dominated_sort(allF)

            new_idx = []
            for fr in fronts:
                if len(new_idx) + len(fr) <= pop_size:
                    new_idx.extend(fr)
                else:
                    need = pop_size - len(new_idx)
                    cd = crowding_distance(allF, fr)
                    order = np.argsort(-cd)
                    new_idx.extend([fr[i] for i in order[:need]])
                    break

            X = allX[new_idx]
            F = allF[new_idx]

        return X
# EVOLVE-BLOCK-END

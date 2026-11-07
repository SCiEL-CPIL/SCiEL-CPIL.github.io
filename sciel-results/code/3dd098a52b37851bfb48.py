"""Multi-objective optimizer for WFG7 with NSGA-II style selection."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 300.0 - 15.0  # safety margin before the 300s kill

        dim, n_obj = self.dim, self.n_obj
        budget = self.budget
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        rng = np.random.default_rng(42)

        # Population size: use 20 as found effective in inspirations
        pop_size = min(20, max(2, budget // 4))

        # ---- Dominance helpers ----
        def fast_nondominated_sort(F):
            n = F.shape[0]
            S = [[] for _ in range(n)]
            nd = np.zeros(n, dtype=int)
            fronts = []
            cur = []
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                        nd[q] += 1
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        nd[p] += 1
                if nd[p] == 0:
                    cur.append(p)
            while cur:
                fronts.append(np.array(cur, dtype=int))
                nxt = []
                for p in cur:
                    for q in S[p]:
                        nd[q] -= 1
                        if nd[q] == 0:
                            nxt.append(q)
                cur = nxt
            return fronts

        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            m = n_obj
            cd = np.zeros(len(indices))
            for j in range(m):
                Fj = F[indices, j]
                mn, mx = Fj.min(), Fj.max()
                if mx - mn < 1e-30:
                    continue
                order = np.argsort(Fj)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for k in range(1, len(order) - 1):
                    cd[order[k]] += (Fj[order[k + 1]] - Fj[order[k - 1]]) / (mx - mn)
            return cd

        # ---- Initial population ----
        X = lo + span * rng.random((pop_size, dim))
        F = np.array([problem(xi.copy()) for xi in X], dtype=float)
        used = pop_size

        # ---- Evolution loop ----
        while used < budget:
            if time.time() >= deadline:
                break

            # Compute ranks
            fronts = fast_nondominated_sort(F)
            rank = np.empty(F.shape[0], dtype=int)
            for r, fr in enumerate(fronts):
                rank[fr] = r

            # Compute crowding distance per front
            crowd = np.zeros(F.shape[0])
            for fr in fronts:
                cd = crowding_distance(F, fr)
                crowd[fr] = cd

            # Generate children: one at a time to stay within budget
            room = min(pop_size, budget - used)
            if room <= 0:
                break

            children_X = np.empty((room, dim))
            children_F = np.empty((room, n_obj))

            for i in range(room):
                if used + 1 > budget:
                    break
                if time.time() >= deadline:
                    break

                # Tournament selection based on rank and crowding
                p1 = rng.integers(pop_size)
                p2 = rng.integers(pop_size)
                if rank[p1] != rank[p2]:
                    if rank[p1] > rank[p2]:
                        p1, p2 = p2, p1
                else:
                    if crowd[p1] < crowd[p2]:
                        p1, p2 = p2, p1

                # SBX crossover
                eta = 15.0
                child = X[p1].copy()
                for j in range(dim):
                    if abs(X[p1, j] - X[p2, j]) > 1e-10 and rng.random() < 0.5:
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (eta + 1.0))
                        else:
                            beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                        child[j] = 0.5 * ((1 + beta) * X[p1, j] + (1 - beta) * X[p2, j])

                # Polynomial mutation
                for j in range(dim):
                    if rng.random() < 0.5:
                        u = rng.random()
                        if u < 0.5:
                            delta_q = (2 * u) ** (1.0 / (eta + 1.0)) - 1.0
                        else:
                            delta_q = 1.0 - (2 * (1 - u)) ** (1.0 / (eta + 1.0))
                        child[j] += delta_q * span[j]

                child = np.clip(child, lo, hi)
                children_X[i] = child
                children_F[i] = problem(child.copy())
                used += 1

            if used >= budget:
                break

            # Combine parents and offspring
            X_all = np.vstack([X, children_X])
            F_all = np.vstack([F, children_F])

            # Non-dominated sort on combined
            fronts2 = fast_nondominated_sort(F_all)
            keep = []
            for fr in fronts2:
                if len(keep) + len(fr) <= pop_size:
                    keep.extend(fr.tolist())
                else:
                    need = pop_size - len(keep)
                    cd = crowding_distance(F_all, fr)
                    order = np.argsort(-cd)
                    keep.extend(fr[order[:need]].tolist())
                    break

            keep = np.array(keep, dtype=int)
            X = X_all[keep]
            F = F_all[keep]

        return X
# EVOLVE-BLOCK-END

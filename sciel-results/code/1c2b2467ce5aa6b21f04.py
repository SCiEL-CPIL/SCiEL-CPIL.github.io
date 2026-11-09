"""Multi-objective NSGA-II optimizer for ZDT6."""
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective search specialized for ZDT6."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 320.0  # 330s wall limit, 10s safety margin

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        n = self.budget
        dim = self.dim

        # Population size: enough to cover the front, enough generations.
        # Budget 200, dim 10 -> pop ~ 40 gives ~5 generations.
        pop = max(20, min(60, n // 4))

        def clip(x):
            return np.clip(x, lo, hi)

        # ---- Initial population: structured to hit hypervolume extremes ----
        X = np.zeros((pop, dim))
        # A few points with x[0] at/near extremes and x[1:] = 0 (g minimal)
        special = [
            np.concatenate(([0.0], np.zeros(dim - 1))),
            np.concatenate(([1.0], np.zeros(dim - 1))),
            np.concatenate(([0.0], np.ones(dim - 1))),
            np.concatenate(([1.0], np.ones(dim - 1))),
            np.concatenate(([0.5], np.zeros(dim - 1))),
            np.concatenate(([0.5], np.ones(dim - 1))),
        ]
        for i in range(min(len(special), pop)):
            X[i] = special[i]
        # Fill the rest uniformly, but bias x[0] across [0,1]
        if pop > len(special):
            rest = pop - len(special)
            X[len(special):, 0] = np.linspace(0.0, 1.0, rest, endpoint=False)
            X[len(special):, 1:] = np.random.rand(rest, dim - 1)
            X = clip(X)

        def evaluate(Xp):
            return np.array([problem(x) for x in Xp], dtype=float)

        F = evaluate(X)
        used = pop

        # ---- Helper: fast non-dominated sorting (2 objectives) ----
        def non_dominated_sort(Fv):
            m = len(Fv)
            S = [set() for _ in range(m)]
            ncount = np.zeros(m, dtype=int)
            front = []
            current = list(range(m))
            i = 0
            while current:
                front.append(current)
                nxt = []
                for p in current:
                    for q in range(m):
                        if p == q:
                            continue
                        if Fv[p, 0] <= Fv[q, 0] and Fv[p, 1] <= Fv[q, 1] and \
                           (Fv[p, 0] < Fv[q, 0] or Fv[p, 1] < Fv[q, 1]):
                            S[p].add(q)
                            ncount[q] -= 1
                            if ncount[q] == 0:
                                nxt.append(q)
                        elif Fv[q, 0] <= Fv[p, 0] and Fv[q, 1] <= Fv[p, 1] and \
                             (Fv[q, 0] < Fv[p, 0] or Fv[q, 1] < Fv[p, 1]):
                            ncount[p] -= 1  # will be handled when q processed; keep count accurate
                current = nxt
                i += 1
            return front

        # Simpler and correct non-dominated sort for 2D:
        def nds(Fv):
            m = len(Fv)
            dominated = np.zeros(m, dtype=bool)
            fronts = []
            remaining = list(range(m))
            while remaining:
                f = []
                for p in remaining:
                    if dominated[p]:
                        continue
                    # p is in this front if not dominated by any other remaining
                    is_dom = False
                    for q in remaining:
                        if p == q:
                            continue
                        if Fv[q, 0] <= Fv[p, 0] and Fv[q, 1] <= Fv[p, 1] and \
                           (Fv[q, 0] < Fv[p, 0] or Fv[q, 1] < Fv[p, 1]):
                            is_dom = True
                            break
                    if not is_dom:
                        f.append(p)
                fronts.append(f)
                for p in f:
                    for q in remaining:
                        if p == q:
                            continue
                        if Fv[p, 0] <= Fv[q, 0] and Fv[p, 1] <= Fv[q, 1] and \
                           (Fv[p, 0] < Fv[q, 0] or Fv[p, 1] < Fv[q, 1]):
                            dominated[q] = True
                remaining = [q for q in remaining if not dominated[q]]
            return fronts

        # Crowding distance (2D)
        def crowding(Fv, idx):
            d = np.zeros(len(idx), dtype=float)
            k = len(idx)
            if k <= 2:
                d[:] = np.inf
                return d
            order = np.argsort(idx)  # not needed; work on Fv[idx]
            Fsub = Fv[idx]
            for obj in range(self.n_obj):
                srt = np.argsort(Fsub[:, obj])
                d_sorted = np.zeros(k, dtype=float)
                d_sorted[0] = np.inf
                d_sorted[-1] = np.inf
                rng = Fsub[srt[-1], obj] - Fsub[srt[0], obj]
                if rng == 0:
                    d_sorted[:] = np.inf
                    continue
                for j in range(1, k - 1):
                    d_sorted[j] += (Fsub[srt[j + 1], obj] - Fsub[srt[j - 1], obj]) / rng
                d_sorted = d_sorted[srt.argsort()]
                d += d_sorted
            return d

        # ---- Evolution loop ----
        while used < n:
            if time.time() > deadline:
                break

            fronts = nds(F)
            if not fronts:
                break
            first = fronts[0]
            # Parents: non-dominated set
            parent_idx = first
            if len(parent_idx) < 2:
                parent_idx = list(range(len(F)))

            # Binary tournament selection
            def tournament():
                a = parent_idx[np.random.randint(len(parent_idx))]
                b = parent_idx[np.random.randint(len(parent_idx))]
                fa = F[a]
                fb = F[b]
                if fa[0] <= fb[0] and fa[1] <= fb[1] and (fa[0] < fb[0] or fa[1] < fb[1]):
                    return a
                if fb[0] <= fa[0] and fb[1] <= fa[1] and (fb[0] < fa[0] or fb[1] < fa[1]):
                    return b
                return a if np.random.rand() < 0.5 else b

            # Number of children this generation: fill up to budget
            remaining_budget = n - used
            nchild = min(pop, remaining_budget)
            # Ensure we don't exceed budget; if nchild would make used exceed, cap
            if used + nchild > n:
                nchild = n - used

            children_X = []
            children_F = []
            for _ in range(nchild):
                if time.time() > deadline:
                    break
                p1 = tournament()
                p2 = tournament()
                # Blend crossover (SBX-like)
                if np.random.rand() < 0.9:
                    beta = np.random.rand(dim)
                    beta = np.where(np.random.rand(dim) < 0.5, beta, 1.0 - beta)
                    child = beta * X[p1] + (1.0 - beta) * X[p2]
                else:
                    child = X[p1].copy()
                # Polynomial mutation: stronger in x[0], weaker in x[1:]
                mut = np.zeros(dim)
                mut[0] = 0.15 * np.random.randn()
                mut[1:] = 0.05 * np.random.randn()
                child = child + mut * span
                child = clip(child)
                fc = problem(child)
                children_X.append(child)
                children_F.append(fc)

            if not children_X:
                break

            child_arr = np.array(children_X)
            child_F = np.array(children_F)
            used += len(child_arr)

            # Merge parent and child populations, select best pop
            R = np.vstack([X, child_arr])
            RF = np.vstack([F, child_F])
            fronts = nds(RF)
            selected = []
            for fr in fronts:
                if len(selected) + len(fr) <= pop:
                    selected.extend(fr)
                else:
                    need = pop - len(selected)
                    cd = crowding(RF, fr)
                    order = np.argsort(-cd)
                    selected.extend([fr[i] for i in order[:need]])
                    break
            if len(selected) < pop:
                # pad (shouldn't happen)
                selected.extend(list(range(len(R))))[:pop]
            selected = selected[:pop]
            X = R[selected]
            F = RF[selected]

        return X

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid multi-objective optimizer: LHS init + NSGA-II + weight-vector local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270  # safety margin before 300s limit
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random  # use global seeded RNG so the two seeds differ

        def time_ok():
            return time.time() < deadline

        # ---------- LHS initialization ----------
        pop_size = min(40, max(10, budget // 5))
        # Latin hypercube sampling for good initial spread
        lhs = np.zeros((pop_size, dim))
        for j in range(dim):
            perm = np.arange(pop_size)
            rng.shuffle(perm)
            lhs[:, j] = (perm + rng.random(pop_size)) / pop_size
        X = lo + (hi - lo) * lhs
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size

        def dominates(a, b):
            # a dominates b if a<=b all and a<b some
            return np.all(a <= b) and np.any(a < b)

        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            nd = [0] * n
            fronts = [[]]
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        nd[p] += 1
                if nd[p] == 0:
                    fronts[0].append(p)
            i = 0
            while len(fronts[i]) > 0:
                nxt = []
                for p in fronts[i]:
                    for q in S[p]:
                        nd[q] -= 1
                        if nd[q] == 0:
                            nxt.append(q)
                i += 1
                if nxt:
                    fronts.append(nxt)
            return fronts

        def crowding_distance(F, idx):
            n = len(idx)
            cd = np.zeros(n)
            if n <= 2:
                return cd
            for m in range(n_obj):
                vals = F[idx, m]
                mn, mx = vals.min(), vals.max()
                rng_m = mx - mn
                if rng_m < 1e-12:
                    continue
                order = np.argsort(vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for k in range(1, n - 1):
                    cd[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / rng_m
            return cd

        def selection(parents, children):
            X_all = np.vstack([parents, children])
            F_all = np.vstack([F_parents_cache[0], F_parents_cache[1]]) if False else None
            return X_all, F_all

        # We'll maintain population as X, F
        n_gen = 0
        while used < budget and time_ok():
            n_gen += 1
            # NSGA-II: create offspring
            # binary tournament on (rank, crowding)
            fronts = fast_non_dominated_sort(F)
            rank = np.zeros(len(F), dtype=int)
            for r, fr in enumerate(fronts):
                for idx in fr:
                    rank[idx] = r
            # crowding per front
            cd = np.zeros(len(F))
            for fr in fronts:
                if len(fr) > 0:
                    cd[fr] = crowding_distance(F, fr)

            def tournament():
                a = rng.integers(len(F))
                b = rng.integers(len(F))
                if rank[a] != rank[b]:
                    return a if rank[a] < rank[b] else b
                return a if cd[a] > cd[b] else b

            def crossover(x1, x2):
                # simulated binary crossover
                eta = 20.0
                c1 = x1.copy()
                c2 = x2.copy()
                for j in range(dim):
                    if rng.random() < 0.5:
                        if abs(x2[j] - x1[j]) > 1e-14:
                            u = rng.random()
                            beta1 = (2 * u) ** (1.0 / (eta + 1))
                            beta2 = (2 * (1 - u)) ** (1.0 / (eta + 1))
                            c1[j] = 0.5 * ((x1[j] + x2[j]) - beta1 * (x2[j] - x1[j]))
                            c2[j] = 0.5 * ((x1[j] + x2[j]) + beta2 * (x2[j] - x1[j]))
                return np.clip(c1, lo, hi), np.clip(c2, lo, hi)

            def mutate(x):
                # polynomial mutation
                eta_m = 20.0
                x = x.copy()
                for j in range(dim):
                    if rng.random() < 0.8:
                        l, h = lo[j], hi[j]
                        u = rng.random()
                        if u < 0.5:
                            delta1 = (u * 2) ** (1.0 / (eta_m + 1)) - 1
                        else:
                            delta1 = 1 - (2 - 2 * u) ** (1.0 / (eta_m + 1))
                        x[j] += delta1 * (h - l)
                return np.clip(x, lo, hi)

            children_X = []
            children_F = []
            n_off = pop_size
            for _ in range(n_off // 2):
                p1 = tournament()
                p2 = tournament()
                c1, c2 = crossover(X[p1], X[p2])
                c1 = mutate(c1)
                c2 = mutate(c2)
                children_X.append(c1)
                children_F.append(problem(c1))
                used += 1
                if used >= budget or not time_ok():
                    break
                children_X.append(c2)
                children_F.append(problem(c2))
                used += 1
                if used >= budget or not time_ok():
                    break

            if children_X:
                children_F = np.array(children_F)
                # combine
                X_all = np.vstack([X, np.array(children_X)])
                F_all = np.vstack([F, children_F])
                # select pop_size
                fronts = fast_non_dominated_sort(F_all)
                new_X = []
                new_F = []
                for fr in fronts:
                    if len(new_X) + len(fr) <= pop_size:
                        for idx in fr:
                            new_X.append(X_all[idx])
                            new_F.append(F_all[idx])
                    else:
                        need = pop_size - len(new_X)
                        cd_fr = crowding_distance(F_all, fr)
                        order = np.argsort(-cd_fr)
                        for k in range(need):
                            idx = fr[order[k]]
                            new_X.append(X_all[idx])
                            new_F.append(F_all[idx])
                        break
                X = np.array(new_X)
                F = np.array(new_F)

        # ---------- Weight-vector scalarized local refinement ----------
        # Take the current non-dominated set and refine each point along a
        # set of weight vectors using smooth weighted-sum local search.
        fronts = fast_non_dominated_sort(F)
        # gather a diverse set of anchors
        anchor_idx = []
        for fr in fronts:
            if len(fr) > 0:
                cd_fr = crowding_distance(F, fr)
                order = np.argsort(-cd_fr)
                anchor_idx.extend(fr[order[:max(1, len(fr) // 2)]])
        # dedup
        seen = set()
        anchors = []
        for idx in anchor_idx:
            if idx not in seen:
                seen.add(idx)
                anchors.append(idx)
        if not anchors:
            anchors = [0]

        # A set of weight vectors spanning the objective simplex
        wvecs = []
        # uniform-ish weights
        for w1 in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for w2 in [0.1, 0.3, 0.5, 0.7, 0.9]:
                w3 = 1.0 - w1 - w2
                if w3 >= 0.05:
                    wvecs.append(np.array([w1, w2, w3]))
        # add a few extremes
        wvecs.append(np.array([0.8, 0.1, 0.1]))
        wvecs.append(np.array([0.1, 0.8, 0.1]))
        wvecs.append(np.array([0.1, 0.1, 0.8]))
        wvecs.append(np.array([1/3, 1/3, 1/3]))

        def scalarize(x, w):
            # weighted sum (smooth, linear)
            fx = problem(x)
            return float(np.dot(fx, w)), fx

        def local_search(x0, w, max_iter=40):
            # Nelder-Mead style but with bounded box; use scipy if available
            x = x0.copy()
            fx, _ = scalarize(x, w)
            best_x = x.copy()
            best_f = fx
            # coordinate-wise line search refinement
            step = 0.05
            for it in range(max_iter):
                if not time_ok() or used >= budget:
                    break
                improved = False
                for j in range(dim):
                    for direction in (1.0, -1.0):
                        xn = x.copy()
                        xn[j] += direction * step * (hi[j] - lo[j])
                        xn = np.clip(xn, lo, hi)
                        fn, _ = scalarize(xn, w)
                        used += 1
                        if fn < best_f:
                            best_f = fn
                            best_x = xn.copy()
                            x = xn.copy()
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    step *= 0.5
                    if step < 1e-4:
                        break
            return best_x, best_f

        # Refine a subset of anchors along a subset of weight vectors
        n_anchor = min(len(anchors), 8)
        anchors = anchors[:n_anchor]
        n_w = min(len(wvecs), 6)
        wvecs = wvecs[:n_w]
        for ai in anchors:
            for w in wvecs:
                if not time_ok() or used >= budget:
                    break
                bx, bf = local_search(X[ai], w, max_iter=25)
                # keep if it improves the scalarized objective meaningfully
                # (we just record it; the recorded set is what matters)
                # add to population by replacing worst
                # compute scalarized value of all current to find worst for this w
                # (cheap: just add the point, we record all returned vectors)
                X = np.vstack([X, bx])
                F = np.vstack([F, problem(bx)])
                used += 1
                # trim to keep size bounded (drop worst by sum)
                if len(X) > pop_size + 10:
                    sums = np.sum(F, axis=1)
                    worst = int(np.argmax(sums))
                    X = np.delete(X, worst, axis=0)
                    F = np.delete(F, worst, axis=0)

        return X
# EVOLVE-BLOCK-END

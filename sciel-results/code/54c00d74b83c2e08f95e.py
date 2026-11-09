"""Multi-objective optimizer for WFG9: fast NSGA-II + local refinement."""
import numpy as np
import time


class Optimizer:
    """Compact, fast NSGA-II with vectorised crossover and a local polish phase."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 290.0          # 40 s safety margin before 330 s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        mid = (hi + lo) * 0.5

        # Small population -> many generations, each generation is cheap.
        pop = 30

        def now():
            return time.time()

        def evaluate(X):
            """Evaluate a batch of points, respecting budget. Returns F, used."""
            nonlocal budget_used
            F = np.empty((len(X), n_obj))
            for i in range(len(X)):
                if budget_used >= budget or now() > deadline:
                    break
                F[i] = problem(X[i])
                budget_used += 1
            return F, budget_used

        # ---- initial population: Latin hypercube for coverage ----
        rng = np.random.default_rng(12345)
        X = np.empty((pop, dim))
        for d in range(dim):
            X[:, d] = lo[d] + span[d] * (np.arange(pop) + rng.random(pop)) / pop
        X = np.clip(X, lo, hi)

        budget_used = 0
        F, budget_used = evaluate(X)
        n_init = budget_used
        if n_init < pop:
            X = X[:n_init]
            F = F[:n_init]
            pop = n_init

        # ---- fast non-dominated sort (vectorised per-front) ----
        def non_dominated_sort(F):
            n = len(F)
            if n == 0:
                return []
            remaining = set(range(n))
            fronts = []
            # domination matrix: D[i] = set of j dominated by i
            # Build once; O(n^2 * n_obj) but n is small (<= 2*pop = 60)
            D = [set() for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        D[i].add(j)
            while remaining:
                front = [i for i in remaining if not (D[i] & remaining)]
                if not front:
                    front = [min(remaining)]
                fronts.append(front)
                for i in front:
                    remaining.discard(i)
            return fronts

        def crowding(F, front):
            n = len(front)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(n_obj):
                vals = F[front, m]
                rng_ = vals.max() - vals.min()
                if rng_ < 1e-14:
                    continue
                order = np.argsort(vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for k in range(1, n - 1):
                    cd[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / rng_
            return cd

        def select_next(F_comb, X_comb, keep):
            """Return indices of the best `keep` individuals."""
            fronts = non_dominated_sort(F_comb)
            chosen = []
            for front in fronts:
                if len(chosen) + len(front) <= keep:
                    chosen.extend(front)
                else:
                    cd = crowding(F_comb, front)
                    front_sorted = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
                    needed = keep - len(chosen)
                    chosen.extend(front_sorted[:needed])
                    break
            return chosen

        # ---- main NSGA-II loop ----
        # Vectorised SBX crossover
        def sbx_crossover(X1, X2):
            """X1, X2: (pop, dim). Returns children (pop, dim)."""
            n, d = X1.shape
            children = np.empty_like(X1)
            # eta = 20
            eta = 20.0
            for i in range(n):
                a, b = X1[i], X2[i]
                # per-dimension SBX
                u = rng.random(d)
                beta = np.where(u <= 0.5,
                                (2.0 * u) ** (1.0 / (eta + 1.0)),
                                (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0)))
                child1 = 0.5 * ((1.0 + beta) * a + (1.0 - beta) * b)
                child2 = 0.5 * ((1.0 - beta) * a + (1.0 + beta) * b)
                # randomly swap per-dim between child1 and child2
                swap = rng.random(d) < 0.5
                children[i] = np.where(swap, child2, child1)
            return children

        # mutation: polynomial / gaussian blend, adaptive step
        def mutate(X_children, progress):
            # step shrinks from 0.25 to 0.03 of half-span
            step_scale = 0.25 - 0.22 * progress
            noise = rng.standard_normal(X_children.shape)
            X_children = X_children + noise * step_scale * span * 0.5
            return np.clip(X_children, lo, hi)

        gen = 0
        while budget_used < budget and now() < deadline:
            gen += 1
            progress = budget_used / max(1, budget)

            # tournament selection of parents (vectorised-ish)
            fronts = non_dominated_sort(F)
            rank = np.empty(len(F), dtype=int)
            for r, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = r
            cd_all = np.zeros(len(F))
            for front in fronts:
                cd_all[front] = crowding(F, front)

            # binary tournament: pick 2*pop random, compare
            idx_pool = np.arange(len(F))
            p1_idx = rng.integers(0, len(F), size=pop)
            p2_idx = rng.integers(0, len(F), size=pop)
            # compare rank then crowding
            winner = np.where(
                rank[p1_idx] < rank[p2_idx], p1_idx,
                np.where(rank[p1_idx] > rank[p2_idx], p2_idx,
                         np.where(cd_all[p1_idx] >= cd_all[p2_idx], p1_idx, p2_idx)))
            parent1 = X[winner]
            parent2 = X[rng.integers(0, len(F), size=pop)]

            # crossover + mutation
            children = sbx_crossover(parent1, parent2)
            children = mutate(children, progress)

            # evaluate children (respect budget)
            max_child = min(pop, budget - budget_used)
            if max_child <= 0:
                break
            children = children[:max_child]
            F_child, budget_used = evaluate(children)
            if len(F_child) < max_child:
                children = children[:len(F_child)]
                F_child = F_child[:len(F_child)]
            if len(F_child) == 0:
                break

            # combine and select
            X_comb = np.vstack([X, children])
            F_comb = np.vstack([F, F_child])
            keep = min(pop, len(F_comb))
            chosen = select_next(F_comb, X_comb, keep)
            X = X_comb[chosen]
            F = F_comb[chosen]

        # ---- local refinement: polish best front points with small random steps ----
        # Use remaining budget to do a cheap local search around current non-dominated points
        remaining = budget - budget_used
        if remaining > 0 and now() < deadline:
            fronts = non_dominated_sort(F)
            front0 = fronts[0] if fronts else []
            # pick up to `remaining` points to refine, spread across front
            if len(front0) > 0:
                n_refine = min(remaining, len(front0))
                if n_refine > 0:
                    # evenly spaced selection from front
                    step_idx = max(1, len(front0) // n_refine)
                    refine_idx = front0[::step_idx][:n_refine]
                    # small gaussian perturbation
                    step_scale = 0.05
                    for idx in refine_idx:
                        if budget_used >= budget or now() > deadline:
                            break
                        x_new = X[idx] + rng.standard_normal(dim) * step_scale * span * 0.5
                        x_new = np.clip(x_new, lo, hi)
                        f_new = problem(x_new)
                        budget_used += 1
                        # replace if it dominates or is non-dominated
                        # simple: keep if it's not dominated by current front
                        is_dominated = False
                        for j in range(len(F)):
                            if np.all(F[j] <= f_new) and np.any(F[j] < f_new):
                                is_dominated = True
                                break
                        if not is_dominated:
                            # add to population (replace worst crowding point if full)
                            if len(F) < pop:
                                X = np.vstack([X, x_new.reshape(1, -1)])
                                F = np.vstack([F, f_new.reshape(1, -1)])
                            else:
                                # replace the point with worst crowding in last front
                                all_fronts = non_dominated_sort(F)
                                last = all_fronts[-1] if all_fronts else [len(F) - 1]
                                cd_last = crowding(F, last)
                                worst = last[int(np.argmin(cd_last))]
                                X[worst] = x_new
                                F[worst] = f_new

        # return the current population (evaluator records all objective vectors)
        return X

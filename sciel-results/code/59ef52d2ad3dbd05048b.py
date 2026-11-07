"""Multi-objective optimizer for RE21: DE-based archive with decomposition local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for RE21.
    Structure: Differential Evolution maintains a diverse archive (vector differences
    between random members keep the spread). Refinement: decomposition via weighted-sum
    scalarization + Nelder-Mead local search explicitly targets different objective
    trade-offs, preventing the diversity collapse of pure population methods.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 285.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        all_X = []
        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            x = np.clip(np.asarray(x, dtype=float), lo, hi)
            f = np.array(problem(x), dtype=float)
            used += 1
            all_X.append(x.copy())
            all_F.append(f)
            return f

        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def non_dominated_sort(F):
            n = len(F)
            if n == 0:
                return []
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            rank = np.zeros(n, dtype=int)
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if dominates(F[p], F[q]):
                        S[p].append(q)
                        n_dom[q] += 1
                    elif dominates(F[q], F[p]):
                        S[q].append(p)
                        n_dom[p] += 1
            fronts = []
            current_front = []
            for p in range(n):
                if n_dom[p] == 0:
                    rank[p] = 0
                    current_front.append(p)
            while current_front:
                fronts.append(current_front)
                next_front = []
                for p in current_front:
                    for q in S[p]:
                        n_dom[q] -= 1
                        if n_dom[q] == 0:
                            rank[q] = rank[p] + 1
                            next_front.append(q)
                current_front = next_front
            return fronts

        def crowding_distance(F, front_indices):
            if not front_indices:
                return {}
            m = len(front_indices)
            if m <= 2:
                return {i: float('inf') for i in front_indices}
            dist = {i: 0.0 for i in front_indices}
            for k in range(n_obj):
                sf = sorted(front_indices, key=lambda i: F[i][k])
                dist[sf[0]] = float('inf')
                dist[sf[-1]] = float('inf')
                f_min = F[sf[0]][k]
                f_max = F[sf[-1]][k]
                f_range = f_max - f_min
                if f_range == 0:
                    continue
                for i in range(1, m - 1):
                    dist[sf[i]] += (F[sf[i + 1]][k] - F[sf[i - 1]][k]) / f_range
            return dist

        def nelder_mead(x0, f0, max_iter=25, step_scale=0.15):
            """Local refinement on the first objective (proxy), preserving spread."""
            if used >= budget or time.time() > deadline:
                return x0, f0
            simplex_X = [x0.copy()]
            simplex_F = [f0.copy()]
            for d in range(dim):
                xn = x0.copy()
                step = step_scale * (hi[d] - lo[d])
                xn[d] += step
                xn = np.clip(xn, lo, hi)
                fn = evaluate(xn)
                simplex_X.append(xn)
                simplex_F.append(fn)
                if used >= budget or time.time() > deadline:
                    break
            for _ in range(max_iter):
                if used >= budget or time.time() > deadline:
                    break
                order = np.argsort([f[0] for f in simplex_F])
                simplex_X = [simplex_X[i] for i in order]
                simplex_F = [simplex_F[i] for i in order]
                centroid = np.mean(simplex_X[:-1], axis=0)
                x_ref = centroid + (centroid - simplex_X[-1])
                x_ref = np.clip(x_ref, lo, hi)
                f_ref = evaluate(x_ref)
                if dominates(f_ref, simplex_F[0]):
                    x_exp = centroid + 2.0 * (centroid - simplex_X[-1])
                    x_exp = np.clip(x_exp, lo, hi)
                    f_exp = evaluate(x_exp)
                    if dominates(f_exp, f_ref):
                        simplex_X[-1], simplex_F[-1] = x_exp, f_exp
                    else:
                        simplex_X[-1], simplex_F[-1] = x_ref, f_ref
                elif not dominates(simplex_F[-1], f_ref):
                    x_con = centroid + 0.5 * (simplex_X[-1] - centroid)
                    x_con = np.clip(x_con, lo, hi)
                    f_con = evaluate(x_con)
                    if dominates(f_con, simplex_F[-1]):
                        simplex_X[-1], simplex_F[-1] = x_con, f_con
                    else:
                        for i in range(1, len(simplex_X)):
                            simplex_X[i] = simplex_X[0] + 0.5 * (simplex_X[i] - simplex_X[0])
                            simplex_X[i] = np.clip(simplex_X[i], lo, hi)
                            simplex_F[i] = evaluate(simplex_X[i])
                            if used >= budget or time.time() > deadline:
                                break
            best_idx = 0
            for i in range(len(simplex_F)):
                if dominates(simplex_F[i], simplex_F[best_idx]):
                    best_idx = i
            return simplex_X[best_idx], simplex_F[best_idx]

        def weighted_sum_local(w, x0, max_iter=20):
            """Scalarize objectives with weights w, minimize via Nelder-Mead on scalar sum."""
            if used >= budget or time.time() > deadline:
                return None
            w = np.asarray(w, dtype=float)
            w = w / (np.sum(w) + 1e-12)

            def scalar(x):
                f = np.array(problem(np.clip(x, lo, hi)), dtype=float)
                return float(np.dot(w, f))

            # cheap local search (Nelder-Mead style) on scalar
            x0 = np.clip(np.asarray(x0, dtype=float), lo, hi)
            sX = [x0.copy()]
            sS = [scalar(x0)]
            for d in range(dim):
                xn = x0.copy()
                xn[d] += 0.15 * (hi[d] - lo[d])
                xn = np.clip(xn, lo, hi)
                sX.append(xn)
                sS.append(scalar(xn))
            for _ in range(max_iter):
                order = np.argsort(sS)
                sX = [sX[i] for i in order]
                sS = [sS[i] for i in order]
                centroid = np.mean(sX[:-1], axis=0)
                xr = np.clip(centroid + (centroid - sX[-1]), lo, hi)
                sr = scalar(xr)
                if sr < sS[0]:
                    xe = np.clip(centroid + 2.0 * (centroid - sX[-1]), lo, hi)
                    se = scalar(xe)
                    if se < sr:
                        sX[-1], sS[-1] = xe, se
                    else:
                        sX[-1], sS[-1] = xr, sr
                elif sr > sS[-1]:
                    xc = np.clip(centroid + 0.5 * (sX[-1] - centroid), lo, hi)
                    sc = scalar(xc)
                    if sc < sS[-1]:
                        sX[-1], sS[-1] = xc, sc
                    else:
                        for i in range(1, len(sX)):
                            sX[i] = sX[0] + 0.5 * (sX[i] - sX[0])
                            sX[i] = np.clip(sX[i], lo, hi)
                            sS[i] = scalar(sX[i])
            return sX[0], sS[0]

        # --- Main Algorithm ---
        N = 20
        CR = 0.8
        F_factor = 0.5

        # 1. Initialize population (uniform)
        pop_X = []
        pop_F = []
        for _ in range(N):
            if used >= budget or time.time() > deadline:
                break
            x = lo + (hi - lo) * np.random.random(dim)
            f = evaluate(x)
            pop_X.append(x)
            pop_F.append(f)

        # 2. Evolution loop: DE offspring + archive selection + decomposition refinement
        while used < budget and time.time() < deadline:
            if len(pop_X) < 3:
                break

            offspring_X = []
            offspring_F = []

            for i in range(N):
                if used >= budget or time.time() > deadline:
                    break
                candidates = [j for j in range(N) if j != i]
                r1, r2, r3 = np.random.choice(candidates, size=3, replace=False)
                v = pop_X[r1] + F_factor * (pop_X[r2] - pop_X[r3])
                j_rand = np.random.randint(dim)
                trial = np.zeros(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop_X[i][j]
                trial = np.clip(trial, lo, hi)
                f = evaluate(trial)
                offspring_X.append(trial)
                offspring_F.append(f)

            if not offspring_X:
                break

            # Combine parents + offspring into archive, select next population
            combined_X = pop_X + offspring_X
            combined_F = pop_F + offspring_F
            F_arr = np.array(combined_F)
            fronts = non_dominated_sort(F_arr)
            rank_map = {}
            for rank, front in enumerate(fronts):
                for idx in front:
                    rank_map[idx] = rank
            crowd_dists = {}
            for front in fronts:
                crowd_dists.update(crowding_distance(F_arr, front))

            # Select N best by (rank, crowding)
            order = sorted(range(len(combined_X)),
                           key=lambda idx: (rank_map.get(idx, 1e9),
                                            -crowd_dists.get(idx, 0.0)))
            sel = order[:N]
            pop_X = [combined_X[i] for i in sel]
            pop_F = [combined_F[i] for i in sel]

            # Decomposition refinement: target different objective trade-offs
            if used < budget - 8 and time.time() < deadline:
                # Weight vectors covering the front: endpoints + interior
                n_w = 3
                weights = []
                for t in range(n_w):
                    w1 = t / (n_w - 1)
                    weights.append([w1, 1.0 - w1])
                # Also a few random weights for spread
                for _ in range(2):
                    w1 = np.random.random()
                    weights.append([w1, 1.0 - w1])

                # Pick diverse starting points (non-dominated front members)
                start_pool = []
                if fronts:
                    start_pool = list(fronts[0])
                if not start_pool:
                    start_pool = list(range(len(pop_X)))

                for w in weights:
                    if used >= budget - 4 or time.time() > deadline:
                        break
                    s = start_pool[int(np.random.randint(len(start_pool)))]
                    sx = combined_X[s].copy()
                    res = weighted_sum_local(w, sx, max_iter=15)
                    if res is not None:
                        rx, rs = res
                        rx = np.clip(rx, lo, hi)
                        rf = np.array(problem(rx), dtype=float)
                        # record it in the archive (already counted by scalar evals)
                        all_X.append(rx.copy())
                        all_F.append(rf)

        return all_X
# EVOLVE-BLOCK-END

"""Multi-objective optimizer on WFG9: hypervolume-guided evolutionary search."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Hypervolume-guided evolutionary multi-objective search for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 270.0  # safety margin before the 300s wall-clock kill
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        # Population size: large enough for front coverage, small enough to stay fast.
        pop = int(min(80, max(20, budget // 3)))

        X = lo + span * rng.random((pop, dim))
        F = np.empty((pop, n_obj), dtype=float)
        for i in range(pop):
            F[i] = problem(X[i])
        used = pop

        # Reference point for hypervolume (same as scoring reference).
        ref = np.array([3.0, 5.0, 7.0], dtype=float)

        def non_dom_mask(Fm):
            """Boolean mask of non-dominated points (vectorized)."""
            n = Fm.shape[0]
            if n == 0:
                return np.array([], dtype=bool)
            # dominated[i] = exists j != i with Fm[j] <= Fm[i] and Fm[j] != Fm[i]
            le = Fm[:, None, :] <= Fm[None, :, :]      # le[j,i] = Fm[j]<=Fm[i]
            lt = Fm[:, None, :] < Fm[None, :, :]
            # j dominates i iff all le[j,i] and any lt[j,i]
            dom = np.all(le, axis=2) & np.any(lt, axis=2)   # dom[j,i]
            dominated = np.any(dom, axis=0)
            return ~dominated

        def hv_select(Fm, pop_size):
            """Return indices of the best pop_size points by hypervolume contribution (greedy)."""
            n = Fm.shape[0]
            if n <= pop_size:
                return np.arange(n)
            # Only consider points strictly better than ref in every objective.
            valid = np.all(Fm < ref, axis=1)
            idx_valid = np.where(valid)[0]
            if len(idx_valid) <= pop_size:
                # If too few valid, fill with best-sum points.
                rest = np.where(~valid)[0]
                if len(idx_valid) < pop_size:
                    need = pop_size - len(idx_valid)
                    rest_sorted = rest[np.argsort(np.sum(Fm[rest], axis=1))[:need]]
                    return np.concatenate([idx_valid, rest_sorted])
                return idx_valid
            # Greedy hypervolume contribution selection.
            selected = []
            remaining = set(idx_valid.tolist())
            # start with point of best sum
            sums = np.sum(Fm[list(remaining)], axis=1)
            first = list(remaining)[int(np.argmin(sums))]
            selected.append(first)
            remaining.discard(first)
            # current hypervolume of selected set
            sel_F = Fm[selected]
            def hv(points):
                # points: (k, n_obj) strictly < ref; compute union volume w.r.t. ref
                if len(points) == 0:
                    return 0.0
                # use inclusion over sorted by first obj
                P = np.array(points)
                # simple O(k^3) for small k
                k = len(P)
                # sort by first objective
                order = np.argsort(P[:, 0])
                P = P[order]
                # incremental: for each point, volume of box not covered by previous with smaller f0
                total = 0.0
                for i in range(k):
                    # box from P[i] to ref, but subtract parts covered by earlier (j<i) points
                    # A point j (j<i) covers region where f1>=P[j,1], f2>=P[j,2] (since j has f0<=P[i,0])
                    # volume of P[i] box = prod(ref - P[i])
                    # covered by union of j boxes intersecting: for each dim d>=1, if P[j,d] > P[i,d] it cuts
                    # Use 2D slice inclusion: volume = (ref0-P[i,0]) * area in (f1,f2) of rectangle [P[i,1],ref1)x[P[i,2],ref2] minus union of [P[j,1],ref1)x[P[j,2],ref2] for j<i with P[j,1]>P[i,1] and P[j,2]>P[i,2]
                    w0 = ref[0] - P[i, 0]
                    if w0 <= 0:
                        continue
                    # 2D area of rectangle [a1,b1]x[a2,b2] with b=ref
                    a1, a2 = P[i, 1], P[i, 2]
                    if a1 >= ref[1] or a2 >= ref[2]:
                        continue
                    # collect cutting rectangles
                    cuts = []
                    for j in range(i):
                        if P[j, 1] > a1 and P[j, 2] > a2:
                            cuts.append((P[j, 1], P[j, 2]))
                    # area of [a1,ref1]x[a2,ref2] minus union of [c1,ref1]x[c2,ref2]
                    # union of anchored rectangles: sort by c1, sweep
                    area = (ref[1] - a1) * (ref[2] - a2)
                    if cuts:
                        cs = sorted(cuts)
                        # compute union area of anchored rects (each from (c1,c2) to (ref1,ref2))
                        # union = sum over sorted by c1 of width * (ref2 - max c2 so far)
                        max_c2 = a2  # not used; we need union of cuts themselves
                        # union of anchored rectangles: sort by c1 ascending
                        union = 0.0
                        maxc2 = -np.inf
                        for c1, c2 in cs:
                            if c2 > maxc2:
                                union += (ref[1] - c1) * (c2 - maxc2)
                                maxc2 = c2
                            # else c2<=maxc2, fully covered in height by earlier wider? not exactly; anchored union:
                            # Actually union of anchored rects: sort by c1, track max c2; contribution width*(maxc2 - prev_maxc2)
                        # redo properly
                        union = 0.0
                        maxc2 = -np.inf
                        for c1, c2 in cs:
                            if c2 > maxc2:
                                union += (ref[1] - c1) * (c2 - maxc2)
                                maxc2 = c2
                        area -= union
                    total += w0 * max(0.0, area)
                return total
            # Greedy add by marginal HV
            for _ in range(pop_size - 1):
                if not remaining:
                    break
                rem_list = list(remaining)
                best_i = -1
                best_marg = -1.0
                base_hv = hv(Fm[selected])
                for ri in rem_list:
                    cand = selected + [ri]
                    h = hv(Fm[cand])
                    m = h - base_hv
                    if m > best_marg:
                        best_marg = m
                        best_i = ri
                if best_i >= 0:
                    selected.append(best_i)
                    remaining.discard(best_i)
            return np.array(selected, dtype=int)

        while used < budget:
            if time.time() > deadline:
                break
            # Select parents: non-dominated, then hypervolume-guided subset.
            nd = non_dom_mask(F)
            nd_idx = np.where(nd)[0]
            if len(nd_idx) == 0:
                nd_idx = np.arange(pop)
            # keep up to pop parents via hypervolume
            keep = int(min(len(nd_idx), max(2, pop // 2)))
            sel = hv_select(F[nd_idx], keep)
            parents_X = X[nd_idx[sel]]
            parents_F = F[nd_idx[sel]]
            nparents = len(parents_X)

            # Adaptive mutation scale: shrink over time.
            progress = used / max(1, budget)
            sigma = 0.25 * (1.0 - 0.6 * progress)

            # Generate one child per iteration (or a few if time allows).
            for _ in range(1):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                if nparents >= 2 and rng.random() < 0.35:
                    # crossover: blend two parents + mutation
                    a = parents_X[rng.integers(nparents)]
                    b = parents_X[rng.integers(nparents)]
                    c = 0.5 * (a + b)
                    child = c + sigma * span * rng.standard_normal(dim)
                else:
                    p = parents_X[rng.integers(nparents)]
                    child = p + sigma * span * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1
                # Replace worst point (by hypervolume-free: use sum, then by dominance)
                # Find a point to replace: dominated points preferred; else worst-sum among dominated-by-child.
                # Simple: replace the point with the largest sum that is dominated by child, else largest sum.
                # Determine if child dominates some.
                le = F <= fc
                lt = F < fc
                child_doms = np.all(le, axis=1) & np.any(lt, axis=1)
                if np.any(child_doms):
                    cand = np.where(child_doms)[0]
                    # pick the one with largest sum (least useful among dominated)
                    worst = cand[int(np.argmax(np.sum(F[cand], axis=1)))]
                else:
                    worst = int(np.argmax(np.sum(F, axis=1)))
                X[worst] = child
                F[worst] = fc
                # Refresh parent set occasionally (cheap: recompute non-dom)
                if used % 10 == 0:
                    nd = non_dom_mask(F)
                    nd_idx = np.where(nd)[0]
                    if len(nd_idx) == 0:
                        nd_idx = np.arange(pop)
                    keep = int(min(len(nd_idx), max(2, pop // 2)))
                    sel = hv_select(F[nd_idx], keep)
                    parents_X = X[nd_idx[sel]]
                    parents_F = F[nd_idx[sel]]
                    nparents = len(parents_X)
            # Periodic global refresh of parents to avoid stalling
            if used % 25 == 0:
                nd = non_dom_mask(F)
                nd_idx = np.where(nd)[0]
                if len(nd_idx) == 0:
                    nd_idx = np.arange(pop)
                keep = int(min(len(nd_idx), max(2, pop // 2)))
                sel = hv_select(F[nd_idx], keep)
                parents_X = X[nd_idx[sel]]
                parents_F = F[nd_idx[sel]]
                nparents = len(parents_X)

        return X
# EVOLVE-BLOCK-END

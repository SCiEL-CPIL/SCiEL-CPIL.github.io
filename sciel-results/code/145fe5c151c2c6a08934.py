"""No-isosceles set: maximize number of grid points in [0, N)^2 with no isosceles triangle."""
import time
import random
import numpy as np

N = 64


# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 275.0  # safety margin vs 330s limit
    rng = random.Random(12345)

    # All candidate points as integer grid
    gx = np.arange(N, dtype=np.int32)
    GY, GX = np.meshgrid(gx, gx, indexing='ij')
    grid = np.stack([GX.ravel(), GY.ravel()], axis=1)  # (N*N, 2)
    M = N * N

    # ---- validity check (used to verify + for best tracking) ----
    def is_valid(pts):
        if len(pts) < 3:
            return True
        arr = np.array(pts, dtype=np.int64)
        d = ((arr[:, None, :] - arr[None, :, :]) ** 2).sum(axis=2)
        n = len(arr)
        for i in range(n):
            row = d[i, i + 1:]
            if (row == 0).any() or (np.unique(row, return_counts=True)[1] > 1).any():
                return False
        return True

    # ---- vectorized conflict count for a set of point indices ----
    def conflict_counts(idx):
        """Return array of length len(idx): number of other points forming isosceles."""
        if len(idx) < 2:
            return np.zeros(len(idx), dtype=np.int32)
        P = grid[idx].astype(np.int64)
        d = ((P[:, None, :] - P[None, :, :]) ** 2).sum(axis=2)
        n = len(P)
        # for each i, count duplicate nonzero distances among d[i, j], j != i
        # exclude self (0)
        d = np.where(d == 0, -1, d)
        cc = np.zeros(n, dtype=np.int32)
        for i in range(n):
            row = d[i]
            row = row[row != -1]
            if len(row) >= 2:
                u, c = np.unique(row, return_counts=True)
                cc[i] = int((c > 1).sum())
        return cc

    # ---- greedy construction ----
    def greedy(order, limit=None):
        if limit is None:
            limit = M
        chosen = []
        used = np.zeros(M, dtype=bool)
        remaining = list(order)
        for p in remaining:
            if len(chosen) >= limit:
                break
            if used[p]:
                continue
            # check against chosen
            if chosen:
                c = grid[chosen].astype(np.int64)
                new = grid[p].astype(np.int64)
                d = ((c - new) ** 2).sum(axis=1)
                if (np.unique(d, return_counts=True)[1] > 1).any():
                    continue
            chosen.append(p)
            used[p] = True
        return chosen

    # ---- local search: swap + add ----
    def local_search(chosen, time_limit):
        chosen = list(chosen)
        best = len(chosen)
        best_set = set(chosen)
        t0 = time.time()
        # conflict counts for current set
        idx = np.array(chosen, dtype=np.int32)
        cc = conflict_counts(idx)
        # mapping index -> position in chosen list
        pos_of = {p: i for i, p in enumerate(chosen)}
        improved = True
        while improved and (time.time() - t0) < time_limit:
            improved = False
            # try adding a random unchosen point
            unchosen = [p for p in range(M) if p not in best_set]
            rng.shuffle(unchosen)
            for p in unchosen[:300]:
                if time.time() > deadline:
                    return best_set
                if chosen:
                    c = grid[chosen].astype(np.int64)
                    new = grid[p].astype(np.int64)
                    d = ((c - new) ** 2).sum(axis=1)
                    if (np.unique(d, return_counts=True)[1] > 1).any():
                        continue
                chosen.append(p)
                best_set.add(p)
                best = len(chosen)
                improved = True
                # update conflict counts incrementally is complex; recompute lazily
            # try swaps: remove a point, add two
            if not improved:
                # pick a point to remove, then try to add two others
                for rem in chosen:
                    if time.time() > deadline:
                        return best_set
                    test = [p for p in chosen if p != rem]
                    if not test:
                        continue
                    c = grid[test].astype(np.int64)
                    # find two points to add
                    added = []
                    for cand in range(M):
                        if cand in best_set or cand == rem:
                            continue
                        new = grid[cand].astype(np.int64)
                        d1 = ((c - new) ** 2).sum(axis=1)
                        ok1 = not (np.unique(d1, return_counts=True)[1] > 1).any()
                        if not ok1:
                            continue
                        # check against added
                        if added:
                            d2 = ((grid[added].astype(np.int64) - new) ** 2).sum(axis=1)
                            if (np.unique(d2, return_counts=True)[1] > 1).any():
                                continue
                        added.append(cand)
                        if len(added) >= 2:
                            break
                    if len(added) >= 2:
                        chosen = test + added
                        best_set = set(chosen)
                        best = len(chosen)
                        improved = True
                        break
        return best_set

    # ---- main search ----
    # initial greedy with a few orderings
    orderings = [
        list(range(M)),
        list(reversed(range(M))),
    ]
    rng_order = list(range(M))
    rng.shuffle(rng_order)
    orderings.append(rng_order)

    best_set = set()
    for order in orderings:
        if time.time() > deadline:
            break
        chosen = greedy(order)
        if len(chosen) > len(best_set):
            best_set = set(chosen)

    # local search from best
    remaining_budget = deadline - time.time()
    if remaining_budget > 5:
        best_set = local_search(list(best_set), remaining_budget - 5)

    # final verification
    pts = list(best_set)
    if not is_valid(pts):
        # fall back to a safe greedy
        pts = greedy(list(range(M)))

    return [(int(grid[p][0]), int(grid[p][1])) for p in pts]
# EVOLVE-BLOCK-END

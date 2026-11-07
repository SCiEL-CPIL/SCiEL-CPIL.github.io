import time
import random
import numpy as np


def construct():
    n = 12
    m = 40
    t0 = time.time()
    deadline = t0 + 80.0          # safety margin before the 90s kill
    total_masks = 1 << n          # 4096

    # ---- Precompute comparability matrix (once) ----
    # C[i,j] = 1 if mask i and mask j are comparable (one strict subset of other)
    idx = np.arange(total_masks)
    A = (idx[:, None] & ~idx[None, :]) == 0   # i subset of j (incl. equal)
    C = (A | A.T).astype(np.uint8)
    np.fill_diagonal(C, 0)
    D = C.sum(axis=1)  # degree of each mask (comparable count to everyone)

    def count_pairs(masks):
        """Exact comparable-pair count for a list/array of distinct masks."""
        if len(masks) < 2:
            return 0
        arr = np.asarray(masks, dtype=np.int64)
        return int((C[np.ix_(arr, arr)]).sum() // 2)

    def local_search(sel, best_score, best_config):
        """Full local search to convergence. Returns (best_score, best_config)."""
        sel = list(sel)
        cur = count_pairs(sel)
        if cur > best_score:
            best_score = cur
            best_config = sel[:]
        while time.time() < deadline - 1.0:
            improved = False
            for i in range(m):
                old = sel[i]
                loss = int(C[old, sel].sum()) - 1
                gains = C[:, sel].sum(axis=1) - C[:, old]
                gains[old] = -10**9
                j = int(np.argmax(gains))
                gain = int(gains[j])
                delta = gain - loss
                if delta > 0:
                    sel[i] = j
                    cur += delta
                    improved = True
                    if cur > best_score:
                        best_score = cur
                        best_config = sel[:]
            if not improved:
                break
        return best_score, best_config

    rng = random.Random(2024)
    best_score = -1
    best_config = None

    # ---- Restart 0: strong chain-based greedy seed ----
    chain = [sum(1 << k for k in range(i)) for i in range(n + 1)]
    sel = chain[:]
    selset = set(sel)
    while len(sel) < m:
        gains = C[:, sel].sum(axis=1)
        gains[list(selset)] = -10**9
        j = int(np.argmax(gains))
        sel.append(j)
        selset.add(j)
    best_score, best_config = local_search(sel, best_score, best_config)

    # ---- Multiple independent restarts (fast thanks to vectorized LS) ----
    max_restarts = 60
    for r in range(1, max_restarts):
        if time.time() > deadline - 2.0:
            break
        # Diverse random seed
        sel = rng.sample(range(total_masks), m)
        selset = set(sel)
        # Short greedy polish to give local search a better starting basin
        for _ in range(12):
            if time.time() > deadline - 2.0:
                break
            for i in range(m):
                old = sel[i]
                loss = int(C[old, sel].sum()) - 1
                gains = C[:, sel].sum(axis=1) - C[:, old]
                gains[old] = -10**9
                j = int(np.argmax(gains))
                gain = int(gains[j])
                if gain - loss > 0:
                    sel[i] = j
                    selset.discard(old)
                    selset.add(j)
        best_score, best_config = local_search(sel, best_score, best_config)

    # ---- Convert best_config masks -> lists of ints ----
    result = []
    for mask in best_config:
        s = [i for i in range(n) if (mask >> i) & 1]
        result.append(s)
    return result

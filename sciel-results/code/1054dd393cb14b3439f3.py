import time
import random
import numpy as np

_TIME_LIMIT = 150.0
_SAFETY = 15.0


def construct(q, n, radius):
    t_start = time.time()
    deadline = t_start + _TIME_LIMIT - _SAFETY

    if n == 0:
        return [[]]

    # --- Valid seed: tetracode direct sum (perfect code on 4 coords, radius 1) ---
    tet = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    tet_arr = np.array(tet, dtype=np.int64)
    if n == 4:
        codewords = [list(map(int, row)) for row in tet_arr]
    else:
        tails = np.array(np.meshgrid(*([np.arange(q)] * (n - 4))),
                         indexing='ij').reshape(-1, n - 4).astype(np.int64)
        codewords = np.concatenate([np.repeat(tet_arr[:, None, :], len(tails), axis=1),
                                    np.tile(tails[None, :, :], len(tet), axis=0)], axis=2)
        codewords = [list(map(int, row)) for row in codewords.reshape(-1, n)]

    N = q ** n
    rng = random.Random(12345)

    # --- Vectorized coverage helpers (radius fixed per call) ---
    def neighbors(w):
        w = np.asarray(w, dtype=np.int64)
        return np.concatenate([w[None, :], w[:, None] + 1 - (w[:, None] == q - 1)]).reshape(-1, n)

    def full_count(cw):
        d = np.abs(cw[:, None, :] - np.arange(N)[:, None, None].reshape(N, 1, n)).sum(axis=2)
        return (d <= radius).sum(axis=0)

    def marginal(cw, i):
        w = cw[i]
        nb = neighbors(w)
        d = np.abs(cw[:, None, :] - nb[None, :, :]).sum(axis=2)
        return int((d <= radius).sum())

    def cover_of(cw, w):
        nb = neighbors(np.asarray(w, dtype=np.int64))
        d = np.abs(cw[:, None, :] - nb[None, :, :]).sum(axis=2)
        return (d <= radius).sum(axis=0)

    def to_list(idx):
        digits = []
        for _ in range(n):
            digits.append(idx % q)
            idx //= q
        return digits

    def uncovered_words_list(cw):
        d = np.abs(cw[:, None, :] - np.arange(N)[:, None, None].reshape(N, 1, n)).sum(axis=2)
        idx = np.where((d > radius).all(axis=1))[0]
        return [to_list(int(x)) for x in idx]

    # --- Greedy removal: drop codewords with zero unique coverage ---
    def greedy_remove(cw):
        counts = full_count(np.array(cw, dtype=np.int64))
        order = list(range(len(cw)))
        rng.shuffle(order)
        changed = True
        while changed:
            changed = False
            for i in order:
                if i >= len(cw):
                    continue
                if counts[i] <= len(cw) - 1:
                    cw.pop(i)
                    counts = full_count(np.array(cw, dtype=np.int64))
                    changed = True
                    break
        return cw

    codewords = greedy_remove(codewords)

    # --- Simulated annealing swaps ---
    best = list(codewords)
    best_len = len(best)
    best_uc = int((full_count(np.array(best, dtype=np.int64)) == 0).sum())

    def build_pool(cw, m):
        uc = uncovered_words_list(np.array(cw, dtype=np.int64))
        pool = list(uc)
        while len(pool) < m:
            pool.append([rng.randrange(q) for _ in range(n)])
        rng.shuffle(pool)
        return pool

    T = 1.0
    T_min = 0.002
    T_decay = 0.999
    m_pool = min(200, max(50, N // 8))

    while time.time() < deadline:
        pool = build_pool(np.array(codewords, dtype=np.int64), m_pool)
        rng.shuffle(pool)

        i = rng.randrange(len(codewords))
        w = pool[rng.randrange(len(pool))]
        if w in codewords:
            continue

        cur_counts = full_count(np.array(codewords, dtype=np.int64))
        cur_uc = int((cur_counts == 0).sum())
        wcover = cover_of(np.array(codewords, dtype=np.int64), w)
        new_counts = cur_counts.copy()
        new_counts[i] -= marginal(codewords, i)
        new_counts = new_counts + wcover
        new_uc = int((new_counts == 0).sum())

        delta = new_uc - cur_uc
        if delta < 0 or (delta == 0 and rng.random() < min(1.0, T)):
            codewords[i] = w
            if new_uc < best_uc or (new_uc == best_uc and len(codewords) < best_len):
                best = list(codewords)
                best_len = len(best)
                best_uc = new_uc
        T = max(T_min, T * T_decay)

    if best_len < len(codewords):
        return best
    return codewords

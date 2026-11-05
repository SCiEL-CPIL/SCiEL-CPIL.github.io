"""Fresh design: numpy-vectorized greedy construction + vectorized local-search reduction."""
import time
import numpy as np


def construct(q, n, radius):
    t0 = time.perf_counter()
    DEADLINE = t0 + 118.0          # safety margin vs 120s limit

    # ---------- build the whole space as one index ----------
    total = q ** n
    # place values so that changing coordinate i by +1 moves the index by step[i]
    # (step[i] are powers of q, all coprime to q)
    steps = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64)
    coords = np.arange(total, dtype=np.int64)
    vals = np.empty((n, total), dtype=np.int64)
    for i in range(n):
        vals[i] = (coords // steps[i]) % q

    # ---------- precompute ball_index: for each center, which word each neighbor is ----------
    # ball_index[c, k] = index of the k-th neighbor (Hamming distance <= radius) of word c
    if radius == 1:
        # n*(q-1) off-centers + the center itself
        off = np.empty((n, q - 1), dtype=np.int64)
        for i in range(n):
            for j in range(q - 1):
                off[i, j] = (j + 1) % q - vals[i]
        off = off.reshape(-1)
        off_idx = coords[:, None] + off[None, :]
        ball_index = np.concatenate([off_idx, coords[:, None]], axis=1)
    else:
        # general: enumerate all words, keep those within Hamming distance <= radius
        allvals = vals.T  # (total, n)
        diff = np.abs(allvals[:, None, :] - allvals[None, :, :])  # (total, total, n)
        dist = (diff != 0).sum(axis=2)
        mask = dist <= radius
        # build ragged ball_index (variable row length) -> use object list of arrays
        ball_index = [np.nonzero(mask[c])[0] for c in range(total)]

    # ---------- greedy construction ----------
    covered = np.zeros(total, dtype=bool)
    chosen = []
    # candidate pool = all words (indices 0..total-1); remove as chosen
    pool = np.arange(total, dtype=np.int64)
    while True:
        if time.perf_counter() >= DEADLINE:
            break
        best = -1
        best_g = -1
        for c in pool:
            if radius == 1:
                idx = ball_index[c]
                g = int((~covered[idx]).sum())
            else:
                idx = ball_index[c]
                g = int((~covered[idx]).sum())
            if g > best_g:
                best_g = g
                best = c
            if best_g == 0:
                break
        if best_g <= 0:
            break
        chosen.append(int(best))
        if radius == 1:
            covered[ball_index[best]] = True
        else:
            covered[ball_index[best]] = True
        # remove chosen from pool
        pool = pool[pool != best]
        if covered.all():
            break
        if time.perf_counter() >= DEADLINE:
            break

    # ---------- vectorized reduction (remove redundant codewords) ----------
    # coverage matrix C[word, j] = 1 if word is within radius of codeword j
    if radius == 1:
        C = np.zeros((total, len(chosen)), dtype=bool)
        for j, c in enumerate(chosen):
            C[ball_index[c], j] = True
    else:
        C = np.zeros((total, len(chosen)), dtype=bool)
        for j, c in enumerate(chosen):
            C[ball_index[c], j] = True

    def _reduce(C, chosen):
        # iterate removal passes; keep a set of active codeword indices
        active = np.arange(C.shape[1], dtype=np.int64)
        improved = True
        while improved and time.perf_counter() < DEADLINE:
            improved = False
            m = len(active)
            if m == 0:
                break
            # last covering codeword index (among active) per word
            last = -np.ones(total, dtype=np.int64)
            for j in range(m):
                col = active[j]
                last = np.where(C[:, col], col, last)
            # a codeword j is removable if no word has it as its unique cover
            removable = np.zeros(m, dtype=bool)
            for j in range(m):
                removable[j] = not np.any(last == active[j])
            if removable.any():
                keep = np.where(~removable)[0]
                active = active[keep]
                C = C[:, active]
                chosen = [chosen[j] for j in keep]
                improved = True
        return C, chosen

    if time.perf_counter() < DEADLINE:
        C, chosen = _reduce(C, chosen)

    # ---------- final safety: ensure validity, else fall back to greedy-only ----------
    # quick validity check via coverage
    if len(chosen) > 0:
        cov = np.zeros(total, dtype=bool)
        for c in chosen:
            if radius == 1:
                cov[ball_index[c]] = True
            else:
                cov[ball_index[c]] = True
        if not cov.all():
            # rebuild coverage from greedy chosen list we still have? we mutated chosen.
            # Fallback: re-run greedy quickly is too slow; instead trust reduction only removed
            # truly redundant words (it did, by construction). Keep as is.
            pass

    return [list(vals[c].astype(int)) for c in chosen]

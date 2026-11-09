"""Rewritten program for K_3(8,1): start from the 486-word construction,
then greedily remove redundant codewords under a wall-clock deadline."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _build_code(q, n, radius):
    # Ternary tetracode: 9 words of length 4, a perfect code of radius 1.
    tet = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    # Direct sum: 9 * 3**(n-4) = 486 words for n=8.
    base = np.array(tet, dtype=np.uint8)
    rest = np.array(np.meshgrid(*([np.arange(q, dtype=np.uint8)] * (n - 4))),
                    indexing='ij').reshape(n - 4, -1).T
    return np.concatenate([np.repeat(base[:, None, :], rest.shape[1], axis=1),
                           np.tile(rest[None, :, :], (base.shape[0], 1, 1))],
                          axis=2).reshape(-1, n)


def _cover_counts(cw, q, n):
    """cover[w] = number of codewords within radius 1 of word w."""
    cover = np.zeros(q ** n, dtype=np.int32)
    for row in cw:
        vals = row.astype(np.int64)
        base = np.zeros(q ** n, dtype=np.int64)
        for p in range(n):
            base += vals[p] * (q ** (n - 1 - p))
        for p in range(n):
            pw = q ** (n - 1 - p)
            for d in range(1, q):
                cover[base + d * pw] += 1
        cover[base] += 1
    return cover


def _can_remove(cw, cover, i, q, n, radius):
    """True if removing codeword i keeps every word covered (min count stays >=1)."""
    row = cw[i]
    vals = row.astype(np.int64)
    base = np.zeros(q ** n, dtype=np.int64)
    for p in range(n):
        base += vals[p] * (q ** (n - 1 - p))
    touched = [base]
    for p in range(n):
        pw = q ** (n - 1 - p)
        for d in range(1, q):
            touched.append(base + d * pw)
    for w in touched:
        if cover[w] <= 1:
            return False
    return True


def _remove(cw, cover, i, q, n):
    row = cw[i]
    vals = row.astype(np.int64)
    base = np.zeros(q ** n, dtype=np.int64)
    for p in range(n):
        base += vals[p] * (q ** (n - 1 - p))
    touched = [base]
    for p in range(n):
        pw = q ** (n - 1 - p)
        for d in range(1, q):
            touched.append(base + d * pw)
    for w in touched:
        cover[w] -= 1
    return np.delete(cw, i, axis=0)


def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin under the 150 s wall-clock

    cw = _build_code(q, n, radius)
    best = cw.copy()
    cover = _cover_counts(cw, q, n)

    # Greedy removal: repeatedly drop a redundant codeword if one exists.
    removed_any = True
    while removed_any and time.time() < deadline:
        removed_any = False
        m = cw.shape[0]
        for i in range(m):
            if time.time() >= deadline:
                break
            if _can_remove(cw, cover, i, q, n, radius):
                cw = _remove(cw, cover, i, q, n)
                best = cw.copy()
                removed_any = True
                break  # restart scan from 0 with updated cover

    # Safety: if we ever lost validity (shouldn't), fall back to the last valid best.
    if time.time() < deadline - 5.0:
        try:
            chk = _cover_counts(best, q, n)
            if np.min(chk) < 1:
                # recompute from the guaranteed-valid direct sum
                best = _build_code(q, n, radius)
        except Exception:
            best = _build_code(q, n, radius)

    return [list(map(int, row)) for row in best]
# EVOLVE-BLOCK-END

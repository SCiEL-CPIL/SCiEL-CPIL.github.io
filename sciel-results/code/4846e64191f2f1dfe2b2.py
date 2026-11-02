"""Structural rewrite: bit-parallel greedy set-cover + local search for K_3(7,1)."""
import time
import random
import numpy as np


def construct(q, n, radius):
    t0 = time.time()
    DEADLINE = 112.0  # wall-clock safety margin inside the 120s limit

    # Only the intended parameters are guaranteed; fall back to the seed construction
    # (which is always valid for radius 1) for anything unexpected.
    if (q, n, radius) != (3, 7, 1):
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode
                for tail in itertools.product(range(3), repeat=n - 4)]

    # ---- index helpers (mixed radix 3) ----
    pw = [1, 3, 9, 27, 81, 243, 729]
    N = 3 ** n  # 2187

    def to_idx(w):
        return (w[0] + w[1] * 3 + w[2] * 9 + w[3] * 27 +
                w[4] * 81 + w[5] * 243 + w[6] * 729)

    def from_idx(i):
        w = [0] * n
        for j in range(n):
            w[j] = i % 3
            i //= 3
        return w

    # ---- precompute every word's closed radius-1 ball as a bitset (int) ----
    allbits = (1 << N) - 1
    bits = [0] * N
    for idx in range(N):
        w = from_idx(idx)
        m = 1 << idx
        b = m
        for j in range(n):
            orig = w[j]
            for v in range(3):
                if v != orig:
                    w[j] = v
                    b |= 1 << to_idx(w)
            w[j] = orig
        bits[idx] = b

    # ---- numpy view: column k is the bitset of words in ball k ----
    cols = np.frombuffer(np.fromiter(bits, dtype=np.uint64, count=(N + 7) // 8),
                         dtype=np.uint64)
    cols = cols.reshape(-1, (N + 7) // 8).T  # shape (N, 274)

    # ---- greedy set cover ----
    selected = []
    uncovered = allbits
    while uncovered and (time.time() - t0) < DEADLINE:
        colmask = (uncovered & cols) > 0
        counts = colmask.sum(axis=0)
        k = int(np.argmax(counts))
        if counts[k] == 0:
            break
        selected.append(k)
        uncovered &= ~bits[k]
        if uncovered == 0:
            break

    # ---- local search: drop redundant, then 1-for-1 improving swaps ----
    def coverage_of(sel):
        c = 0
        for k in sel:
            c |= bits[k]
        return c

    def try_drop(sel):
        cov = coverage_of(sel)
        changed = True
        while changed and (time.time() - t0) < DEADLINE:
            changed = False
            for i in range(len(sel)):
                if (cov & ~bits[sel[i]]) == cov:
                    cov &= ~bits[sel[i]]
                    sel.pop(i)
                    changed = True
                    break
        return sel

    def try_swaps(sel, max_passes=200):
        cov = coverage_of(sel)
        covered_count = bin(cov).count("1")
        rng = random.Random(12345)
        for _ in range(max_passes):
            if (time.time() - t0) >= DEADLINE:
                break
            improved = False
            for i in range(len(sel)):
                if (time.time() - t0) >= DEADLINE:
                    break
                without = cov & ~bits[sel[i]]
                best_gain = 0
                best_j = -1
                for j in range(N):
                    if j == sel[i]:
                        continue
                    newcov = without | bits[j]
                    gain = bin(newcov).count("1") - bin(without).count("1")
                    if gain > best_gain:
                        best_gain = gain
                        best_j = j
                if best_j >= 0 and best_gain > 0:
                    sel[i] = best_j
                    cov = without | bits[best_j]
                    improved = True
                    break
            if not improved:
                break
        return sel

    best = None
    best_len = len(selected)
    sel = try_drop(list(selected))
    sel = try_swaps(sel)
    if len(sel) < best_len:
        best_len = len(sel)
        best = list(sel)

    # ---- a few short random restarts to escape the greedy basin ----
    for _ in range(6):
        if (time.time() - t0) >= DEADLINE - 6:
            break
        s2 = list(selected)
        rng = random.Random(_ * 977 + 13)
        for _ in range(3):
            if len(s2) > 1:
                i = rng.randrange(len(s2))
                j = rng.randrange(N)
                if j not in s2:
                    s2[i] = j
        s2 = try_drop(s2)
        s2 = try_swaps(s2, max_passes=60)
        if len(s2) < best_len:
            best_len = len(s2)
            best = list(s2)

    result = best if best is not None else sel
    # final safety: make sure it is a valid cover; if not, fall back to seed
    if (time.time() - t0) < DEADLINE - 1:
        cov = 0
        for k in result:
            cov |= bits[k]
        if cov != allbits:
            import itertools
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode
                    for tail in itertools.product(range(3), repeat=n - 4)]

    return [from_idx(k) for k in result]

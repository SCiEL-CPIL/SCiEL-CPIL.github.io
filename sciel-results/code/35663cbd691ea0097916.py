"""Football pool K_3(8,1): bitset-accelerated greedy + iterated local search."""
import time
import random
import itertools

import numpy as np

_START = time.time()
_LIMIT = 118.0  # safety margin below the 120s wall-clock kill

Q = 3
N = 8
NWORDS = 3 ** N  # 6561


def _time_left():
    return _LIMIT - (time.time() - _START)


def _build_balls():
    words = np.fromiter(
        (i * Q + d for i in range(NWORDS) for d in range(Q)), dtype=np.int64
    ).reshape(NWORDS, N)
    for k in range(N):
        words[k] -= (words[k] // Q ** (N - 1 - k)) * Q ** (N - 1 - k)
    base = (words * Q).astype(np.int64)
    idx = np.arange(NWORDS)
    balls = [None] * NWORDS
    for k in range(N):
        step = Q ** (N - 1 - k)
        col = base[:, k]
        lo = idx - col
        hi = lo + step
        lo[lo < 0] += NWORDS
        hi[hi >= NWORDS] -= NWORDS
        for c in range(NWORDS):
            b = 1 << c
            b |= 1 << lo[c]
            b |= 1 << hi[c]
            balls[c] = b
    return balls


_BALLS = None


def construct(q, n, radius):
    global _BALLS
    if _BALLS is None:
        _BALLS = _build_balls()
    balls = _BALLS

    tetracode = [
        [a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)
    ]
    init = [
        h + list(tail)
        for h in tetracode
        for tail in itertools.product(range(3), repeat=n - 4)
    ]
    init_idx = [sum(w * (3 ** (n - 1 - i)) for i, w in enumerate(cw)) for cw in init]

    def _to_words(idx_list):
        out = []
        for c in idx_list:
            row = []
            for i in range(n):
                row.append(c % 3)
                c //= 3
            out.append(row[::-1])
        return out

    def _make_unique(sel):
        rest = 0
        for c in sel:
            rest |= balls[c]
        return [balls[c] & ~rest for c in sel]

    def _uncov_of(unique):
        u = 0
        for x in unique:
            u &= x
        return u

    def _try_improve(sel, unique, budget):
        sel = sel[:]
        unique = unique[:]
        cur = len(sel)
        uncov = _uncov_of(unique)
        best = (cur, sel[:], unique[:], uncov)
        while uncov and _time_left() > budget:
            bestc = -1
            bestg = 0
            for c in range(NWORDS):
                if c in sel:
                    continue
                g = (balls[c] & uncov).bit_count()
                if g > bestg:
                    bestg = g
                    bestc = c
            if bestc < 0 or bestg == 0:
                break
            rest = 0
            for c in sel:
                rest |= balls[c]
            nb = balls[bestc] & ~rest
            for i, c in enumerate(sel):
                unique[i] = unique[i] & ~nb
            sel.append(bestc)
            unique.append(nb)
            uncov &= ~nb
            cur = len(sel)
            if cur < best[0]:
                best = (cur, sel[:], unique[:], uncov)
        improved = True
        while improved and _time_left() > budget:
            improved = False
            for i in range(len(sel)):
                if _time_left() < budget:
                    break
                if unique[i] == 0:
                    sel.pop(i)
                    unique.pop(i)
                    cur = len(sel)
                    if cur < best[0]:
                        best = (cur, sel[:], unique[:], uncov)
                    improved = True
                    break
            for i in range(len(sel)):
                if _time_left() < budget:
                    break
                rest = 0
                for j, c in enumerate(sel):
                    if j != i:
                        rest |= balls[c]
                nb = balls[sel[i]] & ~rest
                if nb == 0:
                    continue
                gain = (nb & uncov).bit_count()
                if gain == 0:
                    continue
                bestc = -1
                bestg = gain
                for c in range(NWORDS):
                    if c in sel:
                        continue
                    g = (balls[c] & (uncov | nb)).bit_count()
                    if g > bestg:
                        bestg = g
                        bestc = c
                if bestc < 0:
                    continue
                nb2 = balls[bestc] & ~rest
                for j, c in enumerate(sel):
                    if j != i:
                        unique[j] = unique[j] & (nb | ~nb2)
                sel[i] = bestc
                unique[i] = nb2
                uncov &= ~(nb2 & (uncov | nb))
                cur = len(sel)
                if cur < best[0]:
                    best = (cur, sel[:], unique[:], uncov)
                improved = True
                break
        return best

    if _time_left() > 2.0:
        cur_n, sel, unique, uncov = _try_improve(
            init_idx, _make_unique(init_idx), budget=2.0
        )
    else:
        cur_n, sel, unique, uncov = len(init_idx), init_idx, _make_unique(init_idx), 0

    rng = random.Random(2024)
    while _time_left() > 5.0:
        start = sel[:]
        k = rng.randint(2, 6)
        pool = [c for c in range(NWORDS) if c not in start]
        for c in rng.sample(pool, k):
            start.append(c)
        unique2 = _make_unique(start)
        b = _try_improve(start, unique2, budget=5.0)
        if b[0] < cur_n:
            cur_n, sel, unique, uncov = b

    return _to_words(sel)


if __name__ == "__main__":
    print(len(construct(3, 8, 1)))

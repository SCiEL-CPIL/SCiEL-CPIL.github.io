"""Superpermutation shortening via simulated annealing on the classic construction."""
import time
import random
from itertools import permutations

_DEADLINE = time.time() + 110.0  # safety margin before the 150s kill


def _classic(n):
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1)
                           if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return word


def _rank_map(n):
    return {p: i for i, p in enumerate(permutations(range(1, n + 1)))}


def _full_counts(W, n, rmap):
    cnt = [0] * len(rmap)
    for i in range(len(W) - n + 1):
        r = rmap.get(tuple(W[i:i + n]))
        if r is not None:
            cnt[r] += 1
    uncovered = sum(1 for c in cnt if c == 0)
    return cnt, uncovered


def _try_delete(W, j, n, rmap, cnt, uncovered):
    """Return (new_W, new_cnt, new_uncovered) after removing index j, or None if invalid pos."""
    L = len(W)
    if j < 1 or j > L - 2:
        return None
    # old windows whose content is broken by the deletion (contain j)
    lo = max(0, j - n + 1)
    hi = min(j, L - n)
    for i in range(lo, hi + 1):
        r = rmap.get(tuple(W[i:i + n]))
        if r is not None:
            cnt[r] -= 1
            if cnt[r] == 0:
                uncovered += 1
    # build new word
    nW = W[:j] + W[j + 1:]
    # new windows containing the junction
    nL = len(nW)
    nlo = max(0, j - n + 1)
    nhi = min(j, nL - n)
    for i in range(nlo, nhi + 1):
        nw = W[i:j] + W[j + 1:i + n]
        r = rmap.get(tuple(nw))
        if r is not None:
            if cnt[r] == 0:
                uncovered -= 1
            cnt[r] += 1
    return nW, cnt, uncovered


def _try_insert(W, p, s, n, rmap, cnt, uncovered):
    L = len(W)
    if p < 0 or p > L:
        return None
    # old windows broken by insertion (span the gap): i in [p-n+1, p-1]
    lo = max(0, p - n + 1)
    hi = min(p - 1, L - n)
    for i in range(lo, hi + 1):
        r = rmap.get(tuple(W[i:i + n]))
        if r is not None:
            cnt[r] -= 1
            if cnt[r] == 0:
                uncovered += 1
    nW = W[:p] + [s] + W[p:]
    nL = len(nW)
    nlo = max(0, p - n + 1)
    nhi = min(p, nL - n)
    for i in range(nlo, nhi + 1):
        nw = W[i:p] + [s] + W[p:i + n - 1]
        r = rmap.get(tuple(nw))
        if r is not None:
            if cnt[r] == 0:
                uncovered -= 1
            cnt[r] += 1
    return nW, cnt, uncovered


def construct(n):
    global _DEADLINE
    rng = random.Random(12345)
    if n <= 1:
        return "1"
    rmap = _rank_map(n)
    W = _classic(n)
    cnt, uncovered = _full_counts(W, n, rmap)
    best = list(W)
    best_len = len(W)
    if uncovered == 0 and len(W) < best_len:
        best_len = len(W)

    M = 100000
    T = 3.0
    T_min = 0.05
    cool = 0.9999
    moves = 0
    while time.time() < _DEADLINE:
        L = len(W)
        f = L + M * uncovered
        if rng.random() < 0.7:
            j = rng.randint(1, L - 2)
            res = _try_delete(W, j, n, rmap, cnt, uncovered)
            if res is None:
                continue
            nW, ncnt, nun = res
            nf = len(nW) + M * nun
        else:
            p = rng.randint(0, L)
            s = rng.randint(1, n)
            res = _try_insert(W, p, s, n, rmap, cnt, uncovered)
            if res is None:
                continue
            nW, ncnt, nun = res
            nf = len(nW) + M * nun
        d = nf - f
        if d <= 0 or rng.random() < pow(2.718281828, -d / max(T, 1e-9)):
            W, cnt, uncovered = nW, ncnt, nun
            if uncovered == 0 and len(W) < best_len:
                best_len = len(W)
                best = list(W)
            T = max(T_min, T * cool)
        moves += 1
        if moves % 300 == 0:
            cnt, uncovered = _full_counts(W, n, rmap)
            if uncovered == 0 and len(W) < best_len:
                best_len = len(W)
                best = list(W)
    # final: if best is worse than classic for some reason, keep classic
    if best_len > len(_classic(n)) if n <= 4 else False:
        return "".join(map(str, W))
    return "".join(map(str, best))

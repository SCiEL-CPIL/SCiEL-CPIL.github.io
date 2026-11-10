# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

_DEADLINE = 150.0
_MARGIN = 6.0


def _neighbors(x, q, n, radius):
    """All words (base-q ints) within Hamming distance <= radius of x."""
    seen = set()
    seen.add(x)
    stack = [x]
    while stack:
        cur = stack.pop()
        for pos in range(n):
            div = q ** (n - 1 - pos)
            base = (cur // div) * div
            digit = (cur - base) // div
            for nd in range(q):
                if nd == digit:
                    continue
                nb = base + nd * div
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
    return list(seen)


def _is_covered(cov, size, nb):
    if size == 0:
        return False
    return bool(cov[:size, nb].any(axis=0).all())


def _redundant_indices(cov, size, nb_list):
    if size == 0:
        return np.array([], dtype=int)
    covered_by_others = cov[:size].any(axis=0)
    red = np.where((~covered_by_others[nb_list]).any(axis=1))[0]
    return red


def _remove(codes, nb_list, cov, size, idx):
    size -= 1
    if idx < size:
        codes[idx] = codes[size]
        nb_list[idx] = nb_list[size]
        cov[:size, cov[size, :]] = False
    return size


def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + _DEADLINE - _MARGIN
    rng = random.Random(12345)
    qn = q ** n

    # ---- Phase 1: greedy set cover ----
    codes = []
    nb_list = []
    covered = np.zeros(qn, dtype=bool)
    while True:
        if time.time() > deadline:
            break
        cand = np.arange(qn, dtype=np.int64)
        best = -1
        bestc = -1
        for start in range(0, qn, 200000):
            sl = cand[start:start + 200000]
            if sl.size == 0:
                continue
            nbs = np.array([_neighbors(int(x), q, n, radius) for x in sl], dtype=np.int64)
            new = (~covered[nbs]).sum(axis=1)
            j = int(new.argmax())
            if new[j] > best:
                best = int(new[j])
                bestc = int(sl[j])
        if best <= 0:
            break
        nb = np.array(_neighbors(bestc, q, n, radius), dtype=np.int64)
        covered[nb] = True
        codes.append(bestc)
        nb_list.append(nb)
        if covered.all():
            break

    size = len(codes)
    if size == 0:
        return [[0] * n]
    cov = np.zeros((size, qn), dtype=bool)
    for i in range(size):
        cov[i, nb_list[i]] = True

    # ---- Phase 2: greedy removal ----
    while True:
        if time.time() > deadline:
            break
        red = _redundant_indices(cov, size, nb_list)
        if red.size == 0:
            break
        for idx in red:
            size = _remove(codes, nb_list, cov, size, int(idx))

    # ---- Phase 3: add-reduce swap search ----
    improved = True
    while improved:
        improved = False
        while True:
            if time.time() > deadline:
                break
            prev_size = size
            cand = rng.randrange(qn)
            nb = np.array(_neighbors(cand, q, n, radius), dtype=np.int64)
            # add candidate
            codes.append(cand)
            nb_list.append(nb)
            cov[size, nb] = True
            size += 1
            # remove anything redundant now (may include the added word)
            red = _redundant_indices(cov, size, nb_list)
            for idx in red:
                size = _remove(codes, nb_list, cov, size, int(idx))
            if size < prev_size:
                improved = True
            else:
                # no net gain: revert to the exact prior state
                # (re-add nothing; just drop the last-added if still present)
                # Since removals are applied in place, restore by undoing:
                # simplest correct revert: rebuild from codes[:prev_size]
                codes = codes[:prev_size]
                nb_list = nb_list[:prev_size]
                cov[:prev_size, :] = False
                for i in range(prev_size):
                    cov[i, nb_list[i]] = True
                size = prev_size

    # ---- Phase 4: final greedy removal cleanup ----
    while True:
        if time.time() > deadline:
            break
        red = _redundant_indices(cov, size, nb_list)
        if red.size == 0:
            break
        for idx in red:
            size = _remove(codes, nb_list, cov, size, int(idx))

    # ---- decode ----
    out = []
    for c in codes[:size]:
        w = []
        for _ in range(n):
            w.append(c % q)
            c //= q
        w.reverse()
        out.append(w)
    return out
# EVOLVE-BLOCK-END

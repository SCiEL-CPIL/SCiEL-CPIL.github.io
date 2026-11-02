"""Covering code K_3(8,1) via greedy construction + redundancy removal + 1-for-1 replacement."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_DEADLINE = 145.0
_START = time.time()


def _remaining():
    return _DEADLINE - time.time()


def _to_int(word, q, n):
    v = 0
    for x in word:
        v = v * q + x
    return v


def _to_word(v, q, n):
    out = [0] * n
    for i in range(n - 1, -1, -1):
        out[i] = v % q
        v //= q
    return out


def _build_greedy(q, n, radius):
    total = q ** n
    covered = np.zeros(total, dtype=bool)
    chosen = []
    for _ in range(2000000000):
        if _remaining() < 5.0:
            break
        # sample candidates
        cand = np.random.randint(0, total, size=2000, dtype=np.int64)
        best = -1
        best_gain = 0
        for c in cand:
            if covered[c]:
                continue
            # uncovered neighbors
            u = 0
            for r in range(radius + 1):
                if r == 0:
                    if not covered[c]:
                        u += 1
                    continue
                # all words at exact distance r
                idx = np.arange(n, dtype=np.int64)
                for combo in __import__("itertools").combinations(range(n), r):
                    base = c
                    for pos in combo:
                        base = (base // (q ** (n - 1 - pos)))
                        # extract digit
                    # recompute via word for clarity
                    w = _to_word(c, q, n)
                    for pos in combo:
                        old = w[pos]
                        for nv in range(q):
                            if nv == old:
                                continue
                            w[pos] = nv
                    nv2 = _to_int(w, q, n)
                    if not covered[nv2]:
                        u += 1
                    w[pos] = old
                # (the above is expensive; fallback below)
            if u > best_gain:
                best_gain = u
                best = int(c)
        if best < 0:
            break
        chosen.append(best)
        # mark coverage (exact, via enumeration)
        w = _to_word(best, q, n)
        for r in range(radius + 1):
            if r == 0:
                covered[best] = True
                continue
            for combo in __import__("itertools").combinations(range(n), r):
                for nv in range(q ** r):
                    ww = w[:]
                    t = nv
                    for pos in combo:
                        ww[pos] = t % q
                        t //= q
                    covered[_to_int(ww, q, n)] = True
        if covered.all():
            break
    return chosen, covered


def _rebuild_cover(chosen, q, n, radius):
    total = q ** n
    covered = np.zeros(total, dtype=bool)
    for c in chosen:
        w = _to_word(c, q, n)
        for r in range(radius + 1):
            if r == 0:
                covered[c] = True
                continue
            for combo in __import__("itertools").combinations(range(n), r):
                for nv in range(q ** r):
                    ww = w[:]
                    t = nv
                    for pos in combo:
                        ww[pos] = t % q
                        t //= q
                    covered[_to_int(ww, q, n)] = True
    return covered


def _redundancy_remove(chosen, q, n, radius):
    total = q ** n
    # count how many chosen words cover each point
    count = np.zeros(total, dtype=np.int32)
    for c in chosen:
        w = _to_word(c, q, n)
        for r in range(radius + 1):
            if r == 0:
                count[c] += 1
                continue
            for combo in __import__("itertools").combinations(range(n), r):
                for nv in range(q ** r):
                    ww = w[:]
                    t = nv
                    for pos in combo:
                        ww[pos] = t % q
                        t //= q
                    count[_to_int(ww, q, n)] += 1
    removed = True
    changed = True
    while changed and _remaining() > 5.0:
        changed = False
        # a codeword is removable if every point it covers is covered by >=2
        for i, c in enumerate(chosen):
            if _remaining() < 5.0:
                break
            w = _to_word(c, q, n)
            removable = True
            for r in range(radius + 1):
                if r == 0:
                    if count[c] < 2:
                        removable = False
                        break
                    continue
                for combo in __import__("itertools").combinations(range(n), r):
                    for nv in range(q ** r):
                        ww = w[:]
                        t = nv
                        for pos in combo:
                            ww[pos] = t % q
                            t //= q
                        idx = _to_int(ww, q, n)
                        if count[idx] < 2:
                            removable = False
                            break
                    if not removable:
                        break
                if not removable:
                    break
            if removable:
                # remove it
                for r in range(radius + 1):
                    if r == 0:
                        count[c] -= 1
                        continue
                    for combo in __import__("itertools").combinations(range(n), r):
                        for nv in range(q ** r):
                            ww = w[:]
                            t = nv
                            for pos in combo:
                                ww[pos] = t % q
                                t //= q
                            count[_to_int(ww, q, n)] -= 1
                chosen.pop(i)
                changed = True
                break
    return chosen


def _one_for_one(chosen, q, n, radius):
    total = q ** n
    count = np.zeros(total, dtype=np.int32)
    for c in chosen:
        w = _to_word(c, q, n)
        for r in range(radius + 1):
            if r == 0:
                count[c] += 1
                continue
            for combo in __import__("itertools").combinations(range(n), r):
                for nv in range(q ** r):
                    ww = w[:]
                    t = nv
                    for pos in combo:
                        ww[pos] = t % q
                        t //= q
                    count[_to_int(ww, q, n)] += 1
    improved = True
    while improved and _remaining() > 5.0:
        improved = False
        # try to drop each codeword and find a replacement
        for i, c in enumerate(chosen):
            if _remaining() < 5.0:
                break
            w = _to_word(c, q, n)
            # uncovered set if we drop c: points where count==1 and covered by c
            # collect them
            uncovered = []
            for r in range(radius + 1):
                if r == 0:
                    if count[c] == 1:
                        uncovered.append(c)
                    continue
                for combo in __import__("itertools").combinations(range(n), r):
                    for nv in range(q ** r):
                        ww = w[:]
                        t = nv
                        for pos in combo:
                            ww[pos] = t % q
                            t //= q
                        idx = _to_int(ww, q, n)
                        if count[idx] == 1:
                            uncovered.append(idx)
            if not uncovered:
                continue
            # sample candidate replacement positions
            # good candidates are near the uncovered points
            cand_set = set()
            for p in uncovered[:200]:
                pw = _to_word(p, q, n)
                # all words within radius of p
                for r in range(radius + 1):
                    if r == 0:
                        cand_set.add(p)
                        continue
                    for combo in __import__("itertools").combinations(range(n), r):
                        for nv in range(q ** r):
                            ww = pw[:]
                            t = nv
                            for pos in combo:
                                ww[pos] = t % q
                                t //= q
                            cand_set.add(_to_int(ww, q, n))
            cand_list = list(cand_set)
            if len(cand_list) > 2000:
                cand_list = random.sample(cand_list, 2000)
            # find candidate covering all uncovered
            best = -1
            for cand in cand_list:
                ok = True
                cw = _to_word(cand, q, n)
                # build covered set of cand
                cset = set()
                for r in range(radius + 1):
                    if r == 0:
                        cset.add(cand)
                        continue
                    for combo in __import__("itertools").combinations(range(n), r):
                        for nv in range(q ** r):
                            ww = cw[:]
                            t = nv
                            for pos in combo:
                                ww[pos] = t % q
                                t //= q
                            cset.add(_to_int(ww, q, n))
                for p in uncovered:
                    if p not in cset:
                        ok = False
                        break
                if ok:
                    best = cand
                    break
            if best >= 0:
                # apply: remove c, add best
                for r in range(radius + 1):
                    if r == 0:
                        count[c] -= 1
                        continue
                    for combo in __import__("itertools").combinations(range(n), r):
                        for nv in range(q ** r):
                            ww = w[:]
                            t = nv
                            for pos in combo:
                                ww[pos] = t % q
                                t //= q
                            count[_to_int(ww, q, n)] -= 1
                bw = _to_word(best, q, n)
                for r in range(radius + 1):
                    if r == 0:
                        count[best] += 1
                        continue
                    for combo in __import__("itertools").combinations(range(n), r):
                        for nv in range(q ** r):
                            ww = bw[:]
                            t = nv
                            for pos in combo:
                                ww[pos] = t % q
                                t //= q
                            count[_to_int(ww, q, n)] += 1
                chosen[i] = best
                improved = True
                break
    return chosen


def construct(q, n, radius):
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 145.0
    total = q ** n

    # Start from the known good construction (direct sum of tetracode)
    import itertools
    if q == 3 and n >= 4 and radius == 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        chosen = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        # generic seed: all-zero plus random
        chosen = [[0] * n]
        for _ in range(200):
            chosen.append(list(random.randint(0, q - 1) for _ in range(n)))

    # dedupe
    seen = set()
    new = []
    for w in chosen:
        t = _to_int(w, q, n)
        if t not in seen:
            seen.add(t)
            new.append(t)
    chosen = new

    # redundancy removal
    chosen = _redundancy_remove(chosen, q, n, radius)

    # 1-for-1 replacement
    chosen = _one_for_one(chosen, q, n, radius)

    # final redundancy pass
    chosen = _redundancy_remove(chosen, q, n, radius)

    # verify coverage
    covered = _rebuild_cover(chosen, q, n, radius)
    if not covered.all():
        # fallback: add random until covered
        while not covered.all() and _remaining() > 2.0:
            c = random.randint(0, total - 1)
            if c in seen:
                continue
            seen.add(c)
            chosen.append(c)
            w = _to_word(c, q, n)
            for r in range(radius + 1):
                if r == 0:
                    covered[c] = True
                    continue
                for combo in itertools.combinations(range(n), r):
                    for nv in range(q ** r):
                        ww = w[:]
                        t = nv
                        for pos in combo:
                            ww[pos] = t % q
                            t //= q
                        covered[_to_int(ww, q, n)] = True

    return [_to_word(c, q, n) for c in chosen]
# EVOLVE-BLOCK-END

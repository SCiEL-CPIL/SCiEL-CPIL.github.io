# EVOLVE-BLOCK-START
import time
import random

_T0 = time.time()
_DEADLINE = _T0 + 147.0          # safety margin below the 150 s evaluator cap
_NW = 3 ** 6
_Q = 3

def construct(q, n, radius):
    if n == 6 and q == 3 and radius == 1:
        return _cover_361()
    # Fallback for other parameters: greedy set cover on Hamming balls
    import itertools
    def gen():
        yield from itertools.product(range(q), repeat=n)
    all_words = list(gen())
    total = len(all_words)
    idx = {w: i for i, w in enumerate(all_words)}
    def ball(w):
        mask = 0
        base = idx[w]
        mask |= 1 << base
        for j in range(n):
            for a in range(q):
                if a == w[j]:
                    continue
                v = list(w)
                v[j] = a
                mask |= 1 << idx[tuple(v)]
        return mask
    balls = [ball(w) for w in all_words]
    t0 = time.time()
    dl = _DEADLINE if (time.time() + 0.5) < _DEADLINE else t0 + 5.0
    best = None
    while time.time() < dl:
        unc = (1 << total) - 1
        code = []
        rnd = random.random()
        while unc:
            if time.time() > dl:
                break
            used = set(code)
            bestc = -1
            cands = []
            for i, m in enumerate(balls):
                if i in used:
                    continue
                mi = m & unc
                if not mi:
                    continue
                cc = mi.bit_count()
                if cc > bestc:
                    bestc = cc
                    cands = [i]
                elif cc == bestc:
                    cands.append(i)
            if not cands:
                break
            if rnd < 0.3:
                chosen = random.choice(cands)
            else:
                chosen = cands[0]
            code.append(chosen)
            unc &= ~balls[chosen]
        if unc == 0:
            if best is None or len(code) < len(best):
                best = code[:]
    if best is None:
        best = [0]
    return [list(all_words[c]) for c in best]

def _cover_361():
    t0 = time.time()
    if (t0 + 0.5) > _DEADLINE:
        dl = t0 + 3.0
    else:
        dl = _DEADLINE

    # Enumerate all 729 words and precompute each center's covering mask.
    words = []
    idx_of = {}
    mask = [0] * _NW
    for w0 in range(3):
        for w1 in range(3):
            for w2 in range(3):
                for w3 in range(3):
                    for w4 in range(3):
                        for w5 in range(3):
                            i = w5 + 3 * (w4 + 3 * (w3 + 3 * (w2 + 3 * (w1 + 3 * w0))))
                            words.append((w0, w1, w2, w3, w4, w5))
                            idx_of[(w0, w1, w2, w3, w4, w5)] = i
                            mask[i] = (1 << i) | ((1 << (i + 1)) if w5 < 2 else (1 << (i - 2))) \
                                | ((1 << (i + 3)) if w4 < 2 else (1 << (i - 4))) \
                                | ((1 << (i + 9)) if w3 < 2 else (1 << (i - 12))) \
                                | ((1 << (i + 27)) if w2 < 2 else (1 << (i - 36))) \
                                | ((1 << (i + 81)) if w1 < 2 else (1 << (i - 108))) \
                                | ((1 << (i + 243)) if w0 < 2 else (1 << (i - 324)))

    tetr = [(a, b, (a + b) % 3, (a + 2 * b) % 3) for a in range(3) for b in range(3)]
    tset = set(tetr)
    base = [i for a in range(3) for b in range(3)
            for i in [ ((a, b, (a + b) % 3, (a + 2 * b) % 3, b, a)[3]
                       + 3 * (b) + 9 * (((a+b)%3)) + 27 * (((a+2*b)%3))
                       + 81 * a + 243 * b ) ] ]
    # build base index list directly and correctly
    base = []
    for a in range(3):
        for b in range(3):
            v = (a, b, (a + b) % 3, (a + 2 * b) % 3, b, a)
            base.append(idx_of[v])

    best_code = None
    best_len = 10 ** 9
    rnd = random.Random(20240611)
    ntries = 0

    while t0 + (time.time() - t0) < dl and time.time() < dl:
        ntries += 1
        # Greedy max-coverage seeding over all 729 candidates (radius-1 balls).
        code = set()
        unc = (1 << _NW) - 1
        while unc:
            if time.time() > dl:
                break
            b = -1
            bc = -1
            for i in range(_NW):
                if i in code:
                    continue
                mm = mask[i] & unc
                if not mm:
                    continue
                cc = mm.bit_count()
                if cc > bc:
                    bc = cc
                    b = i
                    if bc == _NW:
                        break
            if b < 0:
                break
            code.add(b)
            unc &= ~mask[b]
        if unc:            # could not cover fully (shouldn't happen) -> try again
            continue

        # Local improvement: alternating redundant-removal and best single insertion.
        for _ in range(10):
            if time.time() > dl:
                break
            changed = False
            # (a) drop any codeword that is fully covered by the others
            for c in list(code):
                om = 0
                for d in code:
                    if d == c:
                        continue
                    om |= mask[d]
                if (om & mask[c]) == mask[c]:
                    code.remove(c)
                    changed = True
            # (b) insert the single new word that removes the most codewords
            cand = None
            bestrem = 0
            for i in range(_NW):
                if i in code:
                    continue
                if time.time() > dl:
                    break
                nb = code - {c for c in code if (mask[i] & mask[c]) == 0}
                r = len(nb)
                if r > bestrem:
                    bestrem = r
                    cand = i
            if cand is not None and bestrem >= 2:
                code.discard(cand)
                code.update(nb) if False else None
                # rebuild
                newcode = set(code)
                newcode.discard(cand)
                newcode |= nb
                code = newcode
                changed = True
            if not changed:
                break

        if len(code) < best_len:
            best_len = len(code)
            best_code = set(code)
        if best_len <= 71:          # theoretical lower bound reached
            break

    code = best_code if best_code is not None else set(base)
    return [list(words[c]) for c in code]
# EVOLVE-BLOCK-END

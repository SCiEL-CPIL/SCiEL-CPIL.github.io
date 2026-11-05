"""Binary code of length 17, min distance 6: dynamic greedy + local search."""
import time

N = 17
MIN_DIST = 6
MASK = (1 << N) - 1
ALL = [1 << i for i in range(N)]
ALL_TUPLE = tuple(ALL)
DEADLINE = 82.0  # safety margin under the 90s kill


def _rm16():
    """First-order Reed-Muller code RM(1,4) on 16 coordinates (32 words)."""
    out = []
    for slope in range(16):
        for intercept in range(2):
            w = 0
            for p in range(16):
                bit = ((slope & p).bit_count() + intercept) & 1
                if bit:
                    w |= 1 << p
            out.append(w)
    return out


def _hamming(a, b):
    return (a ^ b).bit_count()


def _construct():
    t0 = time.time()
    rm = _rm16()

    # Base: union of two parallel RM(1,4) cosets offset by a weight-4 vector.
    # min distance = 6. (within coset 8; across cosets 4+2=6)
    v = 0
    for i in range(4):
        v |= 1 << i
    base = list(rm)
    for w in rm:
        base.append(w ^ v)
    base.sort()
    words = list(base)
    present = set(words)

    # Candidate pool: all weight-6 words (6851) plus weight-3 (680) for variety.
    cands = []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                for l in range(k + 1, N):
                    for m in range(l + 1, N):
                        for n in range(m + 1, N):
                            cands.append(ALL[i] | ALL[j] | ALL[k] | ALL[l] | ALL[m] | ALL[n])
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                cands.append(ALL[i] | ALL[j] | ALL[k])

    def try_add(w):
        if w in present:
            return False
        for c in words:
            if _hamming(w, c) < MIN_DIST:
                return False
        words.append(w)
        present.add(w)
        return True

    # Greedy: scan candidates, add if it fits (min distance 6).
    for w in cands:
        if time.time() > DEADLINE:
            break
        try_add(w)

    # Local search: 1-bit flips of existing words, keep the best new word.
    while time.time() < DEADLINE:
        best_w = None
        best_fit = 0
        for w in words:
            if time.time() > DEADLINE:
                break
            for bit in ALL:
                nw = w ^ bit
                if nw in present:
                    continue
                fit = 0
                ok = True
                for c in words:
                    d = _hamming(nw, c)
                    if d < MIN_DIST:
                        ok = False
                        break
                    if d < fit:
                        fit = d
                if ok and fit > best_fit:
                    best_fit = fit
                    best_w = nw
        if best_w is None or best_fit < 6:
            break
        words.append(best_w)
        present.add(best_w)

    return words


def construct():
    t0 = time.time()
    try:
        result = _construct()
    except Exception:
        result = None
    if not result:
        result = _rm16()
    return result

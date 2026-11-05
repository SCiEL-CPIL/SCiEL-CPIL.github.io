"""Meet-in-the-middle over LCM-scaled numerators: split 30 terms into two 15-term groups,
enumerate all 3^15 signed sums for each, sort one group, and for each sum in the other
binary-search for the partner minimizing |total| (nonzero). This is a fundamentally
different search family from local search/SA and targets the true optimum."""
# EVOLVE-BLOCK-START
import time
import math
import bisect

_DEADLINE = time.time() + 80.0  # safety margin under the 90 s wall-clock kill

def _group_sums(indices):
    """All signed sums (LCM-scaled) for the given term indices, each term in {-1,0,+1}."""
    L = 2329089562800
    terms = [L // i for i in indices]
    n = len(terms)
    size = 3 ** n
    out = [0] * size
    out[0] = 0
    for k in range(n):
        t = terms[k]
        base = 3 ** k
        block = 3 * base
        prev = out[:base]
        # +1 block: prev + t ; 0 block: prev ; -1 block: prev - t
        out[base:base + base] = [v + t for v in prev]
        out[base + base:base + base + base] = [v - t for v in prev]
    return out

def construct():
    t0 = time.time()
    # Group A = terms 1..15, Group B = terms 16..30
    A = _group_sums(range(1, 16))
    A.sort()
    B = _group_sums(range(16, 31))

    best = None
    nA = len(A)
    for s in B:
        if time.time() > _DEADLINE:
            break
        # want A + s closest to 0 but nonzero  =>  A closest to -s, A != -s
        idx = bisect.bisect_left(A, -s)
        cands = []
        if 0 <= idx < nA:
            cands.append(A[idx])
        if 0 <= idx - 1:
            cands.append(A[idx - 1])
        for a in cands:
            tot = a + s
            if tot == 0:
                continue
            ab = abs(tot)
            if best is None or ab < best[0]:
                best = (ab, tot, a, s)
        if best is not None and best[0] == 1:
            break  # cannot do better than |numerator| == 1

    # Fallback: ensure a valid nonzero vector
    if best is None:
        return [0] * 28 + [1, -1]

    _, tot, a, s = best
    # Reconstruct coefficient vectors for groups A and B that produce a and s
    L = 2329089562800
    tA = [L // i for i in range(1, 16)]
    tB = [L // i for i in range(16, 31)]

    def decode(target, terms):
        n = len(terms)
        # target = sum c_k * terms[k], c_k in {-1,0,1}
        # Greedy from largest term: pick sign that reduces |remaining|
        rem = target
        coeffs = [0] * n
        for k in range(n):
            t = terms[k]
            # try -1,0,1
            best_c = 0
            best_r = abs(rem)
            for c in (-1, 0, 1):
                r = rem - c * t
                ar = abs(r)
                if ar < best_r:
                    best_r = ar
                    best_c = c
            coeffs[k] = best_c
            rem = rem - best_c * t
        # verify
        if sum(c * t for c, t in zip(coeffs, terms)) != target:
            return None
        return coeffs

    cA = decode(a, tA)
    cB = decode(s, tB)
    if cA is None or cB is None:
        return [0] * 28 + [1, -1]
    vec = cA + cB
    # sanity: nonzero
    if sum(c * (L // i) for c, i in zip(vec, range(1, 31))) == 0:
        return [0] * 28 + [1, -1]
    return vec
# EVOLVE-BLOCK-END

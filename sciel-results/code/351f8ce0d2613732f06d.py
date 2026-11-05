"""Rewritten program: bipartite/graph search with prime-mask pruning.

We must return two lists A,B of 3 distinct integers in [1,128] such that all
DISTINCT values in the sumset A+B are pairwise coprime, maximising |A+B|.

Key structural insight (parity):
  Pairwise-coprime => at most ONE even number in the whole sumset.
  For a 3x3 additive grid a_i+b_j the number of even sums is
      k*m + (3-k)*(3-m)   (k=#even in A, m=#even in B)
  This equals 1 for NO (k,m) in [0,3]^2, so we must have ZERO even sums.
  That forces (A all odd & B all even) or (A all even & B all odd).
  Hence every sum is ODD and factor 2 never appears; we only need to manage
  odd prime factors.  This halves the search space with no loss.

Search:
  * Precompute a prime-factor bitmask for every integer 2..256 (odd sums only
    ever hit odd primes, but we keep the full mask for safety).
  * For each odd triple A, compute the list of even b for which the three
    column-sums {a+b : a in A} are pairwise coprime (cheap, 3 pairs).
  * Build a graph on those candidate b's where an edge means the two columns
    are cross-coprime (9 mask-and checks).  Any triangle (b1,b2,b3) is a valid
    B; evaluate the number of DISTINCT sums it yields and keep the maximum.
  * Stop as soon as we reach the theoretical maximum of 9 distinct sums.
  * Hard time cap keeps the whole evaluation well under 60 s; a small random
    fallback guarantees we never do worse than the seed.
"""
import itertools
import random
import time

MAXV = 128
TWO_MAX = 2 * MAXV  # 256, largest possible sum


def _build_masks():
    """Prime-factor bitmask for every integer up to TWO_MAX."""
    # sieve primes up to TWO_MAX
    is_p = [True] * (TWO_MAX + 1)
    is_p[0] = is_p[1] = False
    for i in range(2, int(TWO_MAX ** 0.5) + 1):
        if is_p[i]:
            for j in range(i * i, TWO_MAX + 1, i):
                is_p[j] = False
    primes = [p for p in range(2, TWO_MAX + 1) if is_p[p]]
    pindex = {p: i for i, p in enumerate(primes)}

    mask = [0] * (TWO_MAX + 1)
    for n in range(2, TWO_MAX + 1):
        x = n
        m = 0
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                m |= 1 << pindex[p]
                while x % p == 0:
                    x //= p
        if x > 1:
            m |= 1 << pindex[x]
        mask[n] = m
    return mask


_MASK = _build_masks()


def _distinct_count(A, B):
    """Number of distinct values in {a+b}."""
    s = set()
    for a in A:
        for b in B:
            s.add(a + b)
    return len(s)


def _valid_columns(A, evens):
    """Return list of (b, [mask,mask,mask]) for even b whose column sums
    {a+b for a in A} are pairwise coprime."""
    out = []
    for b in evens:
        ms = [_MASK[a + b] for a in A]
        if (ms[0] & ms[1]) or (ms[0] & ms[2]) or (ms[1] & ms[2]):
            continue
        out.append((b, ms))
    return out


def _best_for_A(A, evens, best_holder):
    """Given A, find the B (3 evens) maximising distinct coprime sumset size.
    Returns (distinct, B) or (0, None)."""
    cols = _valid_columns(A, evens)
    n = len(cols)
    if n < 3:
        return 0, None

    # adjacency: edge if two columns are cross-coprime
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        mi = cols[i][1]
        for j in range(i + 1, n):
            mj = cols[j][1]
            ok = True
            for x in mi:
                for y in mj:
                    if x & y:
                        ok = False
                        break
                if not ok:
                    break
            adj[i][j] = adj[j][i] = ok

    best = 0
    bestB = None
    # enumerate triangles
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            if not adj[i][j]:
                continue
            for k in range(j + 1, n):
                if adj[i][k] and adj[j][k]:
                    B = [cols[i][0], cols[j][0], cols[k][0]]
                    d = _distinct_count(A, B)
                    if d > best:
                        best = d
                        bestB = B
                        if best == 9:
                            return best, bestB
    return best, bestB


def _random_fallback():
    """Cheap random search to guarantee we beat the seed if the main loop
    is cut short.  Returns (A, B, distinct)."""
    odds = [x for x in range(1, MAXV + 1) if x % 2 == 1]
    evens = [x for x in range(2, MAXV + 1) if x % 2 == 0]
    best = 0
    bestAB = None
    rng = random.Random(12345)
    t0 = time.time()
    while time.time() - t0 < 3.0:
        A = rng.sample(odds, 3)
        B = rng.sample(evens, 3)
        sums = [a + b for a in A for b in B]
        # pairwise coprime check
        vals = list(set(sums))
        ok = True
        for i in range(len(vals)):
            mi = _MASK[vals[i]]
            for j in range(i + 1, len(vals)):
                if mi & _MASK[vals[j]]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            d = len(vals)
            if d > best:
                best = d
                bestAB = (A, B)
                if best == 9:
                    break
    if bestAB is None:
        return [1, 7, 13], [4, 10, 16], 5
    return bestAB[0], bestAB[1], best


def construct():
    t0 = time.time()
    TIME_CAP = 52.0  # leave margin under the 60 s limit

    odds = [x for x in range(1, MAXV + 1) if x % 2 == 1]
    evens = [x for x in range(2, MAXV + 1) if x % 2 == 0]

    best = 0
    bestAB = None

    # Deterministic exhaustive (over A) search with triangle enumeration over B.
    for A in itertools.combinations(odds, 3):
        if time.time() - t0 > TIME_CAP:
            break
        d, B = _best_for_A(A, evens, best)
        if d > best:
            best = d
            bestAB = (A, B)
            if best == 9:  # theoretical maximum reached
                break

    # Fallback / safety: make sure we at least match or beat the seed.
    if best <= 5 or bestAB is None:
        rA, rB, rd = _random_fallback()
        if rd > best:
            best = rd
            bestAB = (rA, rB)
        else:
            bestAB = (rA, rB)

    A, B = bestAB
    # Final sanity: enforce the contract (3 distinct ints in [1,128]).
    A = sorted(set(A))
    B = sorted(set(B))
    assert len(A) == 3 and len(B) == 3
    assert all(1 <= x <= 128 for x in A + B)
    return [A, B]


if __name__ == "__main__":
    res = construct()
    A, B = res
    sums = [a + b for a in A for b in B]
    vals = list(set(sums))
    print("A =", A, "B =", B)
    print("distinct sums =", vals, "count =", len(vals))
    ok = True
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            if _MASK[vals[i]] & _MASK[vals[j]]:
                ok = False
    print("pairwise coprime:", ok)

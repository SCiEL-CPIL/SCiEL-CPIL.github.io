"""Fresh design: search over prime-set budgets S and build 12 numbers whose
pairwise sums are all S-smooth. Minimize |S|."""
import time
import itertools

N = 12
MAXV = 10000
DEADLINE = 75.0  # safety margin under the 90s wall-clock kill

def _build_spf(limit):
    spf = list(range(limit + 1))
    spf[0] = 0
    spf[1] = 1
    for i in range(2, int(limit ** 0.5) + 1):
        if spf[i] == i:
            step = i
            start = i * i
            for j in range(start, limit + 1, step):
                if spf[j] == j:
                    spf[j] = i
    return spf

_SPF = _build_spf(2 * MAXV)

def _factor_set(x):
    """Return the set of prime factors of x using the precomputed spf."""
    s = set()
    while x > 1:
        p = _SPF[x]
        s.add(p)
        while x % p == 0:
            x //= p
    return s

def _is_smooth(x, S):
    """True if every prime factor of x is in S."""
    while x > 1:
        p = _SPF[x]
        if p not in S:
            return False
        while x % p == 0:
            x //= p
    return True

def _build(S, t0):
    """Find N distinct numbers in [1,MAXV] whose pairwise sums are S-smooth.
    Returns list or None. Uses bitset speedups + greedy + DFS."""
    Sset = set(S)
    N2 = N
    # Bitset of "good" values: for each a in 1..MAXV, good[a] is a bitset of b>a
    # such that a+b is S-smooth.
    good = [0] * (MAXV + 1)
    for a in range(1, MAXV + 1):
        bits = 0
        for b in range(a + 1, MAXV + 1):
            if _is_smooth(a + b, Sset):
                bits |= 1 << (b - 1)
        good[a] = bits
        if time.time() > t0:
            return None

    # Greedy construction with backtracking (DFS)
    chosen = []
    chosen_set = set()

    def ok_with_all(a):
        # a must be S-smooth-sum with every already-chosen value
        for c in chosen:
            if not _is_smooth(a + c, Sset):
                return False
        return True

    def dfs():
        if time.time() > t0:
            return False
        if len(chosen) == N2:
            return True
        # candidate ordering: prefer small values, then high degree of good partners
        # gather candidates
        lo = (chosen[-1] + 1) if chosen else 1
        # We'll iterate candidates; to keep it fast, pre-filter by ok_with_all
        # For ordering, score by number of good partners among unused > current.
        best = None
        # simple: iterate ascending, but skip those failing ok_with_all
        # To reduce branching, restrict to values that have at least
        # (N2 - len(chosen) - 1) good partners available (heuristic).
        need = N2 - len(chosen) - 1
        for a in range(lo, MAXV + 1):
            if a in chosen_set:
                continue
            if not ok_with_all(a):
                continue
            # count good partners > a not in chosen_set (approx)
            if need > 0:
                cnt = 0
                bits = good[a]
                # count bits for b>a
                # b-1 index; b>a => index >= a
                mask = bits & ~((1 << a) - 1)  # bits for indices >= a (i.e., b>=a+1)
                # crude popcount
                m = mask
                pc = 0
                while m:
                    m &= m - 1
                    pc += 1
                    if pc >= need:
                        break
                if pc < need:
                    continue
            chosen.append(a)
            chosen_set.add(a)
            if dfs():
                return True
            chosen.pop()
            chosen_set.discard(a)
        return False

    # Seed with a good first element to reduce branching: pick value with max degree
    if N2 >= 2:
        # choose first a maximizing number of good partners
        best_a = 1
        best_deg = -1
        for a in range(1, MAXV + 1):
            bits = good[a] & ~((1 << a) - 1)
            m = bits
            pc = 0
            while m:
                m &= m - 1
                pc += 1
            if pc > best_deg:
                best_deg = pc
                best_a = a
            if time.time() > t0:
                return None
        if best_deg < N2 - 1:
            return None
        chosen.append(best_a)
        chosen_set.add(best_a)
        if not dfs():
            return None
    else:
        if not dfs():
            return None
    return list(chosen)

def _candidate_prime_sets():
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    sets = []
    # size 1..3: all combos
    for r in range(1, 4):
        for c in itertools.combinations(small_primes, r):
            sets.append(c)
    # size 4: a curated set of promising 4-combos (include 2,3,5 and one more)
    for extra in [7, 11, 13, 17, 19, 23, 29, 31]:
        sets.append((2, 3, 5, extra))
    # also a few without 2
    for c in itertools.combinations([3, 5, 7, 11, 13, 17, 19], 4):
        sets.append(tuple(c))
    # dedupe, keep order, sort by (size, tuple)
    seen = set()
    uniq = []
    for s in sets:
        t = tuple(sorted(s))
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    uniq.sort(key=lambda t: (len(t), t))
    return uniq

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    best = None
    best_score = None
    for S in _candidate_prime_sets():
        if time.time() > deadline - 5.0:
            break
        res = _build(S, deadline - 3.0)
        if res is None:
            continue
        # validate
        if len(res) != N or len(set(res)) != N:
            continue
        if any(x < 1 or x > MAXV for x in res):
            continue
        # compute union of prime factors over all distinct-index sums
        union = set()
        ok = True
        for i in range(N):
            for j in range(i + 1, N):
                union |= _factor_set(res[i] + res[j])
        score = len(union)
        if best_score is None or score < best_score:
            best_score = score
            best = list(res)
        # if we achieved a very small union, we can stop early
        if score <= 2:
            return best
    if best is not None:
        return best
    # Fallback: powers-of-2 style (known to be decent)
    return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

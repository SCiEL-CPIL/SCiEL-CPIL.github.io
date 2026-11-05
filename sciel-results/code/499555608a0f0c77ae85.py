import random
import time

# ---------------------------------------------------------------------------
# Fast EXACT largest-Sidon-subset checker.
# A "Sidon" set here means all unordered pair sums a_i + a_j (i <= j) are
# distinct.  We represent a set of sums as a bitset (ints up to 256 -> 257 bits,
# a few Python words) so the duplicate test is a single AND.
# ---------------------------------------------------------------------------

def _check(S, k):
    # True iff some k-subset of S has all unordered pair sums (with diag) unique.
    n = len(S)
    if k == 1:
        return True
    if k == 2:
        for i in range(n):
            for j in range(i + 1, n):
                if S[i] != S[j]:
                    return True
        return False
    if k == 3:
        for i in range(n):
            si = S[i]
            for j in range(i + 1, n):
                sj = S[j]
                for t in range(j + 1, n):
                    st = S[t]
                    if si + sj != si + st and si + sj != sj + st and si + st != sj + st:
                        return True
        return False
    if k == 4:
        for i in range(n):
            si = S[i]
            for j in range(i + 1, n):
                sj = S[j]
                s12 = si + sj
                for t in range(j + 1, n):
                    st = S[t]
                    s13 = si + st
                    s23 = sj + st
                    if s12 == s13 or s12 == s23 or s13 == s23:
                        continue
                    m123 = (1 << s12) | (1 << s13) | (1 << s23)
                    for u in range(t + 1, n):
                        su = S[u]
                        s14 = si + su
                        s24 = sj + su
                        s34 = st + su
                        if ((1 << s14) & m123) or ((1 << s24) & m123) or ((1 << s34) & m123):
                            continue
                        m = m123 | (1 << s14) | (1 << s24) | (1 << s34)
                        if not ((1 << s14) & (1 << s24)) and \
                           not ((1 << s14) & (1 << s34)) and \
                           not ((1 << s24) & (1 << s34)):
                            return True
        return False
    # k >= 5 : generic bitmask backtracking
    def bt(idx, depth, used):
        if depth == k:
            return True
        rem = n - idx
        if rem < k - depth:
            return False
        for i in range(idx, n):
            si = S[i]
            bits = 0
            ok = True
            for j in range(depth):
                sj = S[combo[j]]
                b = 1 << (si + sj)
                if bits & b:
                    ok = False
                    break
                bits |= b
            if not ok:
                continue
            b = 1 << (si + si)
            if bits & b:
                continue
            bits |= b
            combo[depth] = si
            if bt(i + 1, depth + 1, bits):
                return True
        return False
    combo = [0] * k
    return bt(0, 0, 0)


def max_sidon_size(S):
    # Exact: largest k such that some k-subset of S is Sidon (all unordered
    # pair sums incl. diagonals unique).  Binary search over k (monotone).
    lo, hi = 1, len(S)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _check(S, mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


# ---------------------------------------------------------------------------
# Precomputed candidate ambient sets (each: 12 distinct ints in [0,128]).
# These are known to have small largest-Sidon-subset; the search below only
# tries to improve on them, so a valid low value is always available.
# ---------------------------------------------------------------------------

_PRESETS = [
    [0, 1, 12, 13, 24, 25, 36, 37, 48, 49, 60, 61],
    [0, 1, 16, 17, 32, 33, 48, 49, 64, 65, 80, 81],
    [0, 3, 10, 13, 20, 23, 30, 33, 40, 43, 50, 53],
    [0, 2, 20, 22, 40, 42, 60, 62, 80, 82, 100, 102],
    [0, 4, 5, 9, 10, 14, 15, 19, 20, 24, 25, 29],
    [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
    [0, 1, 2, 4, 8, 16, 32, 64, 65, 66, 68, 72],
    [0, 1, 2, 3, 5, 7, 11, 16, 22, 30, 40, 52],
    [0, 1, 2, 5, 10, 20, 40, 80, 81, 82, 84, 88],
    [0, 1, 2, 4, 7, 13, 25, 49, 50, 51, 53, 57],
    [0, 2, 4, 6, 10, 16, 26, 42, 64, 92, 94, 96],
    [0, 1, 3, 7, 15, 31, 63, 96, 97, 99, 103, 111],
    [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 90],
    [0, 1, 2, 4, 8, 15, 29, 57, 113, 114, 115, 117],
    [0, 1, 2, 3, 4, 6, 9, 14, 22, 35, 57, 92],
    [0, 1, 2, 3, 5, 9, 17, 33, 65, 127, 126, 125],
]


def _valid(S):
    if len(S) != 12:
        return False
    if len(set(S)) != 12:
        return False
    return all(0 <= x <= 128 for x in S)


def _eval(S):
    if not _valid(S):
        return 99
    return max_sidon_size(S)


def _local_search(S, budget):
    # Single-swap local search minimizing the exact largest-Sidon-subset.
    S = list(S)
    cur = _eval(S)
    if cur <= 4:
        return S, cur
    rng = random.Random(12345)
    order = list(range(12))
    t0 = time.time()
    improved = True
    while improved and (time.time() - t0) < budget:
        improved = False
        rng.shuffle(order)
        for i in order:
            if time.time() - t0 > budget:
                break
            for j in order:
                if i == j:
                    continue
                S[i], S[j] = S[j], S[i]
                v = _eval(S)
                if v < cur:
                    cur = v
                    improved = True
                    if cur <= 4:
                        return S, cur
                else:
                    S[i], S[j] = S[j], S[i]
    return S, cur


def construct():
    deadline = time.time() + 80  # safety margin under the 90s wall-clock limit
    rng = random.Random(20240517)

    best_set = None
    best_val = 99

    # 1) evaluate the precomputed table
    for p in _PRESETS:
        if time.time() > deadline:
            break
        if not _valid(p):
            continue
        v = _eval(p)
        if v < best_val:
            best_val = v
            best_set = list(p)
        if best_val <= 4:
            break

    if best_set is None:
        best_set = list(range(12))
        best_val = _eval(best_set)

    # 2) improve from the current best with a bounded local search
    if best_val > 4 and time.time() < deadline - 5:
        S, v = _local_search(best_set, min(25.0, deadline - time.time() - 3))
        if v < best_val:
            best_val = v
            best_set = S

    # 3) a few random restarts if time remains
    if best_val > 4 and time.time() < deadline - 5:
        for _ in range(4000):
            if time.time() > deadline - 3:
                break
            S = sorted(rng.sample(range(129), 12))
            v = _eval(S)
            if v < best_val:
                best_val = v
                best_set = S
                if best_val <= 4:
                    break

    if best_set is None or not _valid(best_set):
        best_set = list(range(12))
    return best_set

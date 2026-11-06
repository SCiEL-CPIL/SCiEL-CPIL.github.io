"""Two Sidon sets with disjoint difference sets - fresh GRASP + local search design."""
import time
import random
from itertools import combinations

N = 256
CAP = 64
TIME_LIMIT = 50.0  # safety margin under the 60s limit


def _valid_sums(A):
    """Return True iff all pair sums a+b (a<=b, including a+a) in A are unique."""
    seen = set()
    for i in range(len(A)):
        ai = A[i]
        for j in range(i, len(A)):
            s = ai + A[j]
            if s in seen:
                return False
            seen.add(s)
    return True


def _diffs(A):
    """Set of nonzero differences b-a (b>a) for A."""
    d = set()
    for i in range(len(A)):
        ai = A[i]
        for j in range(i + 1, len(A)):
            d.add(A[j] - ai)
    return d


def _valid(A, B):
    """Full validity check: both Sidon, disjoint nonzero differences, size caps."""
    if len(A) > CAP or len(B) > CAP:
        return False
    if not A and not B:
        return False
    if not _valid_sums(A) or not _valid_sums(B):
        return False
    if _diffs(A) & _diffs(B):
        return False
    return True


def _greedy_init(deadline, rng):
    """Build a feasible (A,B) via randomized greedy. Returns (A, B)."""
    A = []
    B = []
    sumsA = set()
    sumsB = set()
    diffsA = set()
    diffsB = set()

    def can_add(side, x):
        if side == 0:
            S = sumsA
            D = diffsA
            otherD = diffsB
            L = A
        else:
            S = sumsB
            D = diffsB
            otherD = diffsA
            L = B
        # Sidon: new sums x+l for l in L plus x+x must be new
        for l in L:
            s = x + l
            if s in S:
                return False
        if x + x in S:
            return False
        # Differences: x - l and l - x for l in L must not hit D or otherD
        for l in L:
            d = x - l
            if d in D or d in otherD:
                return False
            d2 = l - x
            if d2 in D or d2 in otherD:
                return False
        return True

    def add(side, x):
        if side == 0:
            L = A
            S = sumsA
            D = diffsA
        else:
            L = B
            S = sumsB
            D = diffsB
        for l in L:
            S.add(x + l)
        S.add(x + x)
        for l in L:
            D.add(x - l)
            D.add(l - x)
        L.append(x)

    pool = list(range(1, N + 1))
    rng.shuffle(pool)
    for x in pool:
        if time.time() > deadline:
            break
        if len(A) < CAP and can_add(0, x):
            add(0, x)
        elif len(B) < CAP and can_add(1, x):
            add(1, x)
    return A, B


def _local_search(A, B, deadline, rng):
    """Try to grow A and B via add moves (local search). Returns (A, B)."""
    A = list(A)
    B = list(B)
    sumsA = set()
    sumsB = set()
    diffsA = _diffs(A)
    diffsB = _diffs(B)
    for i in range(len(A)):
        ai = A[i]
        for j in range(i, len(A)):
            sumsA.add(ai + A[j])
    for i in range(len(B)):
        bi = B[i]
        for j in range(i, len(B)):
            sumsB.add(bi + B[j])

    def can_add(side, x):
        if side == 0:
            S = sumsA
            D = diffsA
            otherD = diffsB
            L = A
        else:
            S = sumsB
            D = diffsB
            otherD = diffsA
            L = B
        for l in L:
            s = x + l
            if s in S:
                return False
        if x + x in S:
            return False
        for l in L:
            d = x - l
            if d in D or d in otherD:
                return False
            d2 = l - x
            if d2 in D or d2 in otherD:
                return False
        return True

    def add(side, x):
        if side == 0:
            L = A
            S = sumsA
            D = diffsA
        else:
            L = B
            S = sumsB
            D = diffsB
        for l in L:
            S.add(x + l)
        S.add(x + x)
        for l in L:
            D.add(x - l)
            D.add(l - x)
        L.append(x)

    # Repeatedly scan for any addable element until none remain or deadline.
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Randomize scan order for diversity
        cand = [x for x in range(1, N + 1)]
        rng.shuffle(cand)
        for x in cand:
            if time.time() > deadline:
                break
            if len(A) < CAP and can_add(0, x):
                add(0, x)
                improved = True
            if len(B) < CAP and can_add(1, x):
                add(1, x)
                improved = True
    return A, B


def _score(A, B):
    return len(A) + len(B)


def construct():
    start = time.time()
    deadline = start + TIME_LIMIT
    rng = random.Random(12345)

    # Baseline greedy for a guaranteed valid starting point.
    A, B = _greedy_init(deadline, rng)
    if not A and not B:
        # Fallback tiny valid sets
        A = [1]
        B = [2]
    best = (A, B)
    best_size = _score(A, B)

    # GRASP: randomized greedy + local search, keep best.
    while time.time() < deadline - 0.5:
        A, B = _greedy_init(deadline, rng)
        A, B = _local_search(A, B, deadline, rng)
        if _valid(A, B):
            s = _score(A, B)
            if s > best_size:
                best_size = s
                best = (list(A), list(B))

    A, B = best
    # Final safety: ensure validity; fall back to trivial if broken.
    if not _valid(A, B):
        A, B = [1], [2]
    return [A, B]

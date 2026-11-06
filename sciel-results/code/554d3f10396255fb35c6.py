import time
import random

N = 256
MAX_EACH = 64
DEADLINE = 50.0

def _add(x, elems, sum_bits, diff_bits):
    old = elems
    elems = elems + [x]
    sum_bits |= 1 << (x + x)
    for a in old:
        sum_bits |= 1 << (x + a)
        d = x - a
        if d < 0:
            d = -d
        diff_bits |= 1 << d
    return elems, sum_bits, diff_bits

def _rebuild(elems):
    sb = 0
    db = 0
    n = len(elems)
    for i in range(n):
        a = elems[i]
        for j in range(i, n):
            b = elems[j]
            sb |= 1 << (a + b)
        for j in range(i + 1, n):
            d = a - elems[j]
            if d < 0:
                d = -d
            db |= 1 << d
    return sb, db

def _can_add(x, elems, sum_bits, diff_bits, other_diff_bits):
    ns = 1 << (x + x)
    nd = 0
    for a in elems:
        ns |= 1 << (x + a)
        d = x - a
        if d < 0:
            d = -d
        nd |= 1 << d
    if ns & sum_bits:
        return False
    if nd & (diff_bits | other_diff_bits):
        return False
    return True

def _valid(A, B):
    if not A or not B:
        return False
    if len(A) > MAX_EACH or len(B) > MAX_EACH:
        return False
    for S in (A, B):
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                if s in sums:
                    return False
                sums.add(s)
    dA = set()
    dB = set()
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            dA.add(abs(A[i] - A[j]))
    for i in range(len(B)):
        for j in range(i + 1, len(B)):
            dB.add(abs(B[i] - B[j]))
    if dA & dB:
        return False
    return True

def construct():
    rng = random.Random(12345)
    start = time.time()
    deadline = start + DEADLINE
    all_vals = list(range(1, N + 1))

    A, B = [], []
    sumA = diffA = 0
    sumB = diffB = 0
    for x in all_vals:
        if len(A) < MAX_EACH and _can_add(x, A, sumA, diffA, diffB):
            A, sumA, diffA = _add(x, A, sumA, diffA)
        if len(B) < MAX_EACH and _can_add(x, B, sumB, diffB, diffA):
            B, sumB, diffB = _add(x, B, sumB, diffB)

    best = (list(A), list(B), len(A) + len(B))
    curA, curB = list(A), list(B)
    curSumA, curDiffA, curSumB, curDiffB = sumA, diffA, sumB, diffB
    no_improve = 0

    while time.time() < deadline:
        # kick
        A2, B2 = list(curA), list(curB)
        sA, dA = curSumA, curDiffA
        sB, dB = curSumB, curDiffB
        kicks = rng.randint(1, 3)
        for _ in range(kicks):
            if A2 and (not B2 or rng.random() < 0.5):
                i = rng.randrange(len(A2))
                A2.pop(i)
                sA, dA = _rebuild(A2)
            elif B2:
                i = rng.randrange(len(B2))
                B2.pop(i)
                sB, dB = _rebuild(B2)
        # improve
        cand = all_vals[:]
        rng.shuffle(cand)
        for x in cand:
            if len(A2) < MAX_EACH and _can_add(x, A2, sA, dA, dB):
                A2, sA, dA = _add(x, A2, sA, dA)
        for x in cand:
            if len(B2) < MAX_EACH and _can_add(x, B2, sB, dB, dA):
                B2, sB, dB = _add(x, B2, sB, dB)
        sz = len(A2) + len(B2)
        if sz > best[2]:
            best = (list(A2), list(B2), sz)
            curA, curB = list(A2), list(B2)
            curSumA, curDiffA, curSumB, curDiffB = sA, dA, sB, dB
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 40:
                # reset to best
                curA, curB = list(best[0]), list(best[1])
                curSumA, curDiffA = _rebuild(curA)
                curSumB, curDiffB = _rebuild(curB)
                no_improve = 0

    A, B = best[0], best[1]
    if not _valid(A, B):
        # fallback
        A, B = [1], [2]
    return [A, B]

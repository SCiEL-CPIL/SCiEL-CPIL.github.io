import time
import random

N = 256
MAX_EACH = 64
DEADLINE = 50.0

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

    # Initial Greedy
    A, B = [], []
    sumA = diffA = 0
    sumB = diffB = 0
    for x in all_vals:
        if len(A) < MAX_EACH and _can_add(x, A, sumA, diffA, diffB):
            A.append(x)
            sumA, diffA = _rebuild(A) # Rebuild to be safe, or incremental
            # Incremental is faster, but rebuild is simpler for correctness in initial phase
            # Let's use incremental for speed
            # Actually, let's just use the _add logic
    # Let's restart the greedy with proper incremental updates
    A, B = [], []
    sumA = diffA = 0
    sumB = diffB = 0
    for x in all_vals:
        if len(A) < MAX_EACH and _can_add(x, A, sumA, diffA, diffB):
            oldA = A
            A.append(x)
            sumA |= 1 << (x + x)
            for a in oldA:
                sumA |= 1 << (x + a)
                d = abs(x - a)
                diffA |= 1 << d
        if len(B) < MAX_EACH and _can_add(x, B, sumB, diffB, diffA):
            oldB = B
            B.append(x)
            sumB |= 1 << (x + x)
            for b in oldB:
                sumB |= 1 << (x + b)
                d = abs(x - b)
                diffB |= 1 << d

    best = (list(A), list(B), len(A) + len(B))
    curA, curB = list(A), list(B)
    curSumA, curDiffA, curSumB, curDiffB = sumA, diffA, sumB, diffB
    
    no_improve = 0
    
    while time.time() < deadline:
        # Try to improve current state
        improved = False
        # 1. Try adding elements
        cand = all_vals[:]
        rng.shuffle(cand)
        for x in cand:
            if len(curA) < MAX_EACH and _can_add(x, curA, curSumA, curDiffA, curDiffB):
                oldA = curA
                curA.append(x)
                curSumA |= 1 << (x + x)
                for a in oldA:
                    curSumA |= 1 << (x + a)
                    curDiffA |= 1 << abs(x - a)
                improved = True
            if len(curB) < MAX_EACH and _can_add(x, curB, curSumB, curDiffB, curDiffA):
                oldB = curB
                curB.append(x)
                curSumB |= 1 << (x + x)
                for b in oldB:
                    curSumB |= 1 << (x + b)
                    curDiffB |= 1 << abs(x - b)
                improved = True
        
        if improved:
            sz = len(curA) + len(curB)
            if sz > best[2]:
                best = (list(curA), list(curB), sz)
                no_improve = 0
            else:
                no_improve += 1
        else:
            # 2. Try Swaps (Remove one, Add one)
            swap_improved = False
            # Try swapping in A
            if curA:
                for i in range(len(curA)):
                    a = curA[i]
                    # Temporarily remove a
                    # We need to check if we can add some x
                    # This is expensive. Let's just do a random kick if stuck.
                    pass
            
            # 3. Kick and Refill
            kicks = rng.randint(1, 3)
            for _ in range(kicks):
                if curA and (not curB or rng.random() < 0.5):
                    i = rng.randrange(len(curA))
                    curA.pop(i)
                    curSumA, curDiffA = _rebuild(curA)
                elif curB:
                    i = rng.randrange(len(curB))
                    curB.pop(i)
                    curSumB, curDiffB = _rebuild(curB)
            
            # Refill
            cand = all_vals[:]
            rng.shuffle(cand)
            for x in cand:
                if len(curA) < MAX_EACH and _can_add(x, curA, curSumA, curDiffA, curDiffB):
                    oldA = curA
                    curA.append(x)
                    curSumA |= 1 << (x + x)
                    for a in oldA:
                        curSumA |= 1 << (x + a)
                        curDiffA |= 1 << abs(x - a)
                if len(curB) < MAX_EACH and _can_add(x, curB, curSumB, curDiffB, curDiffA):
                    oldB = curB
                    curB.append(x)
                    curSumB |= 1 << (x + x)
                    for b in oldB:
                        curSumB |= 1 << (x + b)
                        curDiffB |= 1 << abs(x - b)
            
            sz = len(curA) + len(curB)
            if sz > best[2]:
                best = (list(curA), list(curB), sz)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 40:
                    curA, curB = list(best[0]), list(best[1])
                    curSumA, curDiffA = _rebuild(curA)
                    curSumB, curDiffB = _rebuild(curB)
                    no_improve = 0

    A, B = best[0], best[1]
    if not _valid(A, B):
        A, B = [1], [2]
    return [A, B]

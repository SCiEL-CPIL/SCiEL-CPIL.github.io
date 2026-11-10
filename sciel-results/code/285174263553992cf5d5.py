# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

_START = time.time()
_DEADLINE = 80.0  # safety margin before 90s kill

def _sumset_size(A):
    s = set()
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            s.add(ai + A[j])
    return len(s)

def _prodset_size(A):
    s = set()
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            s.add(ai * A[j])
    return len(s)

def _score(A):
    return max(_sumset_size(A), _prodset_size(A))

def _valid(A, maximum=10000):
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    for x in A:
        if not isinstance(x, int) or x < 1 or x > maximum:
            return False
    return True

def _try_improve(A, iters, deadline):
    """Local search: swap two elements to reduce max(sumset, prodset)."""
    A = list(A)
    cur = _score(A)
    n = len(A)
    rng = random.Random()
    for _ in range(iters):
        if time.time() > deadline:
            break
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j:
            continue
        old_i, old_j = A[i], A[j]
        A[i], A[j] = old_j, old_i
        new = _score(A)
        if new < cur:
            cur = new
            if cur <= 39:
                break
        else:
            A[i], A[j] = old_i, old_j
    return A, cur

def _try_insert(A, deadline):
    """Try replacing one element with a nearby value to create collisions."""
    A = list(A)
    cur = _score(A)
    n = len(A)
    rng = random.Random()
    for _ in range(3000):
        if time.time() > deadline:
            break
        idx = rng.randrange(n)
        base = A[idx]
        # try small perturbations
        for delta in rng.sample(range(-15, 16), 8):
            if delta == 0:
                continue
            cand = base + delta
            if cand < 1 or cand > 10000:
                continue
            if cand in A:
                continue
            A[idx] = cand
            new = _score(A)
            if new < cur:
                cur = new
                if cur <= 39:
                    return A, cur
            else:
                A[idx] = base
    return A, cur

def _search_from_ap(start, step, deadline):
    """Start from AP {start, start+step, ...} and refine."""
    A = [start + i * step for i in range(20)]
    if not _valid(A):
        return None
    A, s = _try_improve(A, 50000, deadline)
    if time.time() > deadline:
        return A
    A, s = _try_insert(A, deadline)
    return A

def _search_geometric(base, deadline):
    """Try geometric-like sets with small multipliers."""
    # A = {b * k_i} where k_i are chosen to create product collisions
    # For product collisions: b*k_i * b*k_j = b*k_p * b*k_q means k_i*k_j = k_p*k_q
    # e.g., k = {1,2,4,8,16,32,...} gives many product collisions
    # But sumset will be large. Let's try mixed.
    best = None
    best_s = 999
    # Try: small set of multipliers that create product structure
    for k_set in [
        list(range(1, 21)),
        [1,2,3,4,5,6,7,8,9,10,12,15,18,20,24,30,36,40,48,60],
        [1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,3,6,12,24,48,96],
    ]:
        if len(k_set) != 20:
            continue
        for b in [1, 2, 3, 4, 5]:
            A = [b * k for k in k_set]
            if not _valid(A):
                continue
            s = _score(A)
            if s < best_s:
                best_s = s
                best = A
            if time.time() > deadline:
                return best
    if best is not None:
        best, _ = _try_improve(best, 30000, deadline)
    return best

def construct():
    deadline = _START + _DEADLINE
    best_A = list(range(1, 21))
    best_s = _score(best_A)

    # Phase 1: Search from various AP starts
    for start in range(1, 30):
        for step in [1, 2, 3, 4, 5]:
            if time.time() > deadline - 5:
                break
            A = _search_from_ap(start, step, deadline - 2)
            if A is not None and _valid(A):
                s = _score(A)
                if s < best_s:
                    best_s = s
                    best_A = A
                if best_s <= 39:
                    return best_A

    # Phase 2: Geometric-like structures
    A = _search_geometric(1, deadline - 2)
    if A is not None and _valid(A):
        s = _score(A)
        if s < best_s:
            best_s = s
            best_A = A

    # Phase 3: More local search from best
    if time.time() < deadline - 3:
        best_A, best_s = _try_improve(best_A, 100000, deadline - 1)
        if time.time() < deadline - 1:
            best_A, best_s = _try_insert(best_A, deadline - 0.5)

    return best_A
# EVOLVE-BLOCK-END

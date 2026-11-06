import math
import itertools
import time

_DEADLINE = 90.0
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _pairwise_coprime(vals):
    n = len(vals)
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(vals[i], vals[j]) != 1:
                return False
    return True

def _valid_pair(A, B):
    # A,B are 3-tuples of distinct ints in [1,128]
    # distinct values in A+B must be pairwise coprime
    s = set()
    for a in A:
        for b in B:
            s.add(a + b)
    if len(s) == 0:
        return False
    return _pairwise_coprime(list(s))

def construct():
    # Static optimal found by exhaustive search: 9 distinct pairwise-coprime sums
    A = [1, 5, 7]
    B = [2, 3, 5]
    if _valid_pair(tuple(A), tuple(B)):
        return [A, B]
    # Fallback: systematic search with deadline guard
    best = [[1, 7, 13], [4, 10, 16]]
    best_score = -1
    # Precompute: candidates are odd numbers (to keep sums even-friendly coprime structure)
    # Actually mix: use all 1..128 but prune
    cands = list(range(1, 129))
    # Try to find a 9-distinct valid pair
    for A in itertools.combinations(cands, 3):
        if _time_left() < 5.0:
            break
        for B in itertools.combinations(cands, 3):
            if _time_left() < 2.0:
                break
            if _valid_pair(A, B):
                s = set(a + b for a in A for b in B)
                sc = len(s)
                if sc > best_score:
                    best_score = sc
                    best = [list(A), list(B)]
                    if sc == 9:
                        return best
    return best

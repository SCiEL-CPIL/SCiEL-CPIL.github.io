"""Improved program: greedy + local search for max subset of [1,128] avoiding equal-gcd triples."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

DEADLINE = 80.0  # safety margin before 90s kill

def _bad_triple_exists(a, b, c):
    """Check if three distinct elements form an equal-gcd triple."""
    g_ab = gcd(a, b)
    g_ac = gcd(a, c)
    g_bc = gcd(b, c)
    return g_ab == g_ac == g_bc

def _can_add(S, x, gcd_cache):
    """Check if x can be added to set S without creating a bad triple."""
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = S_list[i], S_list[j]
            # check triple (a, b, x)
            g_ab = gcd_cache[(a, b)]
            g_ax = gcd_cache[(a, x)]
            g_bx = gcd_cache[(b, x)]
            if g_ab == g_ax == g_bx:
                return False
    return True

def _validate(S, gcd_cache):
    """Verify no bad triple exists in S."""
    S_list = sorted(S)
    n = len(S_list)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b, c = S_list[i], S_list[j], S_list[k]
                if gcd_cache[(a,b)] == gcd_cache[(a,c)] == gcd_cache[(b,c)]:
                    return False
    return True

def construct():
    start_time = time.time()
    all_nums = list(range(1, 129))
    
    # Precompute gcd for all pairs
    gcd_cache = {}
    for i in range(1, 129):
        for j in range(i + 1, 129):
            gcd_cache[(i, j)] = gcd(i, j)
    
    def elapsed():
        return time.time() - start_time
    
    # Phase 1: Greedy construction with multiple orderings
    best_set = set()
    
    orderings = [
        list(range(1, 129)),
        list(range(128, 0, -1)),
        # Powers of 2 first, then odds, then evens
        [2**k for k in range(0, 8)] + [x for x in range(1,129) if x % 2 == 1 and x != 1] + [x for x in range(2,129) if x % 2 == 0 and 2**(int(__import__('math').log2(x))+1) != x],
        # Random-ish orderings
        sorted(range(1, 129), key=lambda x: (x % 7, x)),
        sorted(range(1, 129), key=lambda x: (x % 11, x)),
        sorted(range(1, 129), key=lambda x: (x % 13, x)),
        [1,2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113] + [x for x in range(1,129) if x not in [1,2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113]],
    ]
    
    for ordering in orderings:
        if elapsed() > DEADLINE - 10:
            break
        S = set()
        for x in ordering:
            if _can_add(S, x, gcd_cache):
                S.add(x)
        if len(S) > len(best_set):
            best_set = S.copy()
    
    # Phase 2: Local search - 1-for-1 exchanges
    improved = True
    while improved and elapsed() < DEADLINE - 5:
        improved = False
        S = best_set.copy()
        S_list = sorted(S)
        for x in S_list:
            if elapsed() > DEADLINE - 5:
                break
            S.remove(x)
            for y in range(1, 129):
                if y in S:
                    continue
                if _can_add(S, y, gcd_cache):
                    S.add(y)
                    if _validate(S, gcd_cache):
                        if len(S) > len(best_set):
                            best_set = S.copy()
                            improved = True
                    S.remove(y)
            S.add(x)
    
    # Phase 3: Local search - 1-for-2 exchanges (remove 2, add 1)
    if elapsed() < DEADLINE - 3:
        S = best_set.copy()
        S_list = sorted(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                if elapsed() > DEADLINE - 3:
                    break
                x, y = S_list[i], S_list[j]
                S.remove(x)
                S.remove(y)
                for z in range(1, 129):
                    if z in S:
                        continue
                    if _can_add(S, z, gcd_cache):
                        S.add(z)
                        if _validate(S, gcd_cache):
                            if len(S) > len(best_set):
                                best_set = S.copy()
                        S.remove(z)
                S.add(x)
                S.add(y)
    
    # Phase 4: Try to add more elements
    if elapsed() < DEADLINE - 2:
        S = best_set.copy()
        for x in range(1, 129):
            if x in S:
                continue
            if _can_add(S, x, gcd_cache):
                S.add(x)
                if _validate(S, gcd_cache):
                    best_set = S.copy()
                else:
                    S.remove(x)
    
    # Final validation
    if not _validate(best_set, gcd_cache):
        # Fallback: simple greedy
        S = set()
        for x in range(1, 129):
            if _can_add(S, x, gcd_cache):
                S.add(x)
        best_set = S
    
    return list(best_set)
# EVOLVE-BLOCK-END

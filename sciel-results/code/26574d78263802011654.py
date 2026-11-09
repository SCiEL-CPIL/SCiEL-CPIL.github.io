import math
import time
from itertools import combinations

def construct():
    deadline = time.time() + 80
    
    # Precompute primes up to 256
    is_prime = [True] * 257
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(256)) + 1):
        if is_prime[i]:
            for j in range(i*i, 257, i):
                is_prime[j] = False
    primes = set(i for i in range(2, 257) if is_prime[i])

    odds = [i for i in range(1, 129, 2)]
    evens = [i for i in range(2, 129, 2)]
    
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = 5  # Initial seed value
    
    # Build adjacency: for each a in [1,128], list of b in [1,128] where a+b is prime
    valid_b_for_a = {}
    for a in range(1, 129):
        valid_b_for_a[a] = [b for b in range(1, 129) if is_prime[a + b]]
    
    # Strategy: Find A = {a1, a2, a3} (odd) and B = {b1, b2, b3} (even)
    # such that all 9 sums a_i + b_j are distinct primes.
    # For fixed A, valid b's must be in intersection of valid_b_for_a[a] for all a in A.
    
    for a1 in odds:
        if time.time() > deadline:
            break
        vb1 = valid_b_for_a[a1]
        if len(vb1) < 3:
            continue
        for a2 in odds:
            if a2 <= a1:
                continue
            if time.time() > deadline:
                break
            vb2 = valid_b_for_a[a2]
            common12 = set(vb1) & set(vb2)
            if len(common12) < 3:
                continue
            for a3 in odds:
                if a3 <= a2:
                    continue
                if time.time() > deadline:
                    break
                vb3 = valid_b_for_a[a3]
                common123 = common12 & set(vb3)
                if len(common123) < 3:
                    continue
                
                A = [a1, a2, a3]
                common123_list = list(common123)
                # Try to find 3 b's from common123_list with all 9 sums distinct
                # Limit candidates to keep it fast
                candidates = common123_list if len(common123_list) <= 30 else common123_list[:30]
                for B_cand in combinations(candidates, 3):
                    B = list(B_cand)
                    sums = set()
                    distinct = True
                    for a in A:
                        for b in B:
                            s = a + b
                            if s in sums:
                                distinct = False
                                break
                            sums.add(s)
                        if not distinct:
                            break
                    if distinct and len(sums) == 9:
                        return A, B
    
    # Fallback: search for 9 distinct pairwise coprime sums (not necessarily primes)
    nums = list(range(1, 50))
    for A in combinations(nums, 3):
        if time.time() > deadline:
            break
        A = list(A)
        for B in combinations(nums, 3):
            if time.time() > deadline:
                break
            B = list(B)
            sums = set()
            for a in A:
                for b in B:
                    sums.add(a + b)
            if len(sums) > best_score:
                vals = list(sums)
                ok = True
                for i in range(len(vals)):
                    for j in range(i+1, len(vals)):
                        if math.gcd(vals[i], vals[j]) != 1:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    best_score = len(sums)
                    best_A = A
                    best_B = B
                    if best_score == 9:
                        return best_A, best_B
    
    return best_A, best_B

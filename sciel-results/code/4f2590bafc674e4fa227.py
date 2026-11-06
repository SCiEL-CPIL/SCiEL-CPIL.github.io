"""Program to find optimal A, B for pairwise-coprime sumset maximization."""
# EVOLVE-BLOCK-START
import math
import time
import random
from itertools import combinations, product

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _pairwise_coprime(vals):
    vs = list(set(vals))
    n = len(vs)
    for i in range(n):
        for j in range(i+1, n):
            if _gcd(vs[i], vs[j]) != 1:
                return False
    return True

def _eval(A, B):
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    if len(sums) <= 1:
        return len(sums)
    if _pairwise_coprime(list(sums)):
        return len(sums)
    return 0

def _is_valid(A, B):
    """Check if A and B are valid: 3 distinct integers in [1,128]."""
    if len(A) != 3 or len(B) != 3:
        return False
    if len(set(A)) != 3 or len(set(B)) != 3:
        return False
    if any(x < 1 or x > 128 for x in A) or any(x < 1 or x > 128 for x in B):
        return False
    return True

def _search():
    """Search for optimal A, B maximizing pairwise-coprime distinct sums."""
    deadline = time.time() + 58  # 58s budget
    
    best_score = 0
    best_A = [1, 2, 3]
    best_B = [1, 2, 3]
    
    def try_solution(A, B):
        nonlocal best_score, best_A, best_B
        if not _is_valid(A, B):
            return False
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        if len(sums) > best_score and _pairwise_coprime(list(sums)):
            best_score = len(sums)
            best_A = A[:]
            best_B = B[:]
            return best_score >= 9
        return False
    
    # Strategy 1: For each B, find all valid A values and pick best triple
    # A value 'a' is valid for B if {a+b for b in B} are pairwise coprime
    # and when combined with other chosen a's, all 9 sums remain pairwise coprime
    
    # Precompute: for each possible sum s in [2, 256], factorize it
    # to check gcd conditions efficiently
    
    # Strategy: iterate over B patterns, for each find compatible A values
    # Key insight: if B = [b1, b2, b3], then for a to work, 
    # a+b1, a+b2, a+b3 must be pairwise coprime
    # AND for any two a values a1, a2, the sums must all be pairwise coprime
    
    # Efficient approach: For each B, precompute which a values give
    # pairwise-coprime triples {a+b1, a+b2, a+b3}
    
    for b1 in range(1, 129):
        if time.time() > deadline:
            break
        for b2 in range(b1+1, 129):
            if time.time() > deadline:
                break
            for b3 in range(b2+1, 129):
                if time.time() > deadline:
                    break
                
                B = [b1, b2, b3]
                
                # Find all a where {a+b1, a+b2, a+b3} are pairwise coprime
                valid_as = []
                for a in range(1, 129):
                    s1, s2, s3 = a+b1, a+b2, a+b3
                    if _gcd(s1, s2) == 1 and _gcd(s1, s3) == 1 and _gcd(s2, s3) == 1:
                        valid_as.append(a)
                
                if len(valid_as) < 3:
                    continue
                
                # Now pick 3 a's from valid_as such that all 9 sums are pairwise coprime
                # Two a values a_i, a_j cause conflicts if any a_i+b_k shares a factor with a_j+b_l
                # for any k, l
                found = False
                for combo in combinations(valid_as, 3):
                    A = list(combo)
                    sums = set()
                    for a in A:
                        for b in B:
                            sums.add(a + b)
                    if len(sums) == 9 and _pairwise_coprime(list(sums)):
                        if try_solution(A, B):
                            return best_A, best_B
                        found = True
                        break
                if found:
                    continue
    
    # Strategy 2: Random search
    rng = random.Random(42)
    for _ in range(200000):
        if time.time() > deadline:
            break
        A = list(rng.sample(range(1, 129), 3))
        B = list(rng.sample(range(1, 129), 3))
        score = _eval(A, B)
        if score > best_score:
            best_score = score
            best_A = A[:]
            best_B = B[:]
            if best_score >= 9:
                return best_A, best_B
    
    return best_A, best_B

# Pre-compute the best answer
_best_result = _search()

def construct():
    return _best_result
# EVOLVE-BLOCK-END

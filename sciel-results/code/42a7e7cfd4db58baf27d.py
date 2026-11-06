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
    """Search for optimal A, B using structured approach."""
    deadline = time.time() + 85  # 85s budget, leaving 5s safety margin
    
    # Generate all primes in [2, 256]
    is_prime = [True] * 257
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(256)) + 1):
        if is_prime[i]:
            for j in range(i*i, 257, i):
                is_prime[j] = False
    
    # Start with a known good baseline
    best_score = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = _eval(best_A, best_B)
    
    rng = random.Random(12345)
    
    # Key insight: For 9 distinct pairwise coprime sums, they should ideally be primes
    # or at least have no common factors. Let's focus on making sums prime.
    
    # Strategy: Enumerate A from small values, then for each A find B values
    # such that all 9 sums are prime and distinct.
    
    # Phase 1: Systematic search with A from [1..30]
    # For each A, find B values where a1+b, a2+b, a3+b are all prime
    # Then check if 3 such B values give 9 distinct pairwise coprime sums
    
    a_candidates = list(range(1, 31))
    all_A = list(combinations(a_candidates, 3))
    
    for idx_A, A in enumerate(all_A):
        if time.time() > deadline:
            break
        
        A = list(A)
        
        # For this A, find all b in [1,128] such that a1+b, a2+b, a3+b are all prime
        good_bs = []
        for b in range(1, 129):
            s1 = A[0] + b
            s2 = A[1] + b
            s3 = A[2] + b
            if 2 <= s1 <= 256 and 2 <= s2 <= 256 and 2 <= s3 <= 256:
                if is_prime[s1] and is_prime[s2] and is_prime[s3]:
                    good_bs.append(b)
        
        # Now try combinations of 3 from good_bs
        if len(good_bs) >= 3:
            # Try all C(len(good_bs), 3) combinations if not too many, else sample
            if len(good_bs) <= 25:
                b_combos = list(combinations(good_bs, 3))
            else:
                # Sample 500 random combinations
                b_combos = []
                seen = set()
                for _ in range(500):
                    combo = tuple(sorted(rng.sample(good_bs, 3)))
                    if combo not in seen:
                        seen.add(combo)
                        b_combos.append(combo)
            
            for B_tuple in b_combos:
                if time.time() > deadline:
                    break
                B = list(B_tuple)
                if len(set(B)) != 3:
                    continue
                score = _eval(A, B)
                if score > best_score:
                    best_score = score
                    best_A = A[:]
                    best_B = B[:]
                    if best_score >= 9:
                        return best_A, best_B
    
    # Phase 2: Random search with local improvement
    for _ in range(300000):
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
    
    # Phase 3: Local search from best solution
    if best_score < 9:
        for _ in range(150000):
            if time.time() > deadline:
                break
            if rng.random() < 0.5:
                idx = rng.randint(0, 2)
                new_val = rng.randint(1, 128)
                if new_val not in best_A:
                    old = best_A[idx]
                    best_A[idx] = new_val
                    score = _eval(best_A, best_B)
                    if score >= best_score:
                        best_score = score
                        if best_score >= 9:
                            return best_A, best_B
                    else:
                        best_A[idx] = old
            else:
                idx = rng.randint(0, 2)
                new_val = rng.randint(1, 128)
                if new_val not in best_B:
                    old = best_B[idx]
                    best_B[idx] = new_val
                    score = _eval(best_A, best_B)
                    if score >= best_score:
                        best_score = score
                        if best_score >= 9:
                            return best_A, best_B
                    else:
                        best_B[idx] = old
    
    return best_A, best_B

# Pre-compute the best answer
_best_result = _search()

def construct():
    return _best_result
# EVOLVE-BLOCK-END

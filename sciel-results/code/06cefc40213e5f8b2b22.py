"""Construct pairwise-coprime sumsets with 3 distinct integers each in [1,128]."""
import time
from math import gcd
from itertools import combinations

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _pairwise_coprime(vals):
    """Check if all distinct values in vals are pairwise coprime."""
    distinct = sorted(set(vals))
    for i in range(len(distinct)):
        for j in range(i + 1, len(distinct)):
            if gcd(distinct[i], distinct[j]) != 1:
                return False
    return True

def _score(A, B):
    """Return (num_distinct_coprime, is_valid)."""
    sums = [a + b for a in A for b in B]
    distinct = sorted(set(sums))
    if not _pairwise_coprime(distinct):
        return 0, False
    return len(distinct), True

def construct():
    """
    Find A, B (each 3 distinct ints in [1,128]) maximizing distinct pairwise-coprime sums.
    Max possible is 9 (all sums distinct primes).
    """
    deadline = time.time() + 85
    
    # Precompute primes up to 256
    primes = set(p for p in range(2, 257) if _is_prime(p))
    
    best_score = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    # Check baseline
    sc, valid = _score(best_A, best_B)
    if valid:
        best_score = sc
    
    # Strategy 1: Structured search - arithmetic progressions
    # A = [x, x+d, x+2d], B = [y, y+e, y+2e]
    # Sums form a 3x3 grid: (x+y) + i*d + j*e for i,j in {0,1,2}
    
    # To get 9 distinct primes, we need the 3x3 grid to be all primes
    # Let's search efficiently
    
    # Precompute valid AP triples
    ap_as = []
    for x in range(1, 127):
        for d in range(1, 45):
            a3 = x + 2 * d
            if a3 > 128:
                break
            ap_as.append((x, x + d, a3, d))
    
    ap_bs = []
    for y in range(1, 127):
        for e in range(1, 45):
            b3 = y + 2 * e
            if b3 > 128:
                break
            ap_bs.append((y, y + e, b3, e))
    
    # Search for 9 distinct prime sums
    for (a1, a2, a3, d) in ap_as:
        if time.time() > deadline - 15:
            break
        for (b1, b2, b3, e) in ap_bs:
            # Sums: a_i + b_j for i,j in 1,2,3
            # = (a1+b1) + (i-1)*d + (j-1)*e
            # Check if all 9 are primes
            s00 = a1 + b1
            if s00 not in primes:
                continue
            # Quick check: if d and e are such that the grid contains composites, skip
            # All 9 values: s00, s00+e, s00+2e, s00+d, s00+d+e, s00+d+2e, s00+2d, s00+2d+e, s00+2d+2e
            # Check a few key ones first
            if (s00 + e) not in primes or (s00 + d) not in primes:
                continue
            if (s00 + 2*e) not in primes or (s00 + 2*d) not in primes:
                continue
            if (s00 + d + 2*e) not in primes or (s00 + 2*d + e) not in primes:
                continue
            if (s00 + 2*d + 2*e) not in primes:
                continue
            # Verify all 9
            A = [a1, a2, a3]
            B = [b1, b2, b3]
            sums = set(a + b for a in A for b in B)
            if len(sums) == 9 and all(s in primes for s in sums):
                return [A, B]
    
    # Strategy 2: Broader search with random sampling
    import random
    
    # Generate random candidates
    all_vals = list(range(1, 129))
    
    # Try random A and B
    attempts = 0
    while time.time() < deadline - 5 and attempts < 200000:
        attempts += 1
        
        # Randomly choose A and B
        A = random.sample(all_vals, 3)
        B = random.sample(all_vals, 3)
        
        sums = [a + b for a in A for b in B]
        distinct = set(sums)
        
        if len(distinct) <= best_score:
            continue
        
        # Check if all distinct sums are pairwise coprime
        if _pairwise_coprime(list(distinct)):
            sc, valid = _score(A, B)
            if valid and sc > best_score:
                best_score = sc
                best_A = A[:]
                best_B = B[:]
                if best_score == 9:
                    return [best_A, best_B]
    
    # Strategy 3: Targeted search - try to find 9 distinct coprime (not necessarily prime) sums
    # Focus on sums that are primes or products of distinct primes
    # For now, do a more structured random search with bias toward prime-producing configurations
    
    # Try A with small values, B with values that push sums into prime-dense regions
    for a1 in range(1, 30):
        if time.time() > deadline - 3:
            break
        for a2 in range(a1 + 1, 60):
            if time.time() > deadline - 3:
                break
            for a3 in range(a2 + 1, 129):
                if time.time() > deadline - 3:
                    break
                A = [a1, a2, a3]
                # Try random B
                for _ in range(50):
                    if time.time() > deadline - 3:
                        break
                    B = random.sample(all_vals, 3)
                    sums = [a + b for a in A for b in B]
                    distinct = set(sums)
                    if len(distinct) <= best_score:
                        continue
                    if _pairwise_coprime(list(distinct)):
                        sc, valid = _score(A, B)
                        if valid and sc > best_score:
                            best_score = sc
                            best_A = A[:]
                            best_B = B[:]
                            if best_score == 9:
                                return [best_A, best_B]
    
    return [best_A, best_B]

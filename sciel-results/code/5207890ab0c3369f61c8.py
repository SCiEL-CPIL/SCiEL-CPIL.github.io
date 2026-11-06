"""Direct construction: find A, B with 9 distinct pairwise-coprime sums (all primes)."""
import math
import time
import itertools

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
    """Check all pairs are coprime. Returns True if valid."""
    n = len(vals)
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(vals[i], vals[j]) > 1:
                return False
    return True

def construct():
    """Return two lists A,B, each containing exactly 3 distinct integers in [1,128].
    All DISTINCT values in A+B must be pairwise coprime.
    Maximize the cardinality of A+B."""
    t_start = time.time()
    DEADLINE = 85  # safety margin before 90s kill

    # Precompute primes up to 256 (max sum 128+128)
    MAX_SUM = 256
    is_prime = [False] * (MAX_SUM + 1)
    is_prime[2] = True
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, MAX_SUM + 1, i):
                is_prime[j] = False

    # Strategy: Find A, B such that all 9 sums are distinct primes.
    # This guarantees pairwise coprimality and maximum cardinality 9.
    # We fix B to small even numbers and search for A (odd numbers).
    
    candidates_B = []
    small_evens = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    for b in itertools.combinations(small_evens, 3):
        candidates_B.append(list(b))
        
    for B in candidates_B:
        if time.time() - t_start > DEADLINE:
            break
            
        b1, b2, b3 = B
        # Compute differences in B
        b_diffs = set()
        b_diffs.add(abs(b1-b2))
        b_diffs.add(abs(b1-b3))
        b_diffs.add(abs(b2-b3))
        
        # Find all a in [1, 128] such that a+b1, a+b2, a+b3 are all prime
        valid_as = []
        for a in range(1, 129):
            if a % 2 == 0: continue # A must be odd to keep sums odd (since B is even)
            s1 = a + b1
            s2 = a + b2
            s3 = a + b3
            if s1 > MAX_SUM or s2 > MAX_SUM or s3 > MAX_SUM:
                continue
            if is_prime[s1] and is_prime[s2] and is_prime[s3]:
                valid_as.append(a)
                
        if len(valid_as) < 3:
            continue
            
        # Find 3 distinct a's from valid_as such that their pairwise differences are not in b_diffs
        for a1, a2, a3 in itertools.combinations(valid_as, 3):
            d12 = abs(a1 - a2)
            d13 = abs(a1 - a3)
            d23 = abs(a2 - a3)
            
            if d12 in b_diffs or d13 in b_diffs or d23 in b_diffs:
                continue
                
            # If differences are disjoint, sums are distinct.
            # Since they are all prime, they are pairwise coprime.
            return [a1, a2, a3], [b1, b2, b3]
            
    # If not found with small evens, try a broader search for B
    # We can relax the "small" constraint but keep the prime check.
    all_evens = list(range(2, 129, 2))
    # To avoid O(N^3) blowup, we limit B to max element < 60
    for B in itertools.combinations(all_evens, 3):
        if time.time() - t_start > DEADLINE:
            break
        if max(B) > 60:
            continue
        b1, b2, b3 = B
        b_diffs = set([abs(b1-b2), abs(b1-b3), abs(b2-b3)])
        
        valid_as = []
        for a in range(1, 129):
            if a % 2 == 0: continue
            s1 = a + b1
            s2 = a + b2
            s3 = a + b3
            if s1 > MAX_SUM or s2 > MAX_SUM or s3 > MAX_SUM:
                continue
            if is_prime[s1] and is_prime[s2] and is_prime[s3]:
                valid_as.append(a)
                
        if len(valid_as) < 3:
            continue
            
        for a1, a2, a3 in itertools.combinations(valid_as, 3):
            d12 = abs(a1 - a2)
            d13 = abs(a1 - a3)
            d23 = abs(a2 - a3)
            if d12 in b_diffs or d13 in b_diffs or d23 in b_diffs:
                continue
            return [a1, a2, a3], [b1, b2, b3]
            
    # Fallback: Return a known valid solution with cardinality 5
    # A = [1, 7, 13], B = [4, 10, 16]
    # Sums: 5, 11, 17, 11, 17, 23, 17, 23, 29
    # Distinct: {5, 11, 17, 23, 29} - All prime, so pairwise coprime.
    return [1, 7, 13], [4, 10, 16]

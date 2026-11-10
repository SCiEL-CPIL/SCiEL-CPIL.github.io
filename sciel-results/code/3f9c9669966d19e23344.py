"""Find two sets A, B of 3 distinct integers in [1,128] maximizing distinct pairwise-coprime sums."""
import time
from math import gcd
from itertools import combinations

def _pairwise_coprime(vals):
    """Check if all distinct values in vals are pairwise coprime."""
    distinct = list(set(vals))
    n = len(distinct)
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(distinct[i], distinct[j]) != 1:
                return False
    return True

def _evaluate(A, B):
    """Return (num_distinct_sums, is_valid) for sets A, B."""
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    distinct = list(sums)
    n = len(distinct)
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(distinct[i], distinct[j]) != 1:
                return 0, False
    return len(distinct), True

def _is_prime(n):
    """Check if n is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def _search(deadline):
    """Search for best (A, B) pair maximizing distinct coprime sums."""
    # Best known from seed
    best_count = 5
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    # Strategy: Model as finding a 3x3 grid where all 9 sums are distinct primes
    # (primes are automatically pairwise coprime)
    # 
    # For a given A = {a1, a2, a3} and B = {b1, b2, b3}, the 9 sums are:
    # a_i + b_j for i,j in {1,2,3}
    # 
    # Key insight: if we fix A, then for each b in B, the three values a1+b, a2+b, a3+b
    # must all be prime, and must be coprime to all other sums.
    # 
    # Even stronger: if all 9 sums are primes, they're automatically pairwise coprime.
    # 
    # Let's use a structured approach:
    # 1. Precompute primes in [2, 256]
    # 2. For each candidate A, find B elements such that all a_i + b are prime for all i
    # 3. Then check if the 9 sums are all distinct
    
    primes = set()
    for n in range(2, 257):
        if _is_prime(n):
            primes.add(n)
    
    # For a given A = (a1, a2, a3), a value b is "compatible" if a1+b, a2+b, a3+b are all prime
    # We need 3 distinct compatible b values
    
    # Precompute for efficiency: for each a, which b values make a+b prime?
    # b must be in [1, 128], so a+b in [a+1, a+128]
    
    # Let's iterate over A triples and for each, find compatible B values
    # C(128,3) = 341,376 which is manageable if we're fast per A
    
    # Actually, let's be smarter. The constraint is:
    # For b to be compatible with A=(a1,a2,a3): a1+b, a2+b, a3+b all prime
    # This means b must be in the intersection of:
    #   {p - a1 : p prime, p-a1 in [1,128]}
    #   {p - a2 : p prime, p-a2 in [1,128]}  
    #   {p - a3 : p prime, p-a3 in [1,128]}
    
    # Precompute for each a in [1,128], the set of b in [1,128] such that a+b is prime
    prime_b_for_a = {}
    for a in range(1, 129):
        valid_bs = set()
        for b in range(1, 129):
            if a + b in primes:
                valid_bs.add(b)
        prime_b_for_a[a] = valid_bs
    
    # Now for each A = (a1, a2, a3), compatible B values are:
    # prime_b_for_a[a1] & prime_b_for_a[a2] & prime_b_for_a[a3]
    
    # We need at least 3 distinct values in this intersection
    # Then check if the 9 sums are all distinct
    
    # Iterate over A triples
    # To speed up, we can be selective about which A to try
    
    # Let's try all A triples but with early termination
    # C(128,3) = 341,376 - should be fast enough with set operations
    
    start_time = time.time()
    
    # Precompute compatibility scores for all A triples in [1,128]
    # Score = number of b values that make all three a_i+b prime
    # Sort by score descending to find 9-prime solutions faster
    
    # First, compute scores for all triples in [1,64] (phase 1)
    # and [1,128] (phase 2)
    
    # For efficiency, precompute pair intersections
    # pair_compat[a1][a2] = prime_b_for_a[a1] & prime_b_for_a[a2]
    # Then triple compat = pair_compat[a1][a2] & prime_b_for_a[a3]
    
    # Build pair compatibility table for [1,128]
    pair_compat = {}
    for a1 in range(1, 129):
        for a2 in range(a1 + 1, 129):
            pair_compat[(a1, a2)] = prime_b_for_a[a1] & prime_b_for_a[a2]
    
    # Compute scores for all triples
    triple_scores = []
    for a1 in range(1, 129):
        for a2 in range(a1 + 1, 129):
            pc = pair_compat[(a1, a2)]
            if len(pc) < 3:
                continue
            for a3 in range(a2 + 1, 129):
                compat = pc & prime_b_for_a[a3]
                score = len(compat)
                if score >= 3:
                    triple_scores.append((score, a1, a2, a3, compat))
    
    # Sort by score descending
    triple_scores.sort(key=lambda x: -x[0])
    
    # Try high-scoring triples first
    for score, a1, a2, a3, compat in triple_scores:
        if time.time() > deadline - 3:
            break
        
        if score < 3:
            break
        
        # Check if any 3-element subset of compat gives 9 distinct sums
        compat_list = sorted(compat)
        n = len(compat_list)
        found = False
        for b1_idx in range(n):
            for b2_idx in range(b1_idx + 1, n):
                for b3_idx in range(b2_idx + 1, n):
                    b1, b2, b3 = compat_list[b1_idx], compat_list[b2_idx], compat_list[b3_idx]
                    A = [a1, a2, a3]
                    B = [b1, b2, b3]
                    
                    # Check if all 9 sums are distinct
                    sums = set()
                    for a in A:
                        for b in B:
                            sums.add(a + b)
                    
                    if len(sums) == 9:
                        best_count = 9
                        best_A = A[:]
                        best_B = B[:]
                        return best_A, best_B
        
        if time.time() > deadline - 5:
            break
    
    # If we haven't found 9, try to find 8 or 7 distinct coprime sums
    # This requires a different approach - not all sums need to be prime
    # They just need to be pairwise coprime
    
    # Let's do a randomized search for these cases
    import random
    random.seed(42)
    
    while time.time() < deadline - 1:
        A = random.sample(range(1, 129), 3)
        B = random.sample(range(1, 129), 3)
        count, valid = _evaluate(A, B)
        if valid and count > best_count:
            best_count = count
            best_A = A[:]
            best_B = B[:]
            if best_count >= 8:
                break
    
    return best_A, best_B

def construct():
    deadline = time.time() + 85  # 85s deadline, leaving 5s margin for 90s limit
    A, B = _search(deadline)
    # Final validation
    count, valid = _evaluate(A, B)
    if not valid:
        # Fallback to seed
        return [1, 7, 13], [4, 10, 16]
    return A, B

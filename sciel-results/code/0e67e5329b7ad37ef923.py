"""Optimized search for pairwise-coprime sumsets of two bounded integer sets."""
import itertools
import math
import time

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _pairwise_coprime(sums):
    """Check if all distinct sums are pairwise coprime."""
    distinct = list(set(sums))
    n = len(distinct)
    for i in range(n):
        for j in range(i + 1, n):
            if _gcd(distinct[i], distinct[j]) != 1:
                return False
    return True

def _evaluate(A, B):
    """Return (num_distinct_sums, is_valid)."""
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    if not _pairwise_coprime(sums):
        return 0, False
    return len(sums), True

def construct():
    deadline = time.time() + 75  # Safety margin before 90s limit
    
    # Start with known good solution
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_count, _ = _evaluate(best_A, best_B)
    
    # Strategy: Try to find 10 or more pairwise-coprime sums
    # Sums range from 2 to 256
    # Pairwise coprime sums: prefer primes, prime powers with distinct primes
    
    # Precompute: for each pair (a,b) with a,b in [1,128], the sum s=a+b
    # We need to pick 3 a's and 3 b's such that all 9 sums are pairwise coprime
    
    # Approach: enumerate candidate sums that are pairwise coprime, then check
    # if they can be covered by 3x3 grid
    
    # Better approach: search over A and B directly with pruning
    
    # Generate all possible A sets (combinations of 3 from [1,128])
    # That's C(128,3) ~ 340k, and for each try B sets... too slow naively.
    
    # Smarter: for a given A, find valid B sets.
    # For a given A = {a1,a2,a3}, the sums are {a1+b, a2+b, a3+b for b in B}
    # We need all 9 sums pairwise coprime.
    
    # Let's try a structured search:
    # 1. Enumerate A sets (with some heuristics to reduce space)
    # 2. For each A, enumerate B sets and check
    
    # To make this tractable, use the fact that if we fix A, the constraint on B
    # is that for each pair of distinct sums (from different a's or same a different b's),
    # they must be coprime.
    
    # Key observation: sums from same a: a+b1, a+b2, a+b3 must be pairwise coprime
    # sums from same b: a1+b, a2+b, a3+b must be pairwise coprime
    # cross sums: (ai+bj) and (ak+bl) for (i,j) != (k,l) must be coprime
    
    # Let's use a randomized/local search approach with the structured constraint
    
    # Phase 1: Try to improve by local search
    import random
    random.seed(42)
    
    def _try_improve(A, B, deadline):
        """Local search: try swapping elements in A or B."""
        A = list(A)
        B = list(B)
        cur_count, _ = _evaluate(A, B)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try replacing one element in A
            for i in range(3):
                for new_val in range(1, 129):
                    if new_val in A and new_val != A[i]:
                        continue
                    new_A = list(A)
                    new_A[i] = new_val
                    new_count, valid = _evaluate(new_A, B)
                    if valid and new_count > cur_count:
                        A = new_A
                        cur_count = new_count
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            # Try replacing one element in B
            for i in range(3):
                for new_val in range(1, 129):
                    if new_val in B and new_val != B[i]:
                        continue
                    new_B = list(B)
                    new_B[i] = new_val
                    new_count, valid = _evaluate(A, new_B)
                    if valid and new_count > cur_count:
                        B = new_B
                        cur_count = new_count
                        improved = True
                        break
                if improved:
                    break
        
        return A, B, cur_count
    
    # Phase 2: Random restarts with local search
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Random initial A and B
        A = list(random.sample(range(1, 129), 3))
        B = list(random.sample(range(1, 129), 3))
        
        A, B, count = _try_improve(A, B, deadline)
        
        if count > best_count:
            best_A, best_B, best_count = A, B, count
            if best_count >= 10:
                break  # Found improvement
    
    # Phase 3: Structured search - try specific patterns
    # Use primes for sums
    # If we can get 10 distinct pairwise-coprime sums, that's great
    
    # Try: A = {p1, p2, p3} where pi are primes, B = {q1, q2, q3} where qi are primes
    # Then sums are pi+qj, which are even (if both odd primes) - not good
    # Use one even in each set
    
    # Try specific constructions
    test_sets = [
        ([1, 2, 3], [4, 5, 7]),
        ([1, 2, 5], [3, 7, 11]),
        ([1, 3, 5], [2, 7, 11]),
        ([1, 2, 7], [3, 5, 11]),
        ([1, 4, 7], [2, 5, 11]),
        ([2, 3, 5], [7, 11, 13]),
        ([1, 2, 3], [5, 7, 11]),
        ([1, 2, 5], [7, 11, 13]),
    ]
    
    for A, B in test_sets:
        count, valid = _evaluate(A, B)
        if valid and count > best_count:
            best_A, best_B, best_count = A, B, count
    
    # Phase 4: More systematic search with constraint propagation
    # For a target set of 10 pairwise-coprime sums, check if 3x3 grid exists
    # This is a constraint satisfaction problem
    
    if time.time() < deadline - 10:
        # Generate candidate pairwise-coprime sum sets of size 10
        # Sums must be in [2, 256]
        # Prefer primes and prime powers
        
        # Precompute primes up to 256
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
        
        primes = [p for p in range(2, 257) if _is_prime(p)]
        
        # Try to find 10 primes that can be arranged as 3x3 sums
        # This is complex, so let's try a different angle:
        # Directly search for A, B with more structured approach
        
        # Try all A with small values (more likely to work)
        for a1 in range(1, 20):
            if time.time() > deadline:
                break
            for a2 in range(a1 + 1, 20):
                if time.time() > deadline:
                    break
                for a3 in range(a2 + 1, 20):
                    if time.time() > deadline:
                        break
                    A = [a1, a2, a3]
                    
                    # For this A, find valid B
                    # B = {b1, b2, b3}, all sums must be pairwise coprime
                    # Sums: a1+b1, a1+b2, a1+b3, a2+b1, a2+b2, a2+b3, a3+b1, a3+b2, a3+b3
                    # All 9 must be pairwise coprime
                    
                    # For speed, precompute which b values work with this A
                    # b must be in [1, 128]
                    # For a given b, the sums involving b are a1+b, a2+b, a3+b
                    # These must be pairwise coprime with each other and with all other sums
                    
                    # Let's just try small B values
                    for b1 in range(1, 30):
                        if time.time() > deadline:
                            break
                        for b2 in range(b1 + 1, 30):
                            if time.time() > deadline:
                                break
                            for b3 in range(b2 + 1, 30):
                                if time.time() > deadline:
                                    break
                                B = [b1, b2, b3]
                                count, valid = _evaluate(A, B)
                                if valid and count > best_count:
                                    best_A, best_B, best_count = A, B, count
    
    return [best_A, best_B]

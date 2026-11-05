import itertools
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin before 90s kill
    
    def pairwise_coprime(vals):
        """Check if all distinct values are pairwise coprime."""
        distinct = list(set(vals))
        n = len(distinct)
        for i in range(n):
            for j in range(i + 1, n):
                if math.gcd(distinct[i], distinct[j]) != 1:
                    return False
        return True
    
    def evaluate(A, B):
        """Return (count, valid) for the sumset A+B."""
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        if len(sums) == 0:
            return 0, False
        return len(sums), pairwise_coprime(sums)
    
    # Precompute: generate candidate sets
    # 128 choose 3 = 349,728 - too many to try all pairs
    # Strategy: 
    # Phase 1: Try specific structured families
    # Phase 2: Random sampling with deadline
    
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_count = 5
    
    # Phase 1: Structured search
    # Try A sets that include 1 (gives sums offset by 1)
    # Try A sets with small numbers
    
    # Generate candidate A sets to try (prioritized)
    # Include 1 in many, try various patterns
    candidates_A = []
    
    # Family 1: A contains 1
    for x in range(2, 129):
        for y in range(x + 1, 129):
            candidates_A.append([1, x, y])
    
    # Family 2: A without 1, small values
    for x in range(1, 40):
        for y in range(x + 1, 40):
            for z in range(y + 1, 40):
                candidates_A.append([x, y, z])
    
    # Remove duplicates and limit
    seen_A = set()
    unique_A = []
    for a in candidates_A:
        key = tuple(a)
        if key not in seen_A:
            seen_A.add(key)
            unique_A.append(a)
    
    # Similarly for B
    candidates_B = []
    for x in range(1, 129):
        for y in range(x + 1, 129):
            for z in range(y + 1, 129):
                candidates_B.append([x, y, z])
    
    # This is too many. Let's be smarter.
    # For each A, we'll try a subset of B's.
    
    # Better approach: iterate over A candidates (limited), and for each, 
    # iterate over B candidates (limited), checking validity.
    
    # Let's limit A to first N candidates and B to first M candidates
    # and check all pairs.
    
    # Actually, let's think differently. 
    # The number of distinct sums is at most 9. For 9 distinct sums, all 9 must be pairwise coprime.
    # Let's try to find A, B where all 9 sums are distinct and pairwise coprime.
    
    # Quick check: can we get 9 distinct pairwise coprime sums?
    # Sums range from 2 to 256. 
    # We need 9 pairwise coprime integers in [2, 256].
    # E.g., primes: 2,3,5,7,11,13,17,19,23 - all pairwise coprime.
    # Can we find A,B such that A+B = {2,3,5,7,11,13,17,19,23}?
    # That's a very specific structure.
    
    # Let's do a focused search:
    # For small A and B values, check all combinations.
    
    max_val = 50  # Start with small values
    
    A_candidates = []
    for a1 in range(1, max_val + 1):
        for a2 in range(a1 + 1, max_val + 1):
            for a3 in range(a2 + 1, max_val + 1):
                A_candidates.append([a1, a2, a3])
    
    B_candidates = []
    for b1 in range(1, max_val + 1):
        for b2 in range(b1 + 1, max_val + 1):
            for b3 in range(b2 + 1, max_val + 1):
                B_candidates.append([b1, b2, b3])
    
    # This is C(50,3)^2 which is about 19600^2 = too many
    # Let's limit further.
    
    # Actually C(50,3) = 19600. 19600^2 is way too many.
    # Let's do C(30,3) = 4060. 4060^2 = 16.5 million - might be doable in 80s with optimized check.
    
    # Let's use a tighter bound and optimize the check.
    
    max_val = 30
    A_candidates = []
    for a1 in range(1, max_val + 1):
        for a2 in range(a1 + 1, max_val + 1):
            for a3 in range(a2 + 1, max_val + 1):
                A_candidates.append((a1, a2, a3))
    
    B_candidates = []
    for b1 in range(1, max_val + 1):
        for b2 in range(b1 + 1, max_val + 1):
            for b3 in range(b2 + 1, max_val + 1):
                B_candidates.append((b1, b2, b3))
    
    # Optimize: precompute gcd table for values up to 256
    max_sum = 256
    gcd_table = [[0] * (max_sum + 1) for _ in range(max_sum + 1)]
    for i in range(1, max_sum + 1):
        for j in range(i, max_sum + 1):
            gcd_table[i][j] = math.gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]
    
    def fast_pairwise_coprime(sums_set):
        """Fast check using precomputed gcd table."""
        distinct = list(sums_set)
        n = len(distinct)
        for i in range(n):
            vi = distinct[i]
            for j in range(i + 1, n):
                vj = distinct[j]
                if gcd_table[vi][vj] != 1:
                    return False
        return True
    
    checked = 0
    for A in A_candidates:
        if time.time() > deadline:
            break
        a1, a2, a3 = A
        for B in B_candidates:
            if time.time() > deadline:
                break
            b1, b2, b3 = B
            # Compute 9 sums
            s = (a1+b1, a1+b2, a1+b3, a2+b1, a2+b2, a2+b3, a3+b1, a3+b2, a3+b3)
            sums_set = set(s)
            count = len(sums_set)
            if count <= best_count:
                continue
            # Check pairwise coprime
            if fast_pairwise_coprime(sums_set):
                best_count = count
                best_A = list(A)
                best_B = list(B)
                if best_count == 9:
                    return [best_A, best_B]
            checked += 1
    
    # Phase 2: If we didn't find 9, try a broader random search
    if time.time() < deadline - 5:
        import random
        random.seed(42)
        for _ in range(500000):
            if time.time() > deadline:
                break
            A = random.sample(range(1, 129), 3)
            A.sort()
            B = random.sample(range(1, 129), 3)
            B.sort()
            sums_set = set(a + b for a in A for b in B)
            count = len(sums_set)
            if count <= best_count:
                continue
            if fast_pairwise_coprime(sums_set):
                best_count = count
                best_A = A
                best_B = B
                if best_count == 9:
                    return [best_A, best_B]
    
    return [best_A, best_B]

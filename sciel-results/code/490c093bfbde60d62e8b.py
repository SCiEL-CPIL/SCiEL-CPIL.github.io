"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import math
import time

def _is_pairwise_coprime(values):
    """Check if all distinct values in the list are pairwise coprime."""
    vs = list(set(values))
    n = len(vs)
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(vs[i], vs[j]) != 1:
                return False
    return True

def _score(A, B):
    """Return (cardinality, valid) for the sumset A+B."""
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    if not _is_pairwise_coprime(list(sums)):
        return (0, False)
    return (len(sums), True)

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # Precompute: for each pair (a, b), the sum a+b
    # We want to find A, B (each 3 distinct ints in [1,128]) maximizing |A+B|
    # with all distinct sums pairwise coprime.
    
    # Strategy: enumerate A from a reduced candidate pool, then for each A
    # enumerate B and check validity.
    
    # To reduce search space, note that sums range from 2 to 256.
    # Pairwise coprime sums are rare. Let's try a smarter approach.
    
    # First, let's try to find if 9 distinct pairwise-coprime sums are achievable.
    # If not, try 8, then 7, etc.
    
    # For efficiency, precompute gcd relationships for sums 2..256
    max_sum = 256
    # Build a compatibility matrix: compat[s1][s2] = True if gcd(s1,s2)==1
    # But that's 255*255 = 65025 entries, fine.
    
    # Better: for a given set of sums, check pairwise coprime.
    
    # Let's do a structured search:
    # For A = (a1, a2, a3) with a1 < a2 < a3
    # For B = (b1, b2, b3) with b1 < b2 < b3
    # Check all 9 sums are pairwise coprime (distinct ones).
    
    # To make this tractable, use pruning:
    # The 9 sums are: a1+b1, a1+b2, a1+b3, a2+b1, a2+b2, a2+b3, a3+b1, a3+b2, a3+b3
    
    # Key insight: if we fix A, then for B, the constraints are:
    # For each pair of sums (ai+bj, ak+bl), gcd must be 1.
    
    # Let's try a heuristic search with pruning.
    
    # Candidate pool: use numbers that are "spread out" to maximize distinct sums.
    # Also, sums should avoid small prime factors to be more likely coprime.
    
    # Let's try: search over A from a smaller pool, then B.
    
    # Pool for each element: let's use [1, 128] but with some structure.
    # Actually, let's just do a smart enumeration.
    
    # Step 1: Try to find a solution with 9 distinct sums.
    # Step 2: If not found quickly, try 8, 7, etc.
    
    best_card = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    # Generate candidate A values
    # To keep search manageable, limit the pool.
    # Use a pool of "interesting" numbers: small numbers, primes, etc.
    # Actually, let's just enumerate all triples from [1, 128] but with pruning.
    
    # That's C(128,3) ≈ 349k for A and same for B = too many.
    
    # Better approach: fix the structure.
    # Let A = (a, a+d1, a+d1+d2) and B = (b, b+e1, b+e1+e2)
    # Or just enumerate with strong pruning.
    
    # Let's use a different strategy: enumerate B for each A, but prune early.
    
    # For a given A = (a1, a2, a3), the sums with B=(b1,b2,b3) are:
    # s11=a1+b1, s12=a1+b2, s13=a1+b3
    # s21=a2+b1, s22=a2+b2, s23=a2+b3
    # s31=a3+b1, s32=a3+b2, s33=a3+b3
    
    # Constraints: all distinct sums must be pairwise coprime.
    
    # Let's try: for each A, enumerate b1, b2, b3 with pruning.
    
    # To reduce the pool, note that sums should be in [2, 256].
    # Let's use a candidate pool for B elements: [1, 128]
    
    # But C(128,3) per A is too much. Let's limit A pool too.
    
    # Use a smaller candidate pool: [1, 64] for both, which is C(64,3) ≈ 41k per set.
    # 41k * 41k is still too much.
    
    # Let's be smarter: for a fixed A, enumerate B with early pruning.
    
    # For fixed A=(a1,a2,a3), enumerate b1 from 1..128, b2 from b1+1..128, b3 from b2+1..128.
    # Prune: after choosing b1, check that a1+b1, a2+b1, a3+b1 are pairwise coprime.
    # After choosing b2, check all sums involving b2 against existing sums.
    # After choosing b3, check all.
    
    # This should be fast enough with pruning.
    
    # Let's limit A to a reasonable pool to keep total time under deadline.
    # Use A from [1, 64] to reduce search space.
    
    pool_max = 64
    candidates = list(range(1, pool_max + 1))
    
    best_found = 0
    best_result = ([1, 7, 13], [4, 10, 16])
    
    # Enumerate A triples
    for a1 in candidates:
        if time.time() > deadline:
            break
        for a2 in candidates:
            if a2 <= a1:
                continue
            if time.time() > deadline:
                break
            for a3 in candidates:
                if a3 <= a2:
                    continue
                
                A = (a1, a2, a3)
                
                # Enumerate B with pruning
                for b1 in candidates:
                    if time.time() > deadline:
                        break
                    # Check sums with b1: a1+b1, a2+b1, a3+b1
                    s1 = a1 + b1
                    s2 = a2 + b1
                    s3 = a3 + b1
                    # These three must be pairwise coprime
                    if math.gcd(s1, s2) != 1 or math.gcd(s1, s3) != 1 or math.gcd(s2, s3) != 1:
                        continue
                    
                    for b2 in candidates:
                        if b2 <= b1:
                            continue
                        if time.time() > deadline:
                            break
                        # Sums with b2
                        s4 = a1 + b2
                        s5 = a2 + b2
                        s6 = a3 + b2
                        # Check against existing sums (s1,s2,s3) and among themselves
                        # First check among themselves
                        if math.gcd(s4, s5) != 1 or math.gcd(s4, s6) != 1 or math.gcd(s5, s6) != 1:
                            continue
                        # Check against existing
                        for s_new in (s4, s5, s6):
                            for s_old in (s1, s2, s3):
                                if s_new != s_old and math.gcd(s_new, s_old) != 1:
                                    break
                            else:
                                continue
                            break
                        else:
                            # All good so far, try b3
                            for b3 in candidates:
                                if b3 <= b2:
                                    continue
                                if time.time() > deadline:
                                    break
                                s7 = a1 + b3
                                s8 = a2 + b3
                                s9 = a3 + b3
                                # Check among themselves
                                if math.gcd(s7, s8) != 1 or math.gcd(s7, s9) != 1 or math.gcd(s8, s9) != 1:
                                    continue
                                # Check against all existing
                                valid = True
                                for s_new in (s7, s8, s9):
                                    for s_old in (s1, s2, s3, s4, s5, s6):
                                        if s_new != s_old and math.gcd(s_new, s_old) != 1:
                                            valid = False
                                            break
                                    if not valid:
                                        break
                                if valid:
                                    # Count distinct sums
                                    all_sums = {s1, s2, s3, s4, s5, s6, s7, s8, s9}
                                    card = len(all_sums)
                                    if card > best_found:
                                        best_found = card
                                        best_result = (list(A), [b1, b2, b3])
    
    if best_found > 0:
        return [best_result[0], best_result[1]]
    else:
        return [[1, 7, 13], [4, 10, 16]]
# EVOLVE-BLOCK-END

import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    # Precompute primes up to 256 (max sum is 128+128=256)
    def sieve_primes(n):
        if n < 2:
            return set()
        is_p = [True] * (n + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_p[i]:
                for j in range(i * i, n + 1, i):
                    is_p[j] = False
        return set(i for i in range(2, n + 1) if is_p[i])

    primes = sieve_primes(256)

    def is_prime(n):
        return n in primes

    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def pairwise_coprime(values):
        """Check if all distinct values in the list are pairwise coprime."""
        unique_vals = sorted(set(values))
        for i in range(len(unique_vals)):
            for j in range(i + 1, len(unique_vals)):
                if gcd(unique_vals[i], unique_vals[j]) > 1:
                    return False
        return True

    # CSP approach using rank-1 structure:
    # s_ij = a_i + b_j
    # If we fix s_11, d_i = s_i1 - s_11 (i=1,2,3), e_j = s_1j - s_11 (j=1,2,3)
    # Then s_ij = s_11 + d_i + e_j
    # Constraints:
    #   - All 9 s_ij are in [2, 256], distinct, pairwise coprime
    #   - a_i = s_i1 - b_1, b_j = s_1j - a_1 for some valid a_1, b_1
    #   - 1 <= a_i <= 128, 1 <= b_j <= 128, distinct within each set
    
    # Strategy: iterate over s_11, d_2, d_3, e_2, e_3 (with d_1=e_1=0)
    # Check that all 9 sums are valid, then derive A and B and verify bounds/distinctness
    
    best_result = None
    best_count = 0

    # Phase 1: Search for 9 distinct pairwise-coprime sums
    # s_11 in [2, 256], d_2, d_3, e_2, e_3 are non-zero integers
    # s_ij = s_11 + d_i + e_j must be in [2, 256]
    
    # To be efficient, we'll iterate over small offsets
    # d_i and e_j should be such that s_11 + d_i + e_j stays in [2, 256]
    
    for s11 in range(2, 257):
        if time.time() > deadline:
            break
        if not is_prime(s11):
            # For max cardinality, primes are ideal, but let's also try non-prime coprime
            # Actually, for 9 pairwise coprime values, they don't all need to be prime
            # But primes guarantee pairwise coprime. Let's focus on primes first.
            continue
        
        # d_2, d_3: offsets for rows 2,3 (d_1 = 0)
        # e_2, e_3: offsets for cols 2,3 (e_1 = 0)
        # All s_ij must be distinct and pairwise coprime
        
        for d2 in range(-100, 101):
            if time.time() > deadline:
                break
            if d2 == 0:
                continue
            s21 = s11 + d2
            if s21 < 2 or s21 > 256:
                continue
            if not is_prime(s21):
                continue
            if gcd(s11, s21) > 1:
                continue
            
            for d3 in range(-100, 101):
                if time.time() > deadline:
                    break
                if d3 == 0 or d3 == d2:
                    continue
                s31 = s11 + d3
                if s31 < 2 or s31 > 256:
                    continue
                if not is_prime(s31):
                    continue
                if gcd(s11, s31) > 1 or gcd(s21, s31) > 1:
                    continue
                
                for e2 in range(-100, 101):
                    if time.time() > deadline:
                        break
                    if e2 == 0:
                        continue
                    s12 = s11 + e2
                    if s12 < 2 or s12 > 256:
                        continue
                    if not is_prime(s12):
                        continue
                    if gcd(s11, s12) > 1 or gcd(s21, s12) > 1:
                        continue
                    
                    # s22 = s11 + d2 + e2
                    s22 = s11 + d2 + e2
                    if s22 < 2 or s22 > 256:
                        continue
                    if not is_prime(s22):
                        continue
                    # Check coprimality with s11, s21, s12
                    if gcd(s11, s22) > 1 or gcd(s21, s22) > 1 or gcd(s12, s22) > 1:
                        continue
                    
                    for e3 in range(-100, 101):
                        if time.time() > deadline:
                            break
                        if e3 == 0 or e3 == e2:
                            continue
                        s13 = s11 + e3
                        if s13 < 2 or s13 > 256:
                            continue
                        if not is_prime(s13):
                            continue
                        if gcd(s11, s13) > 1 or gcd(s21, s13) > 1 or gcd(s12, s13) > 1:
                            continue
                        
                        s23 = s11 + d2 + e3
                        if s23 < 2 or s23 > 256:
                            continue
                        if not is_prime(s23):
                            continue
                        if (gcd(s11, s23) > 1 or gcd(s21, s23) > 1 or 
                            gcd(s12, s23) > 1 or gcd(s13, s23) > 1 or 
                            gcd(s22, s23) > 1):
                            continue
                        
                        s32 = s11 + d3 + e2
                        if s32 < 2 or s32 > 256:
                            continue
                        if not is_prime(s32):
                            continue
                        if (gcd(s11, s32) > 1 or gcd(s21, s32) > 1 or 
                            gcd(s12, s32) > 1 or gcd(s13, s32) > 1 or 
                            gcd(s22, s32) > 1 or gcd(s23, s32) > 1):
                            continue
                        
                        s33 = s11 + d3 + e3
                        if s33 < 2 or s33 > 256:
                            continue
                        if not is_prime(s33):
                            continue
                        if (gcd(s11, s33) > 1 or gcd(s21, s33) > 1 or 
                            gcd(s12, s33) > 1 or gcd(s13, s33) > 1 or 
                            gcd(s22, s33) > 1 or gcd(s23, s33) > 1 or 
                            gcd(s32, s33) > 1):
                            continue
                        
                        # All 9 sums are distinct primes -> pairwise coprime
                        # Now derive A and B
                        # s_ij = a_i + b_j
                        # s_11 = a_1 + b_1
                        # s_21 = a_2 + b_1 => a_2 = a_1 + d2
                        # s_31 = a_3 + b_1 => a_3 = a_1 + d3
                        # s_12 = a_1 + b_2 => b_2 = b_1 + e2
                        # s_13 = a_1 + b_3 => b_3 = b_1 + e3
                        
                        # We need: 1 <= a_i <= 128, 1 <= b_j <= 128
                        # a_1 in [1, 128], a_2 = a_1 + d2 in [1, 128], a_3 = a_1 + d3 in [1, 128]
                        # b_1 in [1, 128], b_2 = b_1 + e2 in [1, 128], b_3 = b_1 + e3 in [1, 128]
                        # Also a_1 + b_1 = s11
                        
                        # So b_1 = s11 - a_1
                        # Constraints on a_1:
                        #   1 <= a_1 <= 128
                        #   1 <= a_1 + d2 <= 128  =>  1-d2 <= a_1 <= 128-d2
                        #   1 <= a_1 + d3 <= 128  =>  1-d3 <= a_1 <= 128-d3
                        #   1 <= s11 - a_1 <= 128  =>  s11-128 <= a_1 <= s11-1
                        #   1 <= s11 - a_1 + e2 <= 128  =>  s11+e2-128 <= a_1 <= s11+e2-1
                        #   1 <= s11 - a_1 + e3 <= 128  =>  s11+e3-128 <= a_1 <= s11+e3-1
                        
                        lo = max(1, 1 - d2, 1 - d3, s11 - 128, s11 + e2 - 128, s11 + e3 - 128)
                        hi = min(128, 128 - d2, 128 - d3, s11 - 1, s11 + e2 - 1, s11 + e3 - 1)
                        
                        if lo > hi:
                            continue
                        
                        # Pick a_1 = lo (or any valid value)
                        a1 = lo
                        b1 = s11 - a1
                        a2 = a1 + d2
                        a3 = a1 + d3
                        b2 = b1 + e2
                        b3 = b1 + e3
                        
                        A = [a1, a2, a3]
                        B = [b1, b2, b3]
                        
                        # Verify all constraints
                        if (len(set(A)) != 3 or len(set(B)) != 3):
                            continue
                        if any(x < 1 or x > 128 for x in A + B):
                            continue
                        
                        # Verify sums
                        sums = [A[i] + B[j] for i in range(3) for j in range(3)]
                        if len(set(sums)) != 9:
                            continue
                        if not pairwise_coprime(sums):
                            continue
                        
                        # Found valid solution with 9 distinct pairwise coprime sums
                        return [A, B]
    
    # Phase 2: If no all-prime solution found, try non-prime pairwise coprime sums
    # This is harder, but let's try a limited search
    # For now, return a fallback if we haven't found one
    
    # Fallback: try to find any valid solution with maximum distinct pairwise coprime sums
    # Use a simpler approach: random sampling with local improvement
    
    import random
    random.seed(42)
    
    best_A = None
    best_B = None
    best_score = 0
    
    for _ in range(50000):
        if time.time() > deadline:
            break
        A = random.sample(range(1, 129), 3)
        B = random.sample(range(1, 129), 3)
        sums = [a + b for a in A for b in B]
        unique_sums = set(sums)
        if len(unique_sums) <= best_score:
            continue
        if pairwise_coprime(list(unique_sums)):
            best_score = len(unique_sums)
            best_A = A
            best_B = B
            if best_score == 9:
                return [best_A, best_B]
    
    if best_A is not None:
        return [best_A, best_B]
    
    # Last resort fallback
    return [[1, 7, 13], [4, 10, 16]]

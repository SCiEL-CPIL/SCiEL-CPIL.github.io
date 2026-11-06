"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import math
import time

def construct():
    start = time.time()
    deadline = start + 82  # Safety margin before 90s kill
    
    best_score = 5
    best_result = [[1, 7, 13], [4, 10, 16]]
    
    max_sum = 256
    
    # Precompute smallest prime factor for each number up to max_sum
    spf = list(range(max_sum + 1))
    for i in range(2, int(max_sum**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, max_sum + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def prime_factors(n):
        """Return set of distinct prime factors of n."""
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return factors
    
    # Precompute prime factors for all numbers 1..max_sum
    pf_cache = [set() for _ in range(max_sum + 1)]
    for n in range(2, max_sum + 1):
        pf_cache[n] = prime_factors(n)
    
    # Precompute pairwise coprimality matrix for speed
    coprime = [[True] * (max_sum + 1) for _ in range(max_sum + 1)]
    for i in range(1, max_sum + 1):
        for j in range(i, max_sum + 1):
            if not pf_cache[i].isdisjoint(pf_cache[j]):
                coprime[i][j] = False
                coprime[j][i] = False
    
    def can_add(value, forbidden_primes):
        """Check if value can be added given the set of forbidden primes."""
        return pf_cache[value].isdisjoint(forbidden_primes)
    
    def find_best_for_A(A, B_candidates, deadline, best_score):
        """Given A triple, find best B triple from B_candidates."""
        a1, a2, a3 = A
        best_local = (0, None)
        
        # For each b, compute the 3 sums and check if they're pairwise coprime
        valid_b_list = []
        for b in B_candidates:
            s1 = a1 + b
            s2 = a2 + b
            s3 = a3 + b
            if s1 > max_sum or s2 > max_sum or s3 > max_sum:
                continue
            # Check pairwise coprime among the 3 sums using precomputed matrix
            if coprime[s1][s2] and coprime[s1][s3] and coprime[s2][s3]:
                valid_b_list.append((b, s1, s2, s3))
        
        if len(valid_b_list) < 3:
            return best_local
        
        # Sort by b value
        valid_b_list.sort(key=lambda x: x[0])
        
        # Limit for performance - use more candidates if we have time
        n = min(len(valid_b_list), 60)
        valid_b_list = valid_b_list[:n]
        
        for i in range(n):
            if time.time() > deadline:
                return best_local
            b1, s1_1, s2_1, s3_1 = valid_b_list[i]
            # Build forbidden prime set from first row
            fp1 = pf_cache[s1_1] | pf_cache[s2_1] | pf_cache[s3_1]
            
            for j in range(i + 1, n):
                b2, s1_2, s2_2, s3_2 = valid_b_list[j]
                # Check row 2 against row 1
                if not (can_add(s1_2, fp1) and can_add(s2_2, fp1) and can_add(s3_2, fp1)):
                    continue
                fp12 = fp1 | pf_cache[s1_2] | pf_cache[s2_2] | pf_cache[s3_2]
                
                for k in range(j + 1, n):
                    b3, s1_3, s2_3, s3_3 = valid_b_list[k]
                    if not (can_add(s1_3, fp12) and can_add(s2_3, fp12) and can_add(s3_3, fp12)):
                        continue
                    # All 9 sums are pairwise coprime!
                    all_sums = [s1_1, s2_1, s3_1, s1_2, s2_2, s3_2, s1_3, s2_3, s3_3]
                    distinct_count = len(set(all_sums))
                    if distinct_count > best_local[0]:
                        best_local = (distinct_count, [b1, b2, b3])
                        if distinct_count >= best_score:
                            return best_local
        return best_local
    
    # Try different search strategies
    strategies = [
        ("odd_A_even_B", list(range(1, 129, 2)), list(range(2, 129, 2))),
        ("even_A_odd_B", list(range(2, 129, 2)), list(range(1, 129, 2))),
        ("odd_A_all_B", list(range(1, 129, 2)), list(range(1, 129))),
        ("all_A_even_B", list(range(1, 129)), list(range(2, 129, 2))),
    ]
    
    for strat_name, A_pool, B_pool in strategies:
        if time.time() > deadline:
            break
        
        # Limit A pool size for performance
        A_search = A_pool[:70]
        B_search = B_pool[:70]
        
        for A in itertools.combinations(A_search, 3):
            if time.time() > deadline:
                break
            local_score, local_B = find_best_for_A(A, B_search, deadline, best_score)
            if local_score > best_score:
                best_score = local_score
                best_result = [list(A), local_B]
                if best_score == 9:
                    return best_result
    
    # Also try a targeted search: look for A with small differences
    # to maximize chance of 9 distinct sums
    small_diff_A = []
    for a1 in range(1, 126):
        for a2 in range(a1 + 1, 127):
            if a2 - a1 > 25:
                break
            for a3 in range(a2 + 1, 128):
                if a3 - a2 > 25:
                    break
                small_diff_A.append((a1, a2, a3))
    
    for A in small_diff_A[:800]:
        if time.time() > deadline:
            break
        # Use all values for B
        local_score, local_B = find_best_for_A(A, list(range(1, 129)), deadline, best_score)
        if local_score > best_score:
            best_score = local_score
            best_result = [list(A), local_B]
            if best_score == 9:
                return best_result
    
    return best_result
# EVOLVE-BLOCK-END

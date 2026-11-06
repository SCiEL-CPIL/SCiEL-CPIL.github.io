"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import math
import time

# EVOLVE-BLOCK-START
def construct():
    """Return two lists A,B, each containing exactly 3 distinct integers in [1,128].
    All DISTINCT values in A+B must be pairwise coprime.
    Maximize the cardinality of A+B."""
    t_start = time.time()
    TIME_LIMIT = 85  # safety margin before 90s kill
    
    # Precompute smallest prime factor for fast gcd checks
    MAX_SUM = 256
    spf = list(range(MAX_SUM + 1))
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def get_prime_factors(n):
        """Return set of prime factors of n."""
        factors = set()
        while n > 1:
            factors.add(spf[n])
            n //= spf[n]
        return factors
    
    # Precompute prime factor sets for all numbers up to MAX_SUM
    prime_factors = [set() for _ in range(MAX_SUM + 1)]
    for i in range(2, MAX_SUM + 1):
        prime_factors[i] = get_prime_factors(i)
    # 1 has no prime factors
    prime_factors[1] = set()
    
    def check_pairwise_coprime(vals):
        """Check if all values in vals are pairwise coprime using precomputed factors."""
        n = len(vals)
        for i in range(n):
            for j in range(i + 1, n):
                if prime_factors[vals[i]] & prime_factors[vals[j]]:
                    return False
        return True
    
    def score_sumset(A, B):
        """Compute the score: number of distinct values in A+B that are pairwise coprime."""
        sums = [a + b for a in A for b in B]
        distinct_sums = list(set(sums))
        if not distinct_sums:
            return 0
        if check_pairwise_coprime(distinct_sums):
            return len(distinct_sums)
        return 0
    
    # Known good seed
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = score_sumset(best_A, best_B)  # Should be 5
    
    # Search strategy: try all combinations from small ranges
    # For each range, try all C(range, 3) for A and C(range, 3) for B
    # To ensure 9 distinct sums, we need the difference sets to not overlap
    # A = {a1, a2, a3}, B = {b1, b2, b3}
    # Sums: a1+b1, a1+b2, a1+b3, a2+b1, a2+b2, a2+b3, a3+b1, a3+b2, a3+b3
    # For all 9 to be distinct, we need:
    # a_i - a_k != b_l - b_j for all (i,k) != (j,l) with appropriate mapping
    # Simpler: if max(A)-min(A) and max(B)-min(B) are such that no difference in A equals negative of difference in B
    
    # Let's just do a thorough search over small ranges
    # Range 1-20: C(20,3) = 1140 for A, same for B -> ~1.3M pairs, might be too slow
    # Let's be smarter: iterate A, then for each A, iterate B
    
    # First, try very small ranges where we can be thorough
    for max_val in [8, 10, 12, 15, 18, 20, 25, 30]:
        if time.time() - t_start > TIME_LIMIT:
            break
        
        range_vals = list(range(1, max_val + 1))
        A_combos = list(itertools.combinations(range_vals, 3))
        B_combos = list(itertools.combinations(range_vals, 3))
        
        for A in A_combos:
            if time.time() - t_start > TIME_LIMIT:
                break
            A = list(A)
            for B in B_combos:
                B = list(B)
                sums = [a + b for a in A for b in B]
                distinct_sums = list(set(sums))
                if len(distinct_sums) < best_score:
                    continue
                if check_pairwise_coprime(distinct_sums):
                    if len(distinct_sums) > best_score:
                        best_score = len(distinct_sums)
                        best_A = A
                        best_B = B
                        if best_score == 9:
                            return [best_A, best_B]
    
    # If we haven't found 9, try a targeted search
    # Focus on sets that are likely to produce 9 distinct sums
    # A = {a, a+d, a+2d}, B = {b, b+e, b+2e} where d*e is not a perfect square (roughly)
    # Actually, simpler: just try arithmetic progressions
    
    if time.time() - t_start < TIME_LIMIT:
        for max_val in [30, 50, 80, 128]:
            if time.time() - t_start > TIME_LIMIT:
                break
            
            range_vals = list(range(1, max_val + 1))
            A_combos = list(itertools.combinations(range_vals, 3))
            B_combos = list(itertools.combinations(range_vals, 3))
            
            for A in A_combos:
                if time.time() - t_start > TIME_LIMIT:
                    break
                A = list(A)
                # Quick check: can this A produce 9 distinct sums with any B?
                # The differences in A are fixed. For 9 distinct sums, we need
                # that no two differences in A match differences in B (with sign)
                for B in B_combos:
                    B = list(B)
                    sums = [a + b for a in A for b in B]
                    distinct_sums = list(set(sums))
                    if len(distinct_sums) < best_score:
                        continue
                    if check_pairwise_coprime(distinct_sums):
                        if len(distinct_sums) > best_score:
                            best_score = len(distinct_sums)
                            best_A = A
                            best_B = B
                            if best_score == 9:
                                return [best_A, best_B]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END

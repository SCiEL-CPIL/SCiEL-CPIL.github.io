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
    
    # Precompute primality for numbers up to 256 (128+128)
    MAX_SUM = 256
    is_prime = [True] * (MAX_SUM + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, MAX_SUM + 1, i):
                is_prime[j] = False
    
    def is_pairwise_coprime(vals):
        """Check if all distinct values in vals are pairwise coprime."""
        distinct_vals = list(set(vals))
        n = len(distinct_vals)
        for i in range(n):
            for j in range(i+1, n):
                if math.gcd(distinct_vals[i], distinct_vals[j]) > 1:
                    return False
        return True
    
    def score_of(A, B):
        """Compute the number of distinct pairwise-coprime sums."""
        sums = [a+b for a in A for b in B]
        distinct_sums = list(set(sums))
        if is_pairwise_coprime(distinct_sums):
            return len(distinct_sums)
        return 0
    
    # Optimized search
    odds = list(range(1, 41, 2))   # 1,3,5,...,39
    evens = list(range(2, 41, 2))  # 2,4,6,...,40
    
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = score_of(best_A, best_B)
    
    # Precompute coprime check for pairs up to 80 (40+40)
    MAX_SUM_LOCAL = 80
    coprime_table = [[False]*(MAX_SUM_LOCAL+1) for _ in range(MAX_SUM_LOCAL+1)]
    for i in range(2, MAX_SUM_LOCAL+1):
        for j in range(2, MAX_SUM_LOCAL+1):
            if i == j:
                coprime_table[i][j] = (i == 1) # 1 is coprime to 1? gcd(1,1)=1. Yes.
            else:
                coprime_table[i][j] = (math.gcd(i, j) == 1)
                
    def fast_score(A, B):
        sums = [a+b for a in A for b in B]
        distinct_sums = list(set(sums))
        if len(distinct_sums) != 9:
            return 0
        # Check pairwise coprime
        n = len(distinct_sums)
        for i in range(n):
            si = distinct_sums[i]
            if si > MAX_SUM_LOCAL:
                # Fallback
                for j in range(i+1, n):
                    if math.gcd(si, distinct_sums[j]) > 1:
                        return 0
            else:
                for j in range(i+1, n):
                    sj = distinct_sums[j]
                    if sj > MAX_SUM_LOCAL:
                        if math.gcd(si, sj) > 1:
                            return 0
                    else:
                        if not coprime_table[si][sj]:
                            return 0
        return 9

    for A in itertools.combinations(odds, 3):
        if time.time() - t_start > TIME_LIMIT:
            break
        A = list(A)
        for B in itertools.combinations(evens, 3):
            if time.time() - t_start > TIME_LIMIT:
                break
            B = list(B)
            s = fast_score(A, B)
            if s > best_score:
                best_score = s
                best_A = A
                best_B = B
                if s == 9:
                    return [best_A, best_B]
                    
    return [best_A, best_B]
# EVOLVE-BLOCK-END

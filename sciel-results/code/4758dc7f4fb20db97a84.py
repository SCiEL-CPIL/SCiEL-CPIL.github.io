import math
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin before 60s limit

    # Precompute gcd table for values up to 256
    max_sum = 256
    gcd_table = [[0] * (max_sum + 1) for _ in range(max_sum + 1)]
    for i in range(1, max_sum + 1):
        for j in range(i, max_sum + 1):
            g = math.gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

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

    # Key insight: for all 9 sums to be pairwise coprime, no two can share factor 2.
    # So all sums must be odd, requiring one set all-odd and the other all-even.
    # This reduces search space dramatically.

    # Generate all-odd and all-even values in [1, 128]
    odd_vals = [x for x in range(1, 129) if x % 2 == 1]
    even_vals = [x for x in range(2, 129) if x % 2 == 0]

    best_A = [1, 3, 5]
    best_B = [2, 4, 6]
    best_count = 0

    # Phase 1: Exhaustive search with small values
    # Since we need pairwise coprime sums, small primes are most likely to work.
    # Try odd values up to 30 and even values up to 30 first.
    
    max_odd = 30
    max_even = 30

    odds_small = [x for x in odd_vals if x <= max_odd]
    evens_small = [x for x in even_vals if x <= max_even]

    A_triples = list(combinations(odds_small, 3))
    B_triples = list(combinations(evens_small, 3))

    for A in A_triples:
        if time.time() > deadline:
            break
        a1, a2, a3 = A
        for B in B_triples:
            if time.time() > deadline:
                break
            b1, b2, b3 = B
            
            # Compute 9 sums
            s = (a1+b1, a1+b2, a1+b3, a2+b1, a2+b2, a2+b3, a3+b1, a3+b2, a3+b3)
            sums_set = set(s)
            count = len(sums_set)
            if count <= best_count:
                continue
            if fast_pairwise_coprime(sums_set):
                best_count = count
                best_A = list(A)
                best_B = list(B)
                if best_count == 9:
                    return [best_A, best_B]

    # Phase 2: Expand search range if needed
    if best_count < 9 and time.time() < deadline - 10:
        max_odd = 50
        max_even = 50

        odds_med = [x for x in odd_vals if x <= max_odd]
        evens_med = [x for x in even_vals if x <= max_even]

        A_triples = list(combinations(odds_med, 3))
        B_triples = list(combinations(evens_med, 3))

        for A in A_triples:
            if time.time() > deadline:
                break
            a1, a2, a3 = A
            for B in B_triples:
                if time.time() > deadline:
                    break
                b1, b2, b3 = B
                
                s = (a1+b1, a1+b2, a1+b3, a2+b1, a2+b2, a2+b3, a3+b1, a3+b2, a3+b3)
                sums_set = set(s)
                count = len(sums_set)
                if count <= best_count:
                    continue
                if fast_pairwise_coprime(sums_set):
                    best_count = count
                    best_A = list(A)
                    best_B = list(B)
                    if best_count == 9:
                        return [best_A, best_B]

    # Phase 3: Random search with larger ranges if still not found
    if best_count < 9 and time.time() < deadline - 5:
        import random
        random.seed(42)
        
        odds_large = [x for x in odd_vals if x <= 100]
        evens_large = [x for x in even_vals if x <= 100]

        for _ in range(5000000):
            if time.time() > deadline:
                break
            A = sorted(random.sample(odds_large, 3))
            B = sorted(random.sample(evens_large, 3))
            s = tuple(a + b for a in A for b in B)
            sums_set = set(s)
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

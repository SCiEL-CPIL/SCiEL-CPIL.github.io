import math
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin before 60s limit

    # Precompute gcd table for values up to 256
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

    # Key insight: for all 9 sums to be odd (necessary for pairwise coprime
    # since two even numbers share factor 2), one set must be all-odd and
    # the other all-even. This reduces search space dramatically.

    # Generate all-odd values in [1, 128]
    odd_vals = [x for x in range(1, 129) if x % 2 == 1]
    even_vals = [x for x in range(2, 129) if x % 2 == 0]

    best_A = [1, 3, 5]
    best_B = [2, 4, 6]
    best_count = 0

    # Phase 1: Focused search with small values
    # Symmetry breaking: Since A+B = B+A, we can assume min(A) < min(B)
    # to avoid checking both (A,B) and (B,A).
    # Note: A is odd, B is even.
    
    max_odd = 40
    max_even = 40

    odds_small = [x for x in odd_vals if x <= max_odd]
    evens_small = [x for x in even_vals if x <= max_even]

    A_triples = list(combinations(odds_small, 3))
    B_triples = list(combinations(evens_small, 3))

    for A in A_triples:
        if time.time() > deadline:
            break
        a1, a2, a3 = A
        min_A = a1
        for B in B_triples:
            if time.time() > deadline:
                break
            b1, b2, b3 = B
            # Symmetry breaking: skip if min(A) > min(B)
            # This ensures we only check each pair of sets once
            # (either as (Odd, Even) or (Even, Odd) effectively, 
            # but here we fixed A=Odd, B=Even. 
            # Wait, if A is Odd and B is Even, A+B is the same as B+A.
            # The pair (A,B) and (B,A) are distinct in our loop 
            # because A_triples are odds and B_triples are evens.
            # We cannot swap A and B within the loop because A must be odd 
            # and B must be even for the parity argument to hold for ALL sums.
            # If we swapped, A would be even and B odd. 
            # But we are only searching A=Odd, B=Even in Phase 1.
            # Phase 3 handles A=Even, B=Odd.
            # So we are NOT double counting (A,B) and (B,A) in Phase 1.
            # Phase 1 finds (Odd, Even). Phase 3 finds (Even, Odd).
            # These are disjoint sets of solutions? 
            # No, the solution (A,B) with A odd, B even is the SAME sumset 
            # as (B,A) with B even, A odd.
            # So Phase 1 and Phase 3 find the SAME solutions!
            # We can remove Phase 3 if Phase 1 is exhaustive enough, 
            # or just keep Phase 1 and remove Phase 3 to save time.
            
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

    # Phase 2: Broader search if needed
    if best_count < 9 and time.time() < deadline - 5:
        import random
        random.seed(12345)
        odds_med = [x for x in odd_vals if x <= 80]
        evens_med = [x for x in even_vals if x <= 80]

        for _ in range(2000000):
            if time.time() > deadline:
                break
            A = sorted(random.sample(odds_med, 3))
            B = sorted(random.sample(evens_med, 3))
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

    # Phase 3 is redundant if Phase 1 is exhaustive for (Odd, Even) 
    # because (Even, Odd) yields the same sumset.
    # However, if Phase 1 range was limited, Phase 3 with same range is redundant.
    # If Phase 3 used a different range, it might help.
    # Given time constraints, we skip Phase 3 to save time for Phase 2 
    # or just rely on Phase 1.
    
    return [best_A, best_B]

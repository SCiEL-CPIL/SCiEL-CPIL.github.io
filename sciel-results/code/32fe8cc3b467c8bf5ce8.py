import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin before 90s kill

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
    #
    # We search for A (all-odd) and B (all-even) such that all 9 sums are
    # distinct and pairwise coprime.

    # Generate all-odd values in [1, 128]
    odd_vals = [x for x in range(1, 129) if x % 2 == 1]
    even_vals = [x for x in range(2, 129) if x % 2 == 0]

    # Strategy: Try small values first (more likely to give distinct small sums)
    # For each combination of 3 odd values for A and 3 even values for B,
    # check if all 9 sums are distinct and pairwise coprime.

    # To keep search tractable, limit to small values first
    # C(64,3) = 41664 odd triples, C(63,3) = 40425 even triples
    # 41664 * 40425 is too many, so we limit ranges.

    best_A = [1, 3, 5]
    best_B = [2, 4, 6]
    best_count = 0

    # Phase 1: Focused search with small values (high chance of finding 9)
    # Try odd values up to 40, even values up to 40
    max_odd = 40
    max_even = 40

    odds_small = [x for x in odd_vals if x <= max_odd]
    evens_small = [x for x in even_vals if x <= max_even]

    # Generate triples
    from itertools import combinations
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

    # Phase 2: Broader search if needed
    if best_count < 9 and time.time() < deadline - 5:
        # Try odd up to 80, even up to 80 but with sampling
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

    # Phase 3: Also try the reverse parity (A even, B odd)
    if best_count < 9 and time.time() < deadline - 5:
        A_triples2 = list(combinations(evens_small, 3))
        B_triples2 = list(combinations(odds_small, 3))

        for A in A_triples2:
            if time.time() > deadline:
                break
            a1, a2, a3 = A
            for B in B_triples2:
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

    return [best_A, best_B]

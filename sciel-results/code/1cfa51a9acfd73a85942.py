import math
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin before 60s limit

    def fast_pairwise_coprime(sums_set):
        """Fast check using math.gcd (C-implemented)."""
        distinct = list(sums_set)
        n = len(distinct)
        for i in range(n):
            vi = distinct[i]
            for j in range(i + 1, n):
                if math.gcd(vi, distinct[j]) != 1:
                    return False
        return True

    # Parity constraint: all sums must be odd => A all-odd, B all-even
    # Sums range from 3 (1+2) to 255 (127+128)
    # Generate odd primes in [3, 255]
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    # Odd primes in valid sum range [3, 255]
    odd_primes = [p for p in range(3, 256, 2) if is_prime(p)]
    # Filter: must be achievable as odd+even where odd in [1,128], even in [2,128]
    # Min sum = 1+2=3, max sum = 127+128=255, all odd primes in [3,255] work

    best_A = [1, 3, 5]
    best_B = [2, 4, 6]
    best_count = 0

    # Precompute best_count for seed
    seed_sums = set(a + b for a in best_A for b in best_B)
    if fast_pairwise_coprime(seed_sums):
        best_count = len(seed_sums)

    # Diophantine approach:
    # Assign primes p[0..8] to positions (i,j) where i in {0,1,2}, j in {0,1,2}
    # Constraints: p[i][j] + p[k][l] = p[i][l] + p[k][j] for all i!=k, j!=l
    # Free variables: p00, p01, p02, p10, p20, p21 (6 free)
    # Derived: p11 = p01 + p10 - p00
    #           p12 = p02 + p10 - p00
    #           p21 = p01 + p20 - p00
    #           p22 = p02 + p20 - p00
    #
    # Then: a0 = p00 - b0, a1 = p10 - b0, a2 = p20 - b0
    #       b0 = p00 - a0 (free choice)
    #       b1 = p01 - a0, b2 = p02 - a0
    #
    # So: choose a0, b0, then a1 = p10 - b0, a2 = p20 - b0
    #     b1 = p01 - a0, b2 = p02 - a0
    #     Check: a1 = p11 - b1 => p10 - b0 = (p01+p10-p00) - (p01-a0) = p10 - p00 + a0
    #     => b0 = p00 - a0 ✓ (consistent by construction)
    #
    # So the free variables are: a0, b0, p10, p20, p01, p02 (6 values)
    # With a0 odd in [1,128], b0 even in [2,128]
    # a1 = p10 - b0 must be odd in [1,128], distinct from a0, a2
    # a2 = p20 - b0 must be odd in [1,128], distinct from a0, a1
    # b1 = p01 - a0 must be even in [2,128], distinct from b0, b2
    # b2 = p02 - a0 must be even in [2,128], distinct from b0, b1
    # All 9 primes must be distinct and pairwise coprime (primes are automatically pairwise coprime if distinct)
    # p11 = p01 + p10 - p00 must be prime
    # p12 = p02 + p10 - p00 must be prime
    # p21 = p01 + p20 - p00 must be prime
    # p22 = p02 + p20 - p00 must be prime

    # Strategy: iterate over 6 free prime variables, check consistency
    # p00 = a0 + b0 (odd prime)
    # p01, p02, p10, p20 are chosen odd primes
    # p11 = p01 + p10 - p00, p12 = p02 + p10 - p00
    # p21 = p01 + p20 - p00, p22 = p02 + p20 - p00
    # All must be odd primes, distinct from each other and from p00,p01,p02,p10,p20

    # To make this tractable, iterate over smaller prime sets
    # Use primes up to ~100 for the free variables to keep derived values in range
    small_primes = [p for p in odd_primes if p <= 100]

    # Even values for b0
    even_vals = list(range(2, 129, 2))
    # Odd values for a0
    odd_vals = list(range(1, 129, 2))

    # Phase 1: Diophantine search with small primes
    # Iterate over p00 (which determines a0+b0), then p01, p02, p10, p20
    # For efficiency, precompute which (a0, b0) pairs give each prime p00

    # Build lookup: for each odd prime p, list of (a0, b0) where a0+b0=p, a0 odd in [1,128], b0 even in [2,128]
    prime_to_pairs = {}
    for p in odd_primes:
        pairs = []
        for a0 in odd_vals:
            b0 = p - a0
            if 2 <= b0 <= 128 and b0 % 2 == 0:
                pairs.append((a0, b0))
        if pairs:
            prime_to_pairs[p] = pairs

    # Phase 1: Iterate over 5 free primes (p00, p01, p02, p10, p20) and check
    # For each combination, derive p11, p12, p21, p22 and check they're primes
    # Then check bounds and distinctness for a and b values

    # Use a set for fast prime lookup
    prime_set = set(odd_primes)

    # Limit search to keep within time
    search_primes = [p for p in small_primes if p in prime_to_pairs]

    count_checked = 0
    for p00 in search_primes:
        if time.time() > deadline:
            break
        pairs_00 = prime_to_pairs[p00]
        for p01 in search_primes:
            if p01 == p00:
                continue
            if time.time() > deadline:
                break
            for p02 in search_primes:
                if p02 in (p00, p01):
                    continue
                if time.time() > deadline:
                    break
                for p10 in search_primes:
                    if p10 in (p00, p01, p02):
                        continue
                    if time.time() > deadline:
                        break
                    # Derive p11 = p01 + p10 - p00
                    p11 = p01 + p10 - p00
                    if p11 <= 0 or p11 > 255 or p11 % 2 == 0:
                        continue
                    if p11 not in prime_set:
                        continue
                    if p11 in (p00, p01, p02, p10):
                        continue

                    for p20 in search_primes:
                        if p20 in (p00, p01, p02, p10):
                            continue
                        if time.time() > deadline:
                            break
                        # Derive p21 = p01 + p20 - p00
                        p21 = p01 + p20 - p00
                        if p21 <= 0 or p21 > 255 or p21 % 2 == 0:
                            continue
                        if p21 not in prime_set:
                            continue
                        if p21 in (p00, p01, p02, p10, p11):
                            continue

                        # Derive p12 = p02 + p10 - p00
                        p12 = p02 + p10 - p00
                        if p12 <= 0 or p12 > 255 or p12 % 2 == 0:
                            continue
                        if p12 not in prime_set:
                            continue
                        if p12 in (p00, p01, p02, p10, p11, p21):
                            continue

                        # Derive p22 = p02 + p20 - p00
                        p22 = p02 + p20 - p00
                        if p22 <= 0 or p22 > 255 or p22 % 2 == 0:
                            continue
                        if p22 not in prime_set:
                            continue
                        if p22 in (p00, p01, p02, p10, p11, p21, p12):
                            continue

                        # All 9 primes are distinct! Now check bounds for a and b
                        # For each (a0, b0) pair with a0+b0=p00:
                        for a0, b0 in pairs_00:
                            a1 = p10 - b0
                            a2 = p20 - b0
                            b1 = p01 - a0
                            b2 = p02 - a0

                            # Check bounds
                            if not (1 <= a1 <= 128 and 1 <= a2 <= 128):
                                continue
                            if not (2 <= b1 <= 128 and 2 <= b2 <= 128):
                                continue

                            # Check parity (should be automatic but verify)
                            if a1 % 2 == 0 or a2 % 2 == 0:
                                continue
                            if b1 % 2 != 0 or b2 % 2 != 0:
                                continue

                            # Check distinctness
                            A = [a0, a1, a2]
                            B = [b0, b1, b2]
                            if len(set(A)) != 3 or len(set(B)) != 3:
                                continue

                            # Verify all 9 sums are the assigned primes
                            sums = set()
                            for i in range(3):
                                for j in range(3):
                                    sums.add(A[i] + B[j])
                            if len(sums) != 9:
                                continue

                            # All 9 are distinct primes, so pairwise coprime is automatic
                            # But let's verify just in case
                            if not fast_pairwise_coprime(sums):
                                continue

                            # Success!
                            best_A = A
                            best_B = B
                            best_count = 9
                            return [best_A, best_B]

    # Phase 2: Fallback to original exhaustive search if Diophantine didn't find solution
    if best_count < 9 and time.time() < deadline - 5:
        odds_small = [x for x in range(1, 129, 2) if x <= 40]
        evens_small = [x for x in range(2, 129, 2) if x <= 40]

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

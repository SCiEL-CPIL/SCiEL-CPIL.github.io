import math
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin for 90s limit

    # Sieve primes up to 256 (max sum = 128+128)
    max_sum = 256
    is_prime_arr = [True] * (max_sum + 1)
    is_prime_arr[0] = False
    is_prime_arr[1] = False
    for i in range(2, int(math.isqrt(max_sum)) + 1):
        if is_prime_arr[i]:
            for j in range(i * i, max_sum + 1, i):
                is_prime_arr[j] = False

    def is_prime(n):
        return 0 <= n <= max_sum and is_prime_arr[n]

    def is_pairwise_coprime(nums):
        if len(nums) < 2:
            return True
        nums = list(nums)
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if math.gcd(nums[i], nums[j]) != 1:
                    return False
        return True

    def evaluate(A, B):
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        if len(sums) == 0:
            return -1
        if not is_pairwise_coprime(sums):
            return -1
        return len(sums)

    # Structural change: direct exhaustive search over all 3-element subsets
    # of [1..128] for A and B, but with smart pruning.
    # Total combos: C(128,3)^2 ≈ 44M^2 — too many.
    # Instead: iterate over all A (C(128,3) ≈ 343k), and for each A,
    # find B that maximizes distinct pairwise-coprime sums.
    # For each A, the 3 elements are fixed. For each B candidate,
    # compute the 9 sums and check.
    #
    # Better structural approach: iterate over all possible A (343k),
    # then for each A, iterate over all possible B (343k) but with early
    # termination. 343k * 343k is too much.
    #
    # Alternative: Since we want 9 distinct pairwise-coprime sums (ideally primes),
    # we can search over pairs of 3-element sets more efficiently by:
    # 1. Precompute all 3-element sets
    # 2. For each pair, check if all 9 sums are pairwise coprime
    # 3. But 343k^2 is ~118 billion — too slow
    #
    # Better: Fix A, then for each B, the sums are {a_i + b_j}.
    # We can precompute for each A the set of all possible sums with any b in [1,128].
    # Actually, let's just do a smarter search:
    #
    # Key insight: if all 9 sums are prime, they're automatically pairwise coprime.
    # So search for A, B such that all a_i + b_j are prime.
    #
    # Strategy: iterate over all A (sorted), then for each A, find B such that
    # all 9 sums are prime. For a fixed A = {a1, a2, a3}, we need b1, b2, b3
    # such that a_i + b_j is prime for all i, j.
    #
    # For each candidate b, check which a_i + b are prime.
    # We need 3 b's such that for each b_j, all a_i + b_j are prime.
    #
    # Precompute for each value v in [1,128] and each a in [1,128] whether a+v is prime.
    # Then for a fixed A, a valid b must satisfy: is_prime(a1+b) and is_prime(a2+b) and is_prime(a3+b).
    # Collect all valid b's, then pick 3 distinct ones.
    #
    # This is O(C(128,3) * 128) for finding valid b's per A, which is ~343k * 128 ≈ 44M — feasible.

    # Precompute prime sums table
    # prime_sum[a][b] = is_prime(a+b) for a,b in 1..128
    # We'll use the sieve array directly.

    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = evaluate(best_A, best_B)

    # Generate all 3-element subsets of [1..128]
    # There are C(128,3) = 343,776 of them
    # Precompute valid_b_sets for each A more efficiently
    all_A = list(combinations(range(1, 129), 3))

    # Precompute for each a in [1,128] the set of b in [1,128] where a+b is prime
    # valid_b_for_a[a] = frozenset of b values
    valid_b_for_a = {}
    for a in range(1, 129):
        vbs = []
        for b in range(1, 129):
            s = a + b
            if s <= max_sum and is_prime_arr[s]:
                vbs.append(b)
        valid_b_for_a[a] = frozenset(vbs)

    for idx, A in enumerate(all_A):
        if time.time() > deadline:
            break

        a1, a2, a3 = A

        # Find intersection of valid b's for a1, a2, a3
        valid_bs = sorted(valid_b_for_a[a1] & valid_b_for_a[a2] & valid_b_for_a[a3])

        # We need 3 distinct b's from valid_bs
        if len(valid_bs) >= 3:
            # Since all sums are prime, they're automatically pairwise coprime.
            # Sums are distinct iff for all i≠j and k≠l, a_i+b_k ≠ a_j+b_l
            # i.e., a_i - a_j ≠ b_l - b_k
            # With distinct a's and distinct b's, duplicates can only occur if
            # a_i + b_k = a_j + b_l for i≠j, k≠l, meaning a_i - a_j = b_l - b_k
            # So we need to avoid this. Try first few triples.
            n_valid = len(valid_bs)
            found = False
            max_check = min(30, n_valid)
            for i in range(max_check - 2):
                for j in range(i + 1, max_check - 1):
                    for k in range(j + 1, max_check):
                        B = [valid_bs[i], valid_bs[j], valid_bs[k]]
                        # Quick check: all 9 sums distinct?
                        sums = set()
                        for a in A:
                            for b in B:
                                sums.add(a + b)
                        if len(sums) == 9:
                            # All prime => pairwise coprime
                            return [list(A), B]
                        found = True
                        if found and len(sums) >= best_score:
                            best_score = len(sums)
                            best_A = list(A)
                            best_B = B

    # Fallback: broader pairwise-coprime search (not necessarily all prime)
    # For cases where no all-prime solution was found quickly
    import random
    random.seed(42)
    
    prime_candidates = [i for i in range(1, 129) if is_prime_arr[i]]
    small_prime_candidates = [i for i in prime_candidates if i <= 64]
    
    for trial in range(100000):
        if time.time() > deadline:
            break
        
        strategy = trial % 3
        if strategy == 0:
            A = list(random.sample(range(1, 129), 3))
            B = list(random.sample(range(1, 129), 3))
        elif strategy == 1:
            A = list(random.sample(prime_candidates, 3)) if len(prime_candidates) >= 3 else list(random.sample(range(1, 129), 3))
            B = list(random.sample(range(1, 129), 3))
        else:
            A = list(random.sample(range(1, 129), 3))
            B = list(random.sample(small_prime_candidates, 3)) if len(small_prime_candidates) >= 3 else list(random.sample(range(1, 129), 3))
        
        score = evaluate(A, B)
        if score > best_score:
            best_score = score
            best_A = A
            best_B = B
            if score == 9:
                return [A, B]

    return [best_A, best_B]

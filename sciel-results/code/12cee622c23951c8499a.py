"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def are_pairwise_coprime(nums):
    unique = list(set(nums))
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            if gcd(unique[i], unique[j]) != 1:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85

    # Default fallback
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_count = 5

    # Precompute primes up to 256 using sieve
    def sieve(n):
        is_p = [True] * (n + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_p[i]:
                for j in range(i*i, n+1, i):
                    is_p[j] = False
        return is_p

    is_prime = sieve(256)
    primes_set = set(i for i in range(2, 257) if is_prime[i])

    # Precompute valid_b_pairs: for each pair (a1, a2), which b make both a1+b and a2+b prime
    valid_b_pairs = {}
    for a1 in range(1, 129):
        for a2 in range(a1 + 1, 129):
            valid_b = [b for b in range(1, 129) if is_prime[a1+b] and is_prime[a2+b]]
            valid_b_pairs[(a1, a2)] = valid_b

    found_9 = False

    # Search for 9 distinct prime sums
    # Iterate a1, a2, a3 over full range [1, 128]
    for a1 in range(1, 129):
        if time.time() > deadline - 5:
            break
        for a2 in range(a1 + 1, 129):
            if time.time() > deadline - 5:
                break
            vb_12 = valid_b_pairs.get((a1, a2), [])
            if len(vb_12) < 3:
                continue

            for a3 in range(a2 + 1, 129):
                if time.time() > deadline - 5:
                    break

                # b must be in vb_12 and a3+b must be prime
                valid_b_123 = [b for b in vb_12 if is_prime[a3 + b]]

                if len(valid_b_123) < 3:
                    continue

                # Try to find 3 b's that yield 9 distinct sums
                # Collision condition: a_i + b_j = a_k + b_l means a_i - a_k = b_l - b_j
                # For small valid_b_123, check all combinations
                if len(valid_b_123) <= 30:
                    candidates = valid_b_123
                else:
                    candidates = valid_b_123[:30]

                for b1, b2, b3 in combinations(candidates, 3):
                    A = [a1, a2, a3]
                    B = [b1, b2, b3]
                    sums = [A[i] + B[j] for i in range(3) for j in range(3)]
                    if len(set(sums)) == 9:
                        best_A = A
                        best_B = B
                        best_count = 9
                        found_9 = True
                        break
                if found_9:
                    break
            if found_9:
                break
        if found_9:
            break

    if found_9:
        return [best_A, best_B]

    # Phase 2: Random search for general pairwise coprime sums
    import random
    rng = random.Random(42)

    attempts = 0
    while time.time() < deadline and attempts < 100000:
        A = sorted(rng.sample(range(1, 129), 3))
        B = sorted(rng.sample(range(1, 129), 3))

        sums = [A[i] + B[j] for i in range(3) for j in range(3)]
        distinct_sums = set(sums)
        count = len(distinct_sums)

        if count > best_count:
            if are_pairwise_coprime(distinct_sums):
                best_count = count
                best_A = A[:]
                best_B = B[:]
                if best_count == 9:
                    return [best_A, best_B]
        attempts += 1

    return [best_A, best_B]
# EVOLVE-BLOCK-END

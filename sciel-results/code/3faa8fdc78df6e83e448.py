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
    deadline = start_time + 85  # safety margin before 90s kill

    # Default fallback
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_count = 5

    # --- Sieve primes up to 256 ---
    def sieve(n):
        is_p = [True] * (n + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_p[i]:
                for j in range(i * i, n + 1, i):
                    is_p[j] = False
        return is_p

    is_prime = sieve(256)

    # --- Phase 1: Targeted search for 9 distinct prime sums ---
    # For A = [a1, a2, a3], find b values where a1+b, a2+b, a3+b are all prime.
    # Then pick 3 such b's giving 9 distinct sums.
    found_9 = False

    # Try A triples with small values first (more likely to have many valid b's)
    # Use a modular pre-filter: for p in small_primes, a2-a1, a3-a1, a3-a2 not ≡ 0 mod p
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    for a1 in range(1, 50):
        if time.time() > deadline - 5:
            break
        for a2 in range(a1 + 1, 60):
            if time.time() > deadline - 5:
                break
            # Quick modular check for (a1, a2)
            ok12 = True
            for p in small_primes:
                if p > 2 and (a2 - a1) % p == 0:
                    ok12 = False
                    break
            if not ok12:
                continue
            for a3 in range(a2 + 1, 70):
                if time.time() > deadline - 5:
                    break
                # Quick modular check for full triple
                ok = True
                for p in small_primes:
                    if p > 2:
                        if (a2 - a1) % p == 0 or (a3 - a1) % p == 0 or (a3 - a2) % p == 0:
                            ok = False
                            break
                if not ok:
                    continue

                A = [a1, a2, a3]
                # Find all b in [1,128] where all three sums are prime
                valid_b = []
                for b in range(1, 129):
                    if is_prime[a1 + b] and is_prime[a2 + b] and is_prime[a3 + b]:
                        valid_b.append(b)

                if len(valid_b) < 3:
                    continue

                # Try all triples from valid_b (or a subset if too many)
                if len(valid_b) <= 30:
                    candidates = valid_b
                else:
                    candidates = valid_b[:30]

                for b1, b2, b3 in combinations(candidates, 3):
                    B = [b1, b2, b3]
                    sums = [A[i] + B[j] for i in range(3) for j in range(3)]
                    if len(set(sums)) == 9:
                        best_A = A[:]
                        best_B = B[:]
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

    # --- Phase 2: Broader pairwise-coprime search with structured A ---
    import random
    rng = random.Random(42)

    # For each A triple, find b's where sums are "good" (prime or coprime-friendly)
    # Then search for 3 b's giving max distinct pairwise-coprime sums
    for a1 in range(1, 40):
        if time.time() > deadline - 3:
            break
        for a2 in range(a1 + 1, 50):
            if time.time() > deadline - 3:
                break
            for a3 in range(a2 + 1, 60):
                if time.time() > deadline - 3:
                    break
                A = [a1, a2, a3]

                # For each b, compute the 3 sums and check if they're pairwise coprime
                good_b = []
                for b in range(1, 129):
                    s1, s2, s3 = a1 + b, a2 + b, a3 + b
                    if gcd(s1, s2) == 1 and gcd(s1, s3) == 1 and gcd(s2, s3) == 1:
                        good_b.append(b)

                if len(good_b) < 3:
                    continue

                # Try to find 3 b's from good_b giving 9 distinct pairwise-coprime sums
                # Use a greedy + local search approach
                # First, try all combinations if small
                if len(good_b) <= 20:
                    for b1, b2, b3 in combinations(good_b, 3):
                        B = [b1, b2, b3]
                        sums = [A[i] + B[j] for i in range(3) for j in range(3)]
                        distinct_sums = set(sums)
                        if len(distinct_sums) == 9 and are_pairwise_coprime(distinct_sums):
                            best_A = A[:]
                            best_B = B[:]
                            best_count = 9
                            return [best_A, best_B]

                # If not found, try random sampling
                for _ in range(50):
                    if time.time() > deadline - 3:
                        break
                    B = sorted(rng.sample(good_b, 3))
                    sums = [A[i] + B[j] for i in range(3) for j in range(3)]
                    distinct_sums = set(sums)
                    count = len(distinct_sums)
                    if count > best_count and are_pairwise_coprime(distinct_sums):
                        best_count = count
                        best_A = A[:]
                        best_B = B[:]
                        if best_count == 9:
                            return [best_A, best_B]

    # --- Phase 3: Random search ---
    attempts = 0
    while time.time() < deadline - 1 and attempts < 100000:
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

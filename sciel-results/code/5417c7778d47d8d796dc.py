"""Search-based program: generate varied candidate (A,B) pairs and keep the best valid one."""
import time
import random
from math import gcd
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80  # safety margin under 90s wall-clock

    def distinct_sums(A, B):
        return {a + b for a in A for b in B}

    def is_pairwise_coprime(vals):
        lst = list(vals)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if gcd(lst[i], lst[j]) != 1:
                    return False
        return True

    def score_of(A, B):
        s = distinct_sums(A, B)
        if len(s) < 2:
            return -1
        if not is_pairwise_coprime(s):
            return -1
        return len(s)

    # Initial best from seed
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = score_of(best_A, best_B)
    if best_score < 0:
        best_score = 0

    # Build a pool of candidate numbers biased toward primes / small primes
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    primes = [n for n in range(1, 129) if is_prime(n)]
    composites = [6, 10, 14, 15, 21, 22, 26, 33, 34, 35, 38, 39, 46, 51, 55, 57, 58, 62, 65, 69, 74, 77, 82, 85, 86, 87, 91, 93, 94, 95, 105, 106, 110, 114, 115, 118, 119]
    pool = list(set(primes + composites + [1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 16, 17, 19, 23, 25, 27, 29, 31, 37]))
    pool = [x for x in pool if 1 <= x <= 128]
    random.shuffle(pool)

    # Try a broader set of structured candidates
    structured = [
        ([1, 2, 3], [4, 5, 6]),
        ([1, 3, 5], [2, 4, 6]),
        ([1, 7, 13], [4, 10, 16]),
        ([2, 3, 5], [7, 11, 13]),
        ([1, 2, 4], [3, 5, 7]),
        ([3, 5, 7], [2, 4, 6]),
        ([1, 4, 9], [2, 3, 5]),
        ([2, 5, 11], [3, 7, 13]),
        ([1, 3, 7], [2, 5, 11]),
        ([4, 9, 16], [2, 3, 5]),
        ([1, 2, 5], [3, 7, 11]),
        ([1, 3, 8], [2, 5, 10]),
        ([2, 4, 9], [1, 5, 7]),
        ([3, 5, 10], [1, 4, 8]),
        ([1, 6, 11], [2, 5, 9]),
        ([2, 7, 12], [1, 4, 8]),
        ([1, 4, 10], [3, 6, 11]),
        ([2, 5, 14], [1, 7, 12]),
        ([1, 8, 15], [2, 5, 11]),
        ([3, 6, 13], [1, 4, 10]),
        ([1, 9, 14], [2, 5, 11]),
        ([4, 7, 12], [1, 5, 9]),
        ([2, 8, 15], [1, 5, 11]),
        ([1, 10, 17], [2, 5, 11]),
        ([3, 7, 14], [1, 4, 10]),
        ([1, 11, 19], [2, 5, 13]),
        ([2, 9, 16], [1, 5, 11]),
        ([4, 8, 15], [1, 5, 11]),
        ([1, 12, 20], [2, 5, 13]),
        ([3, 8, 17], [1, 4, 10]),
        # Additional structured candidates
        ([1, 5, 11], [2, 7, 13]),
        ([2, 6, 11], [1, 5, 9]),
        ([1, 4, 8], [3, 7, 11]),
        ([2, 5, 9], [1, 4, 8]),
        ([1, 6, 10], [3, 7, 11]),
        ([3, 5, 9], [1, 4, 8]),
        ([1, 7, 11], [2, 5, 9]),
        ([2, 6, 10], [1, 5, 9]),
        ([1, 8, 12], [3, 5, 9]),
        ([2, 7, 11], [1, 4, 8]),
        ([1, 9, 13], [2, 5, 10]),
        ([3, 6, 10], [1, 4, 8]),
        ([1, 10, 14], [2, 5, 9]),
        ([2, 8, 12], [1, 5, 9]),
        ([1, 11, 15], [2, 4, 8]),
        ([3, 7, 11], [1, 4, 9]),
        ([1, 12, 16], [2, 5, 10]),
        ([2, 9, 13], [1, 4, 8]),
        ([1, 13, 17], [2, 5, 11]),
        ([3, 8, 12], [1, 4, 9]),
    ]
    for A, B in structured:
        if time.time() > deadline:
            break
        s = score_of(A, B)
        if s > best_score:
            best_score = s
            best_A, best_B = A[:], B[:]

    # Random search over the pool
    attempts = 0
    while time.time() < deadline - 2:
        attempts += 1
        if random.random() < 0.1:
            A = random.sample(range(1, 129), 3)
            B = random.sample(range(1, 129), 3)
        else:
            A = random.sample(pool, min(3, len(pool)))
            B = random.sample(pool, min(3, len(pool)))
            if len(A) < 3 or len(B) < 3:
                continue
        s = score_of(A, B)
        if s > best_score:
            best_score = s
            best_A, best_B = A[:], B[:]
            if best_score >= 9:
                break

    # Smart local refinement: prefer candidates that minimize gcd conflicts
    if best_score < 9 and time.time() < deadline - 5:
        # Pass 1: single-element swaps with constraint-aware scoring
        for _ in range(20000):
            if time.time() > deadline - 3:
                break
            A = best_A[:]
            B = best_B[:]
            if random.random() < 0.5:
                idx = random.randrange(3)
                A[idx] = random.choice(pool)
            else:
                idx = random.randrange(3)
                B[idx] = random.choice(pool)
            if len(set(A)) != 3 or len(set(B)) != 3:
                continue
            if not all(1 <= x <= 128 for x in A) or not all(1 <= x <= 128 for x in B):
                continue
            s = score_of(A, B)
            if s > best_score:
                best_score = s
                best_A, best_B = A[:], B[:]
                if best_score >= 9:
                    break

        # Pass 2: two-element swaps in one set
        if best_score < 9 and time.time() < deadline - 3:
            for _ in range(15000):
                if time.time() > deadline - 2:
                    break
                A = best_A[:]
                B = best_B[:]
                if random.random() < 0.5:
                    idx1, idx2 = random.sample(range(3), 2)
                    A[idx1] = random.choice(pool)
                    A[idx2] = random.choice(pool)
                else:
                    idx1, idx2 = random.sample(range(3), 2)
                    B[idx1] = random.choice(pool)
                    B[idx2] = random.choice(pool)
                if len(set(A)) != 3 or len(set(B)) != 3:
                    continue
                if not all(1 <= x <= 128 for x in A) or not all(1 <= x <= 128 for x in B):
                    continue
                s = score_of(A, B)
                if s > best_score:
                    best_score = s
                    best_A, best_B = A[:], B[:]
                    if best_score >= 9:
                        break

        # Pass 3: simultaneous swap in A and B
        if best_score < 9 and time.time() < deadline - 2:
            for _ in range(10000):
                if time.time() > deadline - 1:
                    break
                A = best_A[:]
                B = best_B[:]
                idxA = random.randrange(3)
                idxB = random.randrange(3)
                A[idxA] = random.choice(pool)
                B[idxB] = random.choice(pool)
                if len(set(A)) != 3 or len(set(B)) != 3:
                    continue
                if not all(1 <= x <= 128 for x in A) or not all(1 <= x <= 128 for x in B):
                    continue
                s = score_of(A, B)
                if s > best_score:
                    best_score = s
                    best_A, best_B = A[:], B[:]
                    if best_score >= 9:
                        break

    return [best_A, best_B]
# EVOLVE-BLOCK-END

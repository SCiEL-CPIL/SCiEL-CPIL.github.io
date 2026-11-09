import math
import time
import random

def construct():
    deadline = time.time() + 50.0
    fallback = [[1, 7, 13], [4, 10, 16]]  # gives 5 distinct prime sums, score 1.8/1.8 baseline
    
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return is_prime

    is_prime = sieve(256)
    prime_set = set(i for i in range(2, 257) if is_prime[i])

    def check_pairwise_coprime(nums):
        distinct = list(set(nums))
        n = len(distinct)
        for i in range(n):
            for j in range(i+1, n):
                if math.gcd(distinct[i], distinct[j]) != 1:
                    return False
        return True

    def count_distinct_coprime(A, B):
        sums = [a + b for a in A for b in B]
        distinct = list(set(sums))
        if not check_pairwise_coprime(distinct):
            return 0
        return len(distinct)

    best_score = 0
    best_result = fallback

    def try_update(A, B):
        nonlocal best_score, best_result
        cnt = count_distinct_coprime(A, B)
        if cnt > best_score:
            best_score = cnt
            best_result = [A, B]
            if cnt == 9:
                return True
        return False

    # Strategy 1: Rank-1 parameterization
    # A = {a, a+u, a+2u}, B = {b, b+v, b+2v}
    # sums = (a+b) + i*u + j*v, i,j in {0,1,2}
    # We search over u, v (positive, distinctness requires u!=0, v!=0)
    # and over base = a+b, then recover a,b.
    
    # For distinctness of A: u != 0. For distinctness of B: v != 0.
    # Also a in [1, 128-2u], b in [1, 128-2v].
    # base = a+b ranges over [2, (128-2u)+(128-2v)] but also a,b each >=1.
    # For given base, u, v: need a in [1, 128-2u], b = base - a in [1, 128-2v]
    #   => a in [max(1, base-(128-2v)), min(128-2u, base-1)]
    
    # Try small u, v first
    found = False
    for u in range(1, 40):
        if time.time() > deadline:
            break
        for v in range(1, 40):
            if time.time() > deadline:
                break
            # base range
            # a in [1, 128-2u], b in [1, 128-2v]
            # base = a+b in [2, (128-2u)+(128-2v)]
            max_a = 128 - 2*u
            max_b = 128 - 2*v
            if max_a < 1 or max_b < 1:
                continue
            min_base = 2
            max_base = max_a + max_b
            for base in range(min_base, max_base + 1):
                # sums
                s00 = base
                s10 = base + u
                s20 = base + 2*u
                s01 = base + v
                s11 = base + u + v
                s21 = base + 2*u + v
                s02 = base + 2*v
                s12 = base + u + 2*v
                s22 = base + 2*u + 2*v
                sums9 = [s00, s10, s20, s01, s11, s21, s02, s12, s22]
                # all must be prime (=> distinct if prime and we check distinctness)
                # primes > 2 are odd; 2 is even. If all 9 are prime, they're pairwise coprime.
                if all(s in prime_set for s in sums9):
                    # need distinct
                    if len(set(sums9)) == 9:
                        # recover a, b
                        # a in [max(1, base - max_b), min(max_a, base-1)]
                        lo = max(1, base - max_b)
                        hi = min(max_a, base - 1)
                        if lo <= hi:
                            a = lo
                            b = base - a
                            A = [a, a+u, a+2*u]
                            B = [b, b+v, b+2*v]
                            # validate
                            if all(1 <= x <= 128 for x in A) and all(1 <= x <= 128 for x in B):
                                if len(set(A)) == 3 and len(set(B)) == 3:
                                    return A, B
    # If rank-1 all-prime not found, do a broader search
    # Strategy 2: random search with the rank-1 form but allowing non-prime coprime
    for it in range(200000):
        if time.time() > deadline:
            break
        u = random.randint(1, 60)
        v = random.randint(1, 60)
        max_a = 128 - 2*u
        max_b = 128 - 2*v
        if max_a < 1 or max_b < 1:
            continue
        a = random.randint(1, max_a)
        b = random.randint(1, max_b)
        A = [a, a+u, a+2*u]
        B = [b, b+v, b+2*v]
        if try_update(A, B):
            return A, B

    # Strategy 3: fully random general search
    for it in range(200000):
        if time.time() > deadline:
            break
        A = random.sample(range(1, 129), 3)
        B = random.sample(range(1, 129), 3)
        if try_update(A, B):
            return A, B

    return best_result

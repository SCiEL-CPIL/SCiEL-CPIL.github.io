import itertools
import math
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin out of 90s

    def is_pairwise_coprime(nums):
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

    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = evaluate(best_A, best_B)  # 5

    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]

    primes = sieve(256)
    prime_set = set(primes)

    # Precompute: for each pair (a1, a2) with a1 < a2, find all b such that
    # a1+b and a2+b are both prime. Store as list of valid b values.
    # This helps in the direct A-set enumeration approach.

    # Strategy 1: Direct enumeration of A sets, then find valid B
    # For a given A = {a1, a2, a3}, we need B = {b1, b2, b3} such that
    # all 9 sums are distinct and pairwise coprime.
    # If all 9 sums are distinct primes, they're automatically pairwise coprime.

    # Precompute prime sums for pairs
    # prime_pair_sums[a1][a2] = set of b in [1,128] such that a1+b and a2+b are both prime
    # But this is 128*128*128 = 2M entries, too much. Instead, compute on the fly with caching.

    # Better: Precompute for each a in [1,128] and each b in [1,128] whether a+b is prime
    is_prime_sum = [[False] * 129 for _ in range(129)]
    for a in range(1, 129):
        for b in range(1, 129):
            s = a + b
            if 2 <= s <= 256 and s in prime_set:
                is_prime_sum[a][b] = True

    # For a given A, find candidate B values where all a+b are prime
    # For A = {a1, a2, a3}, valid b must satisfy: is_prime_sum[a1][b] and is_prime_sum[a2][b] and is_prime_sum[a3][b]
    # And the 9 sums must all be distinct.

    found = False

    # Iterate over all combinations of 3 distinct elements from [1,128] for A
    # C(128,3) = 341,376 combinations. For each, we check if there exists a valid B.
    # To speed up, we can filter A sets where at least 3 values of b give all-prime sums.

    # Precompute for each b, which a values give prime sums
    # prime_a_for_b[b] = set of a in [1,128] such that a+b is prime
    prime_a_for_b = [set() for _ in range(129)]
    for b in range(1, 129):
        for a in range(1, 129):
            if is_prime_sum[a][b]:
                prime_a_for_b[b].add(a)

    # For a given A, the valid b values are those where all of a1, a2, a3 are in prime_a_for_b[b]
    # i.e., b such that a1 in prime_a_for_b[b] and a2 in prime_a_for_b[b] and a3 in prime_a_for_b[b]

    # Let's iterate over A sets and for each, find valid B candidates
    # To make this faster, we can iterate over b values first and build A candidates

    # Alternative: Iterate over B sets, then find valid A
    # Symmetric approach.

    # Let's try: for each pair of b values (b1, b2), find a values where a+b1 and a+b2 are both prime.
    # Then for a third b value b3, check if those a values also give prime sums with b3.

    # Actually, let's just do the direct enumeration with early termination.

    # Precompute: for each triple of distinct a values, how many b values give all-prime sums?
    # This is expensive. Instead, let's iterate smartly.

    # Iterate over b1, b2, b3 (distinct) and find a values where all 3 sums are prime.
    # Then check if we can find 3 such a values that give distinct sums.

    # For fixed B = {b1, b2, b3}, valid a must satisfy:
    # is_prime_sum[a][b1] and is_prime_sum[a][b2] and is_prime_sum[a][b3]
    # And the 9 sums must be distinct.

    # Let's iterate over B sets and find valid A.
    # C(128,3) = 341,376 B sets. For each, find a values where all 3 sums are prime.

    # Precompute for each pair (b1, b2) the intersection of prime_a_for_b[b1] and prime_a_for_b[b2]
    # Then for b3, intersect with prime_a_for_b[b3].

    # To speed up, let's precompute pairwise intersections
    # But that's 128*128 sets, each potentially large. Let's do it on the fly with early exit.

    # Iterate over B sets
    for b1 in range(1, 129):
        if time.time() > deadline - 10:
            break
        for b2 in range(b1 + 1, 129):
            if time.time() > deadline - 10:
                break
            # Find a values where a+b1 and a+b2 are both prime
            common_ab = prime_a_for_b[b1] & prime_a_for_b[b2]
            if len(common_ab) < 3:
                continue
            for b3 in range(b2 + 1, 129):
                if time.time() > deadline - 5:
                    break
                # Find a values where a+b1, a+b2, a+b3 are all prime
                common_abc = common_ab & prime_a_for_b[b3]
                if len(common_abc) < 3:
                    continue

                # Now we have at least 3 candidate a values. Pick 3 and check if all 9 sums are distinct.
                # The sums are: a_i + b_j for i,j in {1,2,3}
                # We need all 9 to be distinct.

                # Sort common_abc for deterministic iteration
                abc_list = sorted(common_abc)
                n_abc = len(abc_list)

                # Try all combinations of 3 from abc_list
                # But n_abc could be large. Let's try a few.
                # First, check if there exist 3 a values giving 9 distinct sums.
                # Sums: a1+b1, a1+b2, a1+b3, a2+b1, a2+b2, a2+b3, a3+b1, a3+b2, a3+b3
                # Distinctness: a_i + b_j = a_k + b_l iff a_i - a_k = b_l - b_j
                # So if all a's are distinct and all b's are distinct, we can still have collisions.
                # E.g., a1=1, a2=2, a3=3, b1=1, b2=2, b3=4: 1+4=2+3=3+2=5

                # Let's just check all triples, but limit the number we check per B set.
                checked = 0
                for i in range(n_abc):
                    if checked >= 50:  # Limit per B set
                        break
                    if time.time() > deadline - 3:
                        break
                    for j in range(i + 1, n_abc):
                        if checked >= 50:
                            break
                        for k in range(j + 1, n_abc):
                            if checked >= 50:
                                break
                            checked += 1
                            a1, a2, a3 = abc_list[i], abc_list[j], abc_list[k]
                            A = [a1, a2, a3]
                            B = [b1, b2, b3]

                            # Check bounds (already satisfied by construction)
                            # Check distinctness of A and B (already distinct by construction)

                            # Compute sums and check distinctness and pairwise coprime
                            sums = set()
                            for a in A:
                                for b in B:
                                    sums.add(a + b)

                            if len(sums) == 9 and is_pairwise_coprime(sums):
                                best_A = A
                                best_B = B
                                best_score = 9
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if found:
            break

    if best_score == 9:
        return [list(best_A), list(best_B)]

    # Fallback: try local search for score 8 or better
    def random_set():
        return sorted(random.sample(range(1, 129), 3))

    def local_search():
        A = random_set()
        B = random_set()
        curr_score = evaluate(A, B)

        for step in range(5000):
            if time.time() > deadline - 2:
                return A, B, curr_score

            if curr_score == 9:
                return A, B, curr_score

            if random.random() < 0.5:
                idx = random.randint(0, 2)
                old = A[idx]
                A[idx] = random.randint(1, 128)
                while len(set(A)) < 3:
                    A[idx] = random.randint(1, 128)
                A.sort()
                new_score = evaluate(A, B)
                if new_score < curr_score:
                    A[idx] = old
                    A.sort()
                else:
                    curr_score = new_score
            else:
                idx = random.randint(0, 2)
                old = B[idx]
                B[idx] = random.randint(1, 128)
                while len(set(B)) < 3:
                    B[idx] = random.randint(1, 128)
                B.sort()
                new_score = evaluate(A, B)
                if new_score < curr_score:
                    B[idx] = old
                    B.sort()
                else:
                    curr_score = new_score

        return A, B, curr_score

    for _ in range(20):
        if time.time() > deadline - 2:
            break
        A_ls, B_ls, score_ls = local_search()
        if score_ls > best_score:
            best_score = score_ls
            best_A = A_ls
            best_B = B_ls
            if best_score == 9:
                break

    return [list(best_A), list(best_B)]

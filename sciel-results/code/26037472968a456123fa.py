import itertools
import math
import time
import random

def construct():
    t_start = time.time()
    TIME_LIMIT = 80.0  # Safety margin before 90s kill

    max_val = 256  # max possible sum is 128+128=256

    # Sieve of Eratosthenes
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(max_val)) + 1):
        if is_prime[i]:
            for j in range(i * i, max_val + 1, i):
                is_prime[j] = False

    # Get lists of odd and even numbers in [1,128]
    odds = [x for x in range(1, 129) if x % 2 == 1]
    evens = [x for x in range(2, 129) if x % 2 == 0]

    def check_pair(A, B):
        """Check if all 9 sums are distinct primes."""
        sums = set()
        for a in A:
            for b in B:
                s = a + b
                if not is_prime[s]:
                    return False
                if s in sums:
                    return False
                sums.add(s)
        return True

    # Strategy 1: Systematic search over small values
    # A from small odds, B from small evens
    search_range = 30
    small_odds = [x for x in odds if x <= search_range]
    small_evens = [x for x in evens if x <= search_range]

    for A in itertools.combinations(small_odds, 3):
        if time.time() - t_start > TIME_LIMIT:
            break
        for B in itertools.combinations(small_evens, 3):
            if time.time() - t_start > TIME_LIMIT:
                break
            if check_pair(A, B):
                return [list(A), list(B)]

    # Try reversed parity
    for A in itertools.combinations(small_evens, 3):
        if time.time() - t_start > TIME_LIMIT:
            break
        for B in itertools.combinations(small_odds, 3):
            if time.time() - t_start > TIME_LIMIT:
                break
            if check_pair(A, B):
                return [list(A), list(B)]

    # Strategy 2: Expand search range
    search_range = 60
    mid_odds = [x for x in odds if x <= search_range]
    mid_evens = [x for x in evens if x <= search_range]

    for A in itertools.combinations(mid_odds, 3):
        if time.time() - t_start > TIME_LIMIT:
            break
        for B in itertools.combinations(mid_evens, 3):
            if time.time() - t_start > TIME_LIMIT:
                break
            if check_pair(A, B):
                return [list(A), list(B)]

    for A in itertools.combinations(mid_evens, 3):
        if time.time() - t_start > TIME_LIMIT:
            break
        for B in itertools.combinations(mid_odds, 3):
            if time.time() - t_start > TIME_LIMIT:
                break
            if check_pair(A, B):
                return [list(A), list(B)]

    # Strategy 3: Random sampling
    random.seed(42)
    while time.time() - t_start < TIME_LIMIT:
        A = random.sample(odds, 3)
        B = random.sample(evens, 3)
        if check_pair(A, B):
            return [sorted(A), sorted(B)]
        A = random.sample(evens, 3)
        B = random.sample(odds, 3)
        if check_pair(A, B):
            return [sorted(A), sorted(B)]

    # Fallback: return seed
    return [[1, 7, 13], [4, 10, 16]]

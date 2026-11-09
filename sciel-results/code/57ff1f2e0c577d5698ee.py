"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0

    def time_exceeded():
        return time.time() - start_time > deadline - 2.0

    def is_smooth(n, prime_set):
        if n <= 1:
            return True
        for p in prime_set:
            while n % p == 0:
                n //= p
        return n == 1

    def count_distinct_primes(arr):
        primes = set()
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                s = arr[i] + arr[j]
                temp = s
                d = 2
                while d * d <= temp:
                    while temp % d == 0:
                        primes.add(d)
                        temp //= d
                    d += 1
                if temp > 1:
                    primes.add(temp)
        return len(primes)

    def try_find_with_prime_set(prime_set, max_val=10000):
        candidates = []
        for x in range(1, max_val + 1):
            if time_exceeded():
                return None
            valid = True
            for y in candidates:
                if not is_smooth(x + y, prime_set):
                    valid = False
                    break
            if valid:
                candidates.append(x)
            if len(candidates) >= 12:
                return candidates[:12]
        if len(candidates) >= 12:
            return candidates[:12]
        return None

    # Try various small prime sets, ordered by size then by likelihood
    prime_set_candidates = [
        {2},
        {2, 3},
        {3},
        {3, 5},
        {2, 3, 5},
        {5},
        {5, 7},
        {2, 3, 5, 7},
        {7},
        {2, 5},
    ]

    best_result = None
    best_count = float('inf')

    for ps in prime_set_candidates:
        if time_exceeded():
            break
        result = try_find_with_prime_set(ps)
        if result is not None and len(result) == 12:
            if len(set(result)) == 12 and all(1 <= x <= 10000 for x in result):
                cnt = count_distinct_primes(result)
                if cnt < best_count:
                    best_count = cnt
                    best_result = result
                if best_count <= len(ps):
                    break

    if best_result is not None:
        return best_result

    # Fallback
    return list(range(1, 13))
# EVOLVE-BLOCK-END

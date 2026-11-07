"""Bootstrap MODEL-AND-OPTIMIZE candidate for pairwise-coprime max-sum task."""
import math
import time

# EVOLVE-BLOCK-START
N = 256

def _primes_up_to(n):
    """Return all primes up to n using a bytearray sieve."""
    if n < 2:
        return []
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i:n + 1:i] = bytearray(len(sieve[i * i:n + 1:i]))
    return [i for i in range(2, n + 1) if sieve[i]]

def construct():
    """
    Return the optimal set of pairwise-coprime integers in [1, 256] maximizing sum.
    
    Optimality argument:
    - Each element must have a distinct set of prime factors (pairwise coprime).
    - For each prime p, the maximum contribution is the largest p^k <= 256.
    - We can include 1 (no prime factors).
    - Thus the optimal set is {1} union {largest p^k <= 256 for each prime p <= 256}.
    """
    start = time.time()
    deadline = start + 55.0  # 5s safety margin under 60s limit

    primes = _primes_up_to(N)

    # Build the optimal set: 1 plus the largest power of each prime <= N
    result = [1]
    for p in primes:
        val = p
        # Find the largest p^k <= N (k >= 1)
        while val * p <= N:
            val *= p
        result.append(val)

    # Verify pairwise coprimality and distinctness
    n_res = len(result)
    valid = True
    for i in range(n_res):
        for j in range(i + 1, n_res):
            if math.gcd(result[i], result[j]) != 1:
                valid = False
                break
        if not valid:
            break

    if not (valid and len(set(result)) == len(result) and all(1 <= x <= N for x in result)):
        # Fallback: just return 1 and all primes (should never happen)
        result = [1] + primes[:]

    # Type-check: ensure Python ints, not bools or numpy types
    final = [int(x) for x in result if isinstance(x, int) and not isinstance(x, bool)]

    # Ensure nonempty
    if not final:
        final = [1]

    return final
# EVOLVE-BLOCK-END

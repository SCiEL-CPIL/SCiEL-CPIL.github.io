"""Rewritten program: pairwise-coprime interval integers maximizing sum, 1..256."""

def construct():
    """
    Return a list of distinct Python ints from 1..256 that are pairwise coprime,
    maximizing their sum.

    Key insight: The optimal strategy is to select, for each prime p <= 256,
    the largest power p^k <= 256. Since each selected number has a distinct
    prime base, all are pairwise coprime. Including 1 (which is coprime with
    everything) adds 1 to the sum at no cost.
    """
    n = 256

    # Sieve of Eratosthenes to find all primes up to 256
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, 16):  # sqrt(256) = 16
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False

    primes = [i for i in range(2, n + 1) if is_prime[i]]

    # For each prime, compute the largest power p^k <= 256
    selected = []
    for p in primes:
        val = p
        while val * p <= n:
            val *= p
        selected.append(val)

    # Include 1 (coprime with all, adds 1 to sum)
    result = [1] + selected

    # Ensure all values are plain Python ints
    result = [int(x) for x in result]

    return result

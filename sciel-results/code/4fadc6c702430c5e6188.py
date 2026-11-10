"""Optimal DP for longest nondecreasing-gap prime subsequence <=10000."""
import time

def _sieve_primes(limit):
    """Return list of all primes <= limit using Sieve of Eratosthenes."""
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    primes = _sieve_primes(10000)
    n = len(primes)

    if n < 2:
        return primes[:]

    # dp[i] = length of longest valid subsequence ending at primes[i]
    # parent[i] = index of previous prime in the subsequence
    dp = [1] * n
    parent = [-1] * n

    best_length = 1
    best_i = 0

    for i in range(1, n):
        pi = primes[i]
        best_dp = 0
        best_j = -1

        for j in range(i):
            gap = pi - primes[j]
            if parent[j] == -1:
                lg = 0
            else:
                lg = primes[j] - primes[parent[j]]
            if gap >= lg:
                if dp[j] > best_dp:
                    best_dp = dp[j]
                    best_j = j

        if best_j >= 0:
            dp[i] = best_dp + 1
            parent[i] = best_j
            if dp[i] > best_length:
                best_length = dp[i]
                best_i = i

        if time.time() > deadline:
            break

    # Reconstruct sequence
    seq = []
    cur = best_i
    while cur >= 0:
        seq.append(primes[cur])
        cur = parent[cur]

    seq.reverse()
    return seq

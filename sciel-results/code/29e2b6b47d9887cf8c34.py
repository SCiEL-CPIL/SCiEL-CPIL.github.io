"""Optimal DP for longest nondecreasing-gap prime subsequence <=10000."""
import time
import bisect

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

    dp_table = [None] * n
    argmax_table = [None] * n
    sorted_struct = [None] * n

    best_length = 2
    best_i = 1
    best_j = 0

    for i in range(1, n):
        pi = primes[i]
        cur_dp = [0] * i
        cur_argmax = [0] * i

        for j in range(i):
            gap = pi - primes[j]

            if sorted_struct[j] is not None:
                sorted_gaps, prefix_maxes, prefix_argmax = sorted_struct[j]
                idx = bisect.bisect_right(sorted_gaps, gap) - 1
                if idx >= 0:
                    val = prefix_maxes[idx] + 1
                    cur_dp[j] = val
                    cur_argmax[j] = prefix_argmax[idx]
                else:
                    cur_dp[j] = 2
                    cur_argmax[j] = -1
            else:
                cur_dp[j] = 2
                cur_argmax[j] = -1

            if cur_dp[j] > best_length:
                best_length = cur_dp[j]
                best_i = i
                best_j = j

        dp_table[i] = cur_dp
        argmax_table[i] = cur_argmax

        if i > 0:
            pairs = []
            for k in range(i):
                pairs.append((pi - primes[k], cur_dp[k], k))

            pairs.sort(key=lambda x: x[0])
            m = len(pairs)
            sorted_gaps = [0] * m
            sorted_vals = [0] * m
            sorted_k = [0] * m
            for idx2, p in enumerate(pairs):
                sorted_gaps[idx2] = p[0]
                sorted_vals[idx2] = p[1]
                sorted_k[idx2] = p[2]

            prefix_maxes = [0] * m
            prefix_argmax = [0] * m
            prefix_maxes[0] = sorted_vals[0]
            prefix_argmax[0] = sorted_k[0]
            for m2 in range(1, m):
                if sorted_vals[m2] > prefix_maxes[m2 - 1]:
                    prefix_maxes[m2] = sorted_vals[m2]
                    prefix_argmax[m2] = sorted_k[m2]
                else:
                    prefix_maxes[m2] = prefix_maxes[m2 - 1]
                    prefix_argmax[m2] = prefix_argmax[m2 - 1]

            sorted_struct[i] = (sorted_gaps, prefix_maxes, prefix_argmax)

        if time.time() > deadline:
            break

    if best_j < 0:
        return primes[:1]

    seq = [primes[best_i], primes[best_j]]
    cur_i = best_i
    cur_j = best_j

    while cur_j >= 0 and argmax_table[cur_i][cur_j] >= 0:
        k = argmax_table[cur_i][cur_j]
        seq.append(primes[k])
        cur_i = cur_j
        cur_j = k

    seq.reverse()

    return seq

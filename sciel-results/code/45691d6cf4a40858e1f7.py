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
    deadline = start_time + 55.0  # Safety margin before 60s kill

    primes = _sieve_primes(10000)
    n = len(primes)

    if n < 2:
        return primes[:]

    # dp_table[j] is a list of length j: dp_table[j][k] = dp[k][j] for k in range(j)
    # argmax_table[j] is a list of length j: argmax_table[j][k] = argmax_k[k][j]
    # sorted_struct[j] = (sorted_gaps, prefix_maxes, prefix_argmax) for querying max dp[k][j]
    #   with primes[j]-primes[k] <= threshold

    dp_table = [None] * n
    argmax_table = [None] * n
    sorted_struct = [None] * n

    best_length = 2
    best_i = 1
    best_j = 0

    for i in range(1, n):
        # Compute dp[j][i] for all j < i
        cur_dp = [0] * i
        cur_argmax = [0] * i

        for j in range(i):
            gap = primes[i] - primes[j]

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

        # Store results
        dp_table[i] = cur_dp
        argmax_table[i] = cur_argmax

        # Build structure for i: for all k < i, (primes[i]-primes[k], dp[k][i], k)
        if i > 0:
            pairs = []
            for k in range(i):
                pairs.append((primes[i] - primes[k], cur_dp[k], k))

            pairs.sort(key=lambda x: x[0])
            sorted_gaps = [p[0] for p in pairs]
            sorted_vals = [p[1] for p in pairs]
            sorted_k = [p[2] for p in pairs]

            prefix_maxes = [0] * len(sorted_vals)
            prefix_argmax = [0] * len(sorted_vals)
            prefix_maxes[0] = sorted_vals[0]
            prefix_argmax[0] = sorted_k[0]
            for m in range(1, len(sorted_vals)):
                if sorted_vals[m] > prefix_maxes[m - 1]:
                    prefix_maxes[m] = sorted_vals[m]
                    prefix_argmax[m] = sorted_k[m]
                else:
                    prefix_maxes[m] = prefix_maxes[m - 1]
                    prefix_argmax[m] = prefix_argmax[m - 1]

            sorted_struct[i] = (sorted_gaps, prefix_maxes, prefix_argmax)

        # Check time
        if time.time() > deadline:
            break

    # Reconstruct the best subsequence
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

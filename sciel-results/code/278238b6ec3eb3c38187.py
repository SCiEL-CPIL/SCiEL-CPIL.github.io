"""Optimal DP for longest nondecreasing-gap prime subsequence <=10000."""
import time
from bisect import bisect_right

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

    # dp[j][i] = longest nondecreasing-gap subsequence ending with ..., primes[j], primes[i]
    # We store dp as a list of lists: dp[j] is a list for i > j
    # parent[j][i] = k such that the predecessor of j in the chain is k (i.e., chain ends ..., primes[k], primes[j], primes[i])
    # If no predecessor, parent[j][i] = -1

    # We'll use: dp[j][i] for j < i
    # dp[j][i] = 2 + max({dp[k][j] : k < j and primes[j] - primes[k] <= primes[i] - primes[j]}, default=-1)
    # So dp[j][i] >= 2

    # For efficiency: for each j, precompute for all k < j:
    #   gap_k = primes[j] - primes[k]
    #   val_k = dp[k][j]
    # Sort by gap_k, build prefix max of val_k
    # Then for each i > j, binary search for largest gap_k <= primes[i] - primes[j]

    # Storage: dp[j] will be a dict or list indexed by i (for i > j)
    # Since n ~ 1229, we can use a 2D list but only fill upper triangle
    # dp[j][i] for j < i: store in a list of length n for each j, but only indices > j are valid
    # To save memory, use: dp[j] = [0] * n, but only dp[j][i] for i > j matters

    dp = [[0] * n for _ in range(n)]
    parent = [[-1] * n for _ in range(n)]

    best_len = 2
    best_j = 0
    best_i = 1

    # Process j from 0 to n-2 (j is the middle/penultimate element)
    for j in range(n - 1):
        if time.time() > deadline:
            break

        pj = primes[j]

        # Build the sorted list of (gap_k, dp[k][j]) for all k < j
        # gap_k = primes[j] - primes[k]
        # We need dp[k][j] which has already been computed since k < j < i and we process j in order
        # Wait - we need to be careful. dp[k][j] is computed when we process index k as the "middle"
        # and j as the "last". So dp[k][j] is available when we're processing j as middle.
        # Yes, because k < j, so when we processed k (as middle), we computed dp[k][j].

        if j > 0:
            # Collect (gap, dp_value, k) for all k < j
            entries = []
            for k in range(j):
                gap_k = pj - primes[k]
                val = dp[k][j]
                if val > 0:  # only if dp[k][j] was computed (it should be for k < j)
                    entries.append((gap_k, val, k))

            # Sort by gap_k
            entries.sort(key=lambda x: x[0])

            # Build prefix max: for each position, the max dp value up to that point
            # and the corresponding k that achieves it
            m_entries = len(entries)
            prefix_max_val = [0] * m_entries
            prefix_max_k = [-1] * m_entries

            if m_entries > 0:
                prefix_max_val[0] = entries[0][1]
                prefix_max_k[0] = entries[0][2]
                for idx in range(1, m_entries):
                    if entries[idx][1] > prefix_max_val[idx - 1]:
                        prefix_max_val[idx] = entries[idx][1]
                        prefix_max_k[idx] = entries[idx][2]
                    else:
                        prefix_max_val[idx] = prefix_max_val[idx - 1]
                        prefix_max_k[idx] = prefix_max_k[idx - 1]

            # Extract sorted gaps for binary search
            sorted_gaps = [e[0] for e in entries]
        else:
            sorted_gaps = []
            prefix_max_val = []
            prefix_max_k = []
            m_entries = 0

        # Now for each i > j, compute dp[j][i]
        for i in range(j + 1, n):
            if time.time() > deadline:
                break

            pi = primes[i]
            gap_ij = pi - pj

            if m_entries == 0:
                # No predecessors, chain is just [primes[j], primes[i]]
                dp[j][i] = 2
                parent[j][i] = -1
            else:
                # Find largest index in sorted_gaps where gap <= gap_ij
                # bisect_right gives insertion point, so index-1 is the last one <= gap_ij
                pos = bisect_right(sorted_gaps, gap_ij) - 1

                if pos >= 0:
                    best_prev_len = prefix_max_val[pos]
                    best_prev_k = prefix_max_k[pos]
                    dp[j][i] = 2 + best_prev_len
                    parent[j][i] = best_prev_k
                else:
                    dp[j][i] = 2
                    parent[j][i] = -1

            if dp[j][i] > best_len:
                best_len = dp[j][i]
                best_j = j
                best_i = i

    # Reconstruct the sequence
    if best_len <= 2:
        return [primes[0], primes[1]]

    # Chain ends with ..., primes[parent[best_j][best_i]], primes[best_j], primes[best_i]
    chain = [primes[best_i], primes[best_j]]
    cur_j = best_j
    cur_i = best_i
    remaining = best_len

    while remaining > 2:
        k = parent[cur_j][cur_i]
        if k < 0:
            break
        chain.append(primes[k])
        cur_i = cur_j
        cur_j = k
        remaining -= 1

    chain.reverse()

    # Verify
    for idx in range(len(chain) - 2):
        if chain[idx + 2] - chain[idx + 1] < chain[idx + 1] - chain[idx]:
            # Fallback: return a valid shorter sequence
            return [primes[0], primes[1]]

    return chain

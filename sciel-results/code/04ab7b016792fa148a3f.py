"""Rewritten: longest prime subsequence <=10000 with nondecreasing gaps via DP on pairs."""
import time
from math import isqrt
import numpy as np
from bisect import bisect_left

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    # Sieve primes up to 10000
    N = 10000
    is_prime = bytearray([1]) * (N + 1)
    is_prime[0] = is_prime[1] = 0
    for p in range(2, isqrt(N) + 1):
        if is_prime[p]:
            for m in range(p * p, N + 1, p):
                is_prime[m] = 0
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    P = len(primes)
    if P < 2:
        return primes

    primes_arr = np.array(primes, dtype=np.int32)
    dp = np.full((P, P), 2, dtype=np.int32)  # dp[i][j] for i<j

    best_len = 2
    best_pair = (0, 1)

    for i in range(1, P):
        # Build suffix max over k in [0, i-1] of dp[k][i]
        col = dp[:i, i]  # shape (i,)
        # suffix[m] = max(col[m], col[m+1], ..., col[i-1])
        # Compute via np.maximum.accumulate on reversed array
        rev = col[::-1]
        suff_rev = np.maximum.accumulate(rev)
        suff = suff_rev[::-1]  # suff[m] = max(col[m:])

        pi = primes[i]
        # For each j > i, compute dp[i][j]
        # T_j = 2*pi - primes[j]
        # k_min_j = bisect_left(primes[:i], T_j)
        # if k_min_j < i: dp[i][j] = 1 + suff[k_min_j] else 2

        j_range = np.arange(i + 1, P)
        T_vals = 2 * pi - primes_arr[j_range]  # shape (P-i-1,)

        # Vectorized bisect_left for each T_val in primes[:i]
        # primes[:i] is sorted
        prefix_primes = primes_arr[:i]
        
        # Use searchsorted to find insertion point
        k_mins = np.searchsorted(prefix_primes, T_vals, side='left')
        
        # Where k_min < i, we can extend; otherwise dp = 2
        valid = k_mins < i
        dp_vals = np.full(len(j_range), 2, dtype=np.int32)
        dp_vals[valid] = 1 + suff[k_mins[valid]]
        
        dp[i, i+1:] = dp_vals
        
        # Track best
        max_in_row = np.max(dp_vals)
        if max_in_row > best_len:
            best_len = int(max_in_row)
            # Find first j where dp[i][j] == best_len
            idx = np.argmax(dp_vals)
            best_pair = (i, int(i + 1 + idx))
        
        if time.time() > deadline - 10:
            break

    # Reconstruct sequence from best_pair
    if best_len <= 2:
        return primes[:best_len] if best_len >= 2 else primes

    # Backtrack
    seq = [primes[best_pair[1]], primes[best_pair[0]]]
    i, j = best_pair
    while len(seq) < best_len:
        target = dp[i, j] - 1
        gap_ij = primes[j] - primes[i]
        found = False
        for k in range(i - 1, -1, -1):
            if dp[k, i] == target and (primes[i] - primes[k]) <= gap_ij:
                seq.append(primes[k])
                j, i = i, k
                found = True
                break
        if not found:
            break
    seq.reverse()
    return seq

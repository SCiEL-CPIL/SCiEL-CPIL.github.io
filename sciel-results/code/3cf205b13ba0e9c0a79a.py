"""Rewritten: longest prime subsequence <=10000 with nondecreasing gaps via DP on pairs."""
import time
from math import isqrt

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

    # dp[i][j] for i<j: max length ending ...p_i,p_j. Use list of lists.
    # dp[i][j] stored in a 2D list; dp[i][j] = 2 baseline.
    dp = [[2] * P for _ in range(P)]  # dp[i][j] only valid for i<j
    # For each i, we need suffix max over k of dp[k][i].
    # Process j from 1..P-1; for each i<j compute dp[i][j].
    # For each i, precompute suffix max array over k=0..i-1 of dp[k][i].
    # We'll compute on the fly: when processing column i (as the second index),
    # dp[k][i] for all k<i are already computed (since k<i<j not required; dp[k][i] needs k<i which is done when column i processed).
    # So iterate i from 1..P-1 as the "previous" index; at that time dp[k][i] known for k<i.
    # Build suffix max for column i.
    best_len = 2
    best_pair = (0, 1)  # indices
    for i in range(1, P):
        # build suffix max over k in [0, i-1] of dp[k][i]
        # suffix[i-1] = dp[i-1][i]; suffix[m] = max(dp[m][i], suffix[m+1])
        suff = [0] * (i + 1)
        suff[i - 1] = dp[i - 1][i]
        for m in range(i - 2, -1, -1):
            v = dp[m][i]
            suff[m] = v if v >= suff[m + 1] else suff[m + 1]
        # Now for each j > i, dp[i][j] = 1 + max over k<i with p_k >= 2*p_i - p_j of dp[k][i]
        # p_k >= 2*p_i - p_j  <=>  k >= first index with primes[k] >= 2*primes[i] - primes[j]
        pi = primes[i]
        for j in range(i + 1, P):
            T = 2 * pi - primes[j]
            # find smallest k in [0, i-1] with primes[k] >= T
            # primes sorted; use bisect
            # if T <= primes[0], k_min=0; if T > primes[i-1], no k
            if T <= primes[0]:
                k_min = 0
            else:
                # bisect_left in primes[0:i]
                lo, hi = 0, i
                while lo < hi:
                    mid = (lo + hi) // 2
                    if primes[mid] < T:
                        lo = mid + 1
                    else:
                        hi = mid
                k_min = lo
            if k_min < i:
                cand = 1 + suff[k_min]
            else:
                cand = 2
            dp[i][j] = cand
            if cand > best_len:
                best_len = cand
                best_pair = (i, j)
        if time.time() > deadline - 10:
            break
    # Reconstruct sequence from best_pair
    if best_len <= 2:
        return primes[:best_len] if best_len >= 2 else primes
    # backtrack
    seq = [primes[best_pair[1]], primes[best_pair[0]]]
    i, j = best_pair
    while len(seq) < best_len:
        # find k < i with dp[k][i] + 1 == dp[i][j] and primes[i]-primes[k] <= primes[j]-primes[i]
        target = dp[i][j] - 1
        gap_ij = primes[j] - primes[i]
        found = False
        for k in range(i - 1, -1, -1):
            if dp[k][i] == target and (primes[i] - primes[k]) <= gap_ij:
                seq.append(primes[k])
                j, i = i, k
                found = True
                break
        if not found:
            break
    seq.reverse()
    return seq

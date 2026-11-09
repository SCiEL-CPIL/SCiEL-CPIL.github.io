"""Rewritten: longest prime subsequence <=10000 with nondecreasing gaps via DP on pairs."""
import time
from math import isqrt
from bisect import bisect_left, bisect_right
import numpy as np

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

    # dp[i][j] for i<j: max length ending ...p_i,p_j. Use numpy 2D array.
    dp = np.full((P, P), 2, dtype=np.int16)
    best_len = 2
    best_pair = (0, 1)
    primes_arr = np.array(primes, dtype=np.int32)
    
    for i in range(1, P):
        # build suffix max over k in [0, i-1] of dp[k][i]
        col_i = dp[:i, i]
        suff = np.maximum.accumulate(col_i[::-1])[::-1]
        pi = primes[i]
        pi_arr = primes_arr[i]
        for j in range(i + 1, P):
            T = 2 * pi - primes[j]
            # find smallest k in [0, i-1] with primes[k] >= T
            k_min = bisect_left(primes, T, 0, i)
            if k_min < i:
                cand = 1 + int(suff[k_min])
            else:
                cand = 2
            dp[i, j] = cand
            if cand > best_len:
                best_len = cand
                best_pair = (i, j)
        if time.time() > deadline - 15:
            break
    
    # Reconstruct sequence from best_pair
    if best_len <= 2:
        return primes[:best_len] if best_len >= 2 else primes
    # backtrack
    seq = [primes[best_pair[1]], primes[best_pair[0]]]
    i, j = best_pair
    while len(seq) < best_len:
        target = int(dp[i, j]) - 1
        gap_ij = primes[j] - primes[i]
        found = False
        for k in range(i - 1, -1, -1):
            if int(dp[k, i]) == target and (primes[i] - primes[k]) <= gap_ij:
                seq.append(primes[k])
                j, i = i, k
                found = True
                break
        if not found:
            break
    seq.reverse()

    # Greedy extension: try to extend the sequence forward
    if len(seq) >= 2 and time.time() < deadline - 10:
        last_gap = seq[-1] - seq[-2]
        last_val = seq[-1]
        idx = bisect_left(primes, last_val + last_gap)
        if idx < P:
            seq.append(primes[idx])
            while time.time() < deadline - 5:
                lg = seq[-1] - seq[-2]
                lv = seq[-1]
                nxt = bisect_left(primes, lv + lg)
                if nxt < P:
                    seq.append(primes[nxt])
                else:
                    break

    # Local search: try to improve by swapping elements
    if len(seq) >= 3 and time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for pos in range(1, len(seq) - 1):
                # Try replacing seq[pos] with another prime
                prev_gap = seq[pos] - seq[pos-1]
                next_gap = seq[pos+1] - seq[pos]
                lo = seq[pos-1] + prev_gap  # must be >= prev_gap from previous
                hi = seq[pos+1] - next_gap  # must leave >= next_gap to next
                if lo > hi:
                    continue
                # Find primes in [lo, hi]
                idx_lo = bisect_left(primes, lo)
                idx_hi = bisect_right(primes, hi)
                for new_idx in range(idx_lo, idx_hi):
                    new_val = primes[new_idx]
                    if new_val == seq[pos]:
                        continue
                    # Check if this improves the sequence
                    new_prev_gap = new_val - seq[pos-1]
                    new_next_gap = seq[pos+1] - new_val
                    if new_prev_gap >= prev_gap and new_next_gap >= next_gap:
                        # Check if this allows further extension
                        seq[pos] = new_val
                        improved = True
                        break
                if improved:
                    break
            # After improvement, try greedy extension again
            if improved and time.time() < deadline - 3:
                last_gap = seq[-1] - seq[-2]
                last_val = seq[-1]
                idx = bisect_left(primes, last_val + last_gap)
                if idx < P:
                    seq.append(primes[idx])
                    while time.time() < deadline - 3:
                        lg = seq[-1] - seq[-2]
                        lv = seq[-1]
                        nxt = bisect_left(primes, lv + lg)
                        if nxt < P:
                            seq.append(primes[nxt])
                        else:
                            break

    return seq

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import bisect

def _sieve_primes(limit):
    """Return numpy array of all primes <= limit."""
    if limit < 2:
        return np.array([], dtype=np.int64)
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return np.where(is_prime)[0]

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill

    # Get all primes <= 10000
    primes = _sieve_primes(10000)
    n = len(primes)

    if n == 0:
        return []
    if n == 1:
        return [int(primes[0])]

    primes_list = primes.tolist()
    primes_arr = primes  # numpy array for vectorized gap computation

    # DP approach: We want the longest subsequence of primes p_{i1} < p_{i2} < ... < p_{ik}
    # such that gaps g_j = p_{i(j+1)} - p_{ij} are nondecreasing: g_1 <= g_2 <= ... <= g_{k-1}
    #
    # State: dp[t][s] = length of longest valid subsequence ending at primes[t]
    #         with immediate predecessor primes[s] (s < t)
    #
    # Transition: dp[t][s] = 1 + max{dp[s][r] : r < s, primes[s]-primes[r] <= primes[t]-primes[s]}
    #             or 2 if no such r exists (just primes[r], primes[s], primes[t] with valid gaps)
    #
    # For each s, we need to answer queries: given gap g = primes[t]-primes[s],
    # find max dp[s][r] over all r where primes[s]-primes[r] <= g.
    #
    # We precompute for each s a sorted list of (gap, dp_value) pairs and prefix max.

    # Precompute gap matrix for efficiency
    # gap[i][j] = primes[j] - primes[i] for i < j
    gap_matrix = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        gap_matrix[i, i+1:] = primes_arr[i+1:] - primes_arr[i]

    dp_table = [None] * n
    parent = [None] * n
    query_data = [None] * n  # query_data[s]: (sorted_gaps, prefix_max_dp, prefix_argmax_r)

    best_val = 1
    best_t = 0
    best_s = -1

    # Local references for speed
    br = bisect.bisect_right

    for t in range(n):
        if time.time() > deadline:
            break

        if t == 0:
            dp_table[t] = []
            parent[t] = []
            continue

        # For this t, compute dp[t][s] for all s in 1..t-1
        gaps_ts = gap_matrix[:, t]  # shape (n,), valid entries 0..t-1

        dp_t = np.full(t, 2, dtype=np.int32)
        parent_t = np.full(t, -1, dtype=np.int32)

        for s in range(1, t):
            gap_ts = int(gaps_ts[s])
            qd = query_data[s]
            if qd is not None:
                sorted_gaps, prefix_max_dp, prefix_argmax_r = qd
                idx = br(sorted_gaps, gap_ts) - 1
                if idx >= 0:
                    best_dp = prefix_max_dp[idx]
                    if best_dp > 0:
                        dp_t[s] = 1 + best_dp
                        parent_t[s] = prefix_argmax_r[idx]

        # Find best in dp_t
        if t > 0:
            max_idx = int(np.argmax(dp_t))
            max_val = int(dp_t[max_idx])
            if max_val > best_val:
                best_val = max_val
                best_t = t
                best_s = max_idx

        dp_table[t] = dp_t.tolist()
        parent[t] = parent_t.tolist()

        # Build query_data[t] for future use
        if t > 0:
            # gaps for r in 0..t-1: primes[t] - primes[r]
            gaps_np = primes_arr[t] - primes_arr[:t]
            # Sort indices by gap value
            order = np.argsort(gaps_np)
            sorted_gaps = gaps_np[order].tolist()
            dp_vals = [dp_t[r] for r in order]
            r_vals = order.tolist()

            prefix_max = [0] * t
            prefix_arg = [-1] * t
            cur_max = 0
            cur_arg = -1
            for i in range(t):
                if dp_vals[i] > cur_max:
                    cur_max = dp_vals[i]
                    cur_arg = r_vals[i]
                prefix_max[i] = cur_max
                prefix_arg[i] = cur_arg

            query_data[t] = (sorted_gaps, prefix_max, prefix_arg)

    # Reconstruct
    if best_s == -1:
        return [primes_list[best_t]]

    seq = [best_t, best_s]
    ct = best_t
    cs = best_s
    while parent[ct] is not None and parent[ct][cs] != -1:
        ns = parent[ct][cs]
        seq.append(ns)
        ct = cs
        cs = ns

    seq.reverse()
    result = [primes_list[i] for i in seq]

    # Verify
    if len(result) >= 2:
        gaps = [result[i+1] - result[i] for i in range(len(result)-1)]
        if not all(gaps[i] <= gaps[i+1] for i in range(len(gaps)-1)):
            # Fallback: return a valid sequence
            result = [primes_list[0]]

    return result

# EVOLVE-BLOCK-END

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

    # DP: dp[t][s] = length of longest valid subsequence ending at primes[t]
    #      with immediate predecessor primes[s] (s < t)
    # Constraint: gaps are nondecreasing, so if ...primes[r], primes[s], primes[t]
    # then primes[s]-primes[r] <= primes[t]-primes[s]
    #
    # dp[t][s] = max(2, 1 + max{dp[s][r] : r < s, primes[s]-primes[r] <= primes[t]-primes[s]})
    #
    # For each s, we precompute a data structure to answer:
    #   query(g) = max{dp[s][r] : r < s, primes[s]-primes[r] <= g}
    # and which r achieved it (for backtracking).

    # Process t from 0 to n-1.
    # For each s < t, compute dp[t][s].
    # Then build the query structure for t.

    # dp_table[t] = list of length t: dp[t][s] for s in 0..t-1
    # parent[t][s] = r that achieved the max (or -1)
    # query_data[s] = (sorted_gaps, prefix_max_dp, prefix_argmax_r)
    #   where sorted_gaps[i] = primes[s] - primes[r_i] sorted
    #   prefix_max_dp[i] = max dp[s][r] for the first i+1 entries (sorted by gap)
    #   prefix_argmax_r[i] = the r that achieved prefix_max_dp[i]

    dp_table = [None] * n
    parent = [None] * n
    query_data = [None] * n  # query_data[s] for s >= 1

    best_val = 1
    best_t = 0
    best_s = -1

    for t in range(n):
        if time.time() > deadline:
            break

        if t == 0:
            dp_table[t] = []
            parent[t] = []
            continue

        dp_t = [0] * t
        parent_t = [-1] * t

        for s in range(t):
            gap_ts = primes_list[t] - primes_list[s]

            if s == 0:
                dp_t[s] = 2
                parent_t[s] = -1
            else:
                qd = query_data[s]
                if qd is None:
                    dp_t[s] = 2
                    parent_t[s] = -1
                else:
                    sorted_gaps, prefix_max_dp, prefix_argmax_r = qd
                    idx = bisect.bisect_right(sorted_gaps, gap_ts) - 1
                    if idx >= 0:
                        best_dp = prefix_max_dp[idx]
                        if best_dp > 0:
                            dp_t[s] = 1 + best_dp
                            parent_t[s] = prefix_argmax_r[idx]
                        else:
                            dp_t[s] = 2
                            parent_t[s] = -1
                    else:
                        dp_t[s] = 2
                        parent_t[s] = -1

            if dp_t[s] > best_val:
                best_val = dp_t[s]
                best_t = t
                best_s = s

        dp_table[t] = dp_t
        parent[t] = parent_t

        # Build query_data[t]
        if t > 0:
            gaps_list = []
            for r in range(t):
                g = primes_list[t] - primes_list[r]
                gaps_list.append((g, dp_t[r], r))
            gaps_list.sort(key=lambda x: x[0])

            sorted_gaps = [g for g, _, _ in gaps_list]
            dp_vals = [v for _, v, _ in gaps_list]
            r_vals = [r for _, _, r in gaps_list]

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

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    """
    Find the longest increasing subsequence of primes <= 10000
    where successive gaps are nondecreasing.
    
    Uses DP with state (current_prime_index, last_gap) optimized via
    the observation that we can track the best (longest) sequence ending
    at each prime with each possible last gap value.
    
    Key insight: dp[i] = dict mapping last_gap -> max_length
    For each prime p_i, for each previous prime p_j < p_i with gap g = p_i - p_j,
    we can extend any sequence ending at p_j with last_gap <= g.
    
    To optimize, for each prime i, we maintain:
    - best_len[i][g] = max length of valid sequence ending at prime i with last gap exactly g
    - We also maintain a prefix-max structure over gaps for quick lookup
    
    Since gaps are bounded (max gap between consecutive primes <= 10000 is ~200),
    and primes <= 10000 number ~1229, we can use an array-based approach.
    
    Actually, let's use a cleaner DP:
    dp[i][j] = max length of nondecreasing-gap prime subsequence ending at prime i
               with the last gap being exactly j (or at most j for prefix max)
    
    But gaps can be large. Let's think differently.
    
    For each pair (j, i) with j < i, gap = primes[i] - primes[j].
    dp[i] = list of (last_gap, max_length) pairs, where for each possible last_gap,
    we store the max length of a valid sequence ending at primes[i] with that last gap.
    
    Transition: for prime i, for each previous prime j < i:
        gap = primes[i] - primes[j]
        For each (prev_gap, length) in dp[j] where prev_gap <= gap:
            candidate_length = length + 1
            Update dp[i] for last_gap = gap with max(candidate_length)
    
    Also, a sequence of length 1 ending at i has no "last gap", so we initialize
    dp[i] with entries for all possible gaps >= (primes[i] - primes[0])... 
    Actually, simpler: initialize dp[i] with length 1 for all gaps >= 0 (conceptually).
    
    To make this efficient, for each i, we want:
    dp[i][g] = 1 + max{ dp[j][g'] : j < i, g' <= g, primes[j] + g == primes[i] }
              = 1 + max{ dp[j][g'] : j < i, primes[j] == primes[i] - g, g' <= g }
    
    For fixed i and gap g, the only j that works is j where primes[j] = primes[i] - g.
    So for each i, for each possible gap g (from 1 to primes[i] - primes[0]):
        prev_prime = primes[i] - g
        If prev_prime is in the primes list at index j:
            dp[i][g] = 1 + max{ dp[j][g'] : g' <= g }
    
    We can precompute for each prime i and each gap g whether primes[i]-g is a prime.
    
    Let's use:
    - primes: list of primes <= 10000
    - prime_to_idx: dict mapping prime value to index
    - dp[i] = array where dp[i][g] = max length ending at prime i with last gap exactly g
    - prefix_dp[i][g] = max{ dp[i][g'] : g' <= g }
    
    Max gap is primes[-1] - primes[0] = 9973 - 2 = 9971, but we only need gaps up to 
    primes[i] - 2 for each i.
    
    For efficiency, let's cap the gap array size. The maximum gap between any two primes
    <= 10000 is 9971, but most gaps are small. However, we need to handle all possible gaps.
    
    Let's use a list of dicts or sparse representation. Actually, with ~1229 primes and
    max gap ~10000, a dense array of size 10001 for each prime would be 1229 * 10001 ≈ 12M
    entries, which is feasible in Python if we use numpy or just lists.
    
    Let's use numpy for speed.
    """
    deadline = time.time() + 85  # 5s safety margin
    
    # Sieve of Eratosthenes
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(limit + 1) if is_prime[i]]
    N = len(primes)
    
    if N == 0:
        return []
    if N == 1:
        return [primes[0]]
    
    # Map prime value to index
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Max possible gap
    max_gap = primes[-1] - primes[0]
    
    # dp[i][g] = max length of valid sequence ending at primes[i] with last gap exactly g
    # We'll use a list of arrays, each of size max_gap + 1
    # Initialize with 0 (meaning no sequence ends here with that gap)
    # A sequence of length 1 ending at primes[i] can be considered to have any "last gap"
    # for the purpose of extending, but actually it has no last gap. Let's handle this by
    # saying that a sequence of length 1 can be extended by any gap >= 0.
    
    # Better approach: dp[i][g] = max length of valid sequence ending at primes[i] 
    # where the LAST gap is exactly g. For length-1 sequences, there's no last gap,
    # so we treat them specially.
    
    # Let's define:
    # best[i][g] = max length of a valid nondecreasing-gap prime subsequence ending at 
    #              primes[i] with last gap exactly g (for g >= 1)
    # A subsequence of length 1 ending at primes[i] has no last gap.
    
    # Transition:
    # For i, for each gap g >= 1:
    #   prev = primes[i] - g
    #   if prev >= 2 and prev in prime_to_idx:
    #       j = prime_to_idx[prev]
    #       # We can extend any sequence ending at j with last gap <= g
    #       # The max such length is prefix_max[j][g]
    #       best[i][g] = max(best[i][g], prefix_max[j][g] + 1)
    #   Also, best[i][g] >= 1 is not valid since length 1 has no gap.
    #   Actually, a sequence of length 2: (primes[j], primes[i]) has gap g.
    #   So best[i][g] >= 2 if primes[i]-g is a prime.
    
    # Let me redefine:
    # dp[i][g] = max length of valid sequence ending at primes[i] with last gap exactly g
    # Base: dp[i][g] = 2 if primes[i] - g is a prime (sequence of two primes)
    # Recurrence: dp[i][g] = max(2, 1 + max{dp[j][g'] : j < i, primes[j] = primes[i]-g, g' <= g})
    
    # prefix[i][g] = max{dp[i][g'] : 1 <= g' <= g}
    
    # We want to find the global max of dp[i][g] over all i, g.
    
    # Use numpy for the DP arrays
    import numpy as np
    
    dp = np.zeros((N, max_gap + 1), dtype=np.int32)
    prefix = np.zeros((N, max_gap + 1), dtype=np.int32)
    
    for i in range(N):
        pi = primes[i]
        # For each possible gap g
        # g ranges from 1 to pi - 2 (since smallest prime is 2)
        for g in range(1, pi - 1):
            if time.time() > deadline:
                break
            prev = pi - g
            if prev < 2:
                break
            if prev in prime_to_idx:
                j = prime_to_idx[prev]
                if j < i:  # should always be true since prev < pi
                    # dp[i][g] = max(dp[i][g], 1 + prefix[j][g])
                    val = 1 + prefix[j][g]
                    if val > dp[i][g]:
                        dp[i][g] = val
        # Compute prefix max for row i
        if max_gap > 0:
            prefix[i] = np.maximum.accumulate(dp[i])
        else:
            prefix[i] = dp[i]
    
    # Find the maximum length
    max_len = 0
    best_i = 0
    best_g = 0
    for i in range(N):
        row_max = np.max(dp[i]) if max_gap > 0 else 0
        if row_max > max_len:
            max_len = row_max
            best_i = i
            best_g = np.argmax(dp[i])
    
    if max_len <= 1:
        return [primes[0]]
    
    # Reconstruct the sequence
    # We need to backtrack. For that, we need to store the parent.
    # Let's redo with parent tracking.
    
    # This is getting complex. Let me use a different approach:
    # Store parent[i][g] = the index j that gave the max for dp[i][g]
    
    # Actually, let's just do a simpler approach given the constraints.
    # N ~ 1229, max_gap ~ 9971. The double loop is ~1229 * 5000 avg = ~6M iterations.
    # That should be fast enough.
    
    # Let me restart with parent tracking.
    
    dp = np.zeros((N, max_gap + 1), dtype=np.int32)
    prefix = np.zeros((N, max_gap + 1), dtype=np.int32)
    parent = np.full((N, max_gap + 1), -1, dtype=np.int32)
    
    for i in range(N):
        pi = primes[i]
        for g in range(1, pi - 1):
            if time.time() > deadline:
                break
            prev = pi - g
            if prev < 2:
                break
            if prev in prime_to_idx:
                j = prime_to_idx[prev]
                if j < i:
                    val = 1 + prefix[j][g]
                    if val > dp[i][g]:
                        dp[i][g] = val
                        parent[i][g] = j
        if max_gap > 0:
            prefix[i] = np.maximum.accumulate(dp[i])
    
    # Find max
    max_len = 0
    best_i = 0
    best_g = 0
    for i in range(N):
        if max_gap > 0:
            row_max = int(np.max(dp[i]))
        else:
            row_max = 0
        if row_max > max_len:
            max_len = row_max
            best_i = i
            best_g = int(np.argmax(dp[i]))
    
    if max_len <= 1:
        return [primes[0]]
    
    # Backtrack
    result = []
    i, g = best_i, best_g
    while i >= 0:
        result.append(primes[i])
        if parent[i][g] == -1:
            break
        # The gap from parent[i][g] to i is g
        prev_i = parent[i][g]
        # The last gap at prev_i was some g' <= g
        # We need to find g' such that dp[prev_i][g'] + 1 = dp[i][g]
        # and g' <= g
        # Since prefix[prev_i][g] = max{dp[prev_i][g'] : g' <= g}
        # We need to find the specific g' that achieved this max
        # Let's find the largest g' <= g such that dp[prev_i][g'] == prefix[prev_i][g]
        # Actually, any such g' works for backtracking
        target = dp[i][g] - 1
        # Find g' in [1, g] where dp[prev_i][g'] == target
        found = False
        for g2 in range(g, 0, -1):
            if dp[prev_i][g2] == target:
                g = g2
                found = True
                break
        if not found:
            # Fallback
            g = g  # shouldn't happen
        i = prev_i
    
    result.reverse()
    return result
# EVOLVE-BLOCK-END

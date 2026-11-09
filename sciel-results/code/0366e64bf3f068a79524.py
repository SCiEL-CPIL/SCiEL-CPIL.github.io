def construct():
    from math import isqrt
    import time
    import bisect

    deadline = time.time() + 55

    N = 10000
    is_prime = bytearray(b'\x01') * (N + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))

    primes = [i for i in range(2, N+1) if is_prime[i]]
    n = len(primes)

    # dp[i][j] = max length of valid subsequence ending at primes[i] with last gap = j
    # We'll use a list of dicts: dp[i] maps gap -> (length, prev_i, prev_gap)
    # But that's memory-heavy. Instead, for each i, we store a list of (gap, length, prev_i, prev_gap_idx)
    # sorted by gap, and we only keep Pareto-optimal states (for a given gap, max length)

    # Actually, let's use a different approach:
    # For each prime i, we want to find the best subsequence ending at i.
    # The state is (last_gap, length). We want to maximize length.
    # For a fixed i, if we have two states with gaps g1 < g2 and lengths l1 >= l2, 
    # then state (g2, l2) is dominated by (g1, l1) because any extension that works 
    # with g2 also works with g1 (since g1 <= g2 means the constraint is easier to satisfy).
    # Wait, that's not quite right. The constraint is that the NEW gap must be >= last_gap.
    # So if g1 < g2, then g1 is "better" (easier to extend) but might have shorter length.
    # We want to keep states that are not dominated: state A dominates state B if 
    # A.gap <= B.gap AND A.length >= B.length.

    # Let's store for each i a list of (gap, length, prev_i, prev_state_idx) 
    # sorted by gap, and pruned so that lengths are strictly increasing with gap.

    dp = [None] * n

    for i in range(n):
        if time.time() > deadline:
            break
        p_i = primes[i]
        # Start with the trivial state: subsequence [primes[i]], last_gap = 0 (or -inf)
        # We'll represent "no previous gap" as gap = -1
        candidates = [(-1, 1, -1, -1)]
        
        for j in range(i - 1, -1, -1):
            if time.time() > deadline:
                break
            new_gap = p_i - primes[j]
            dp_j = dp[j]
            if dp_j is None:
                continue
            
            # For each state in dp_j, if state.gap <= new_gap, we can extend it
            # dp_j is sorted by gap, so we can binary search for the rightmost gap <= new_gap
            # dp_j is a list of (gap, length, prev_i, prev_state_idx)
            # We want all states with gap <= new_gap
            
            # Find the rightmost index where gap <= new_gap
            # Since gaps are sorted, we can use bisect
            # But we need to find the last index where gap <= new_gap
            # Let's store gaps separately for bisect
            
            # Actually, let's just iterate. The list should be short due to pruning.
            for (g_j, l_j, prev_i, prev_s_idx) in dp_j:
                if g_j > new_gap:
                    break  # Since sorted by gap, no more states will work
                candidates.append((new_gap, l_j + 1, j, dp_j.index((g_j, l_j, prev_i, prev_s_idx))))
        
        # Now prune candidates: keep only non-dominated states
        # Sort by gap, then for same gap keep max length
        # Then remove states where a state with smaller gap has >= length
        
        if not candidates:
            dp[i] = [(-1, 1, -1, -1)]
            continue
            
        # Sort by gap, then by -length (so longer comes first for same gap)
        candidates.sort(key=lambda x: (x[0], -x[1]))
        
        # Keep only the best (max length) for each gap
        pruned = []
        last_gap = None
        for c in candidates:
            if c[0] != last_gap:
                pruned.append(c)
                last_gap = c[0]
            # else: skip, we already have a state with same gap and >= length
        
        # Now remove dominated states: if state A has gap <= state B.gap and length >= state B.length, B is dominated
        # Since sorted by gap, we can do a forward pass keeping max length so far
        # But we want to keep states where length is strictly greater than all previous states with smaller gap
        # Actually, we want: for increasing gap, length must be strictly increasing to be non-dominated
        
        final_states = []
        max_len_so_far = -1
        for (g, l, pi, psi) in pruned:
            if l > max_len_so_far:
                final_states.append((g, l, pi, psi))
                max_len_so_far = l
            # else: dominated by a state with smaller gap and >= length
        
        if not final_states:
            final_states = [(-1, 1, -1, -1)]
        
        dp[i] = final_states

    # Find the best state overall
    best_len = 0
    best_i = -1
    best_state_idx = -1
    for i in range(n):
        if dp[i] is None:
            continue
        for idx, (g, l, pi, psi) in enumerate(dp[i]):
            if l > best_len:
                best_len = l
                best_i = i
                best_state_idx = idx

    if best_i == -1:
        return [2]

    # Reconstruct
    result = []
    cur_i = best_i
    cur_idx = best_state_idx
    while cur_i != -1:
        result.append(primes[cur_i])
        g, l, pi, psi = dp[cur_i][cur_idx]
        cur_i = pi
        cur_idx = psi

    result.reverse()
    return result

import time

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill

    # Sieve primes up to 10000
    N = 10000
    is_prime = bytearray(b'\x01') * (N + 1)
    is_prime[0] = is_prime[1] = 0
    for x in range(2, int(N ** 0.5) + 1):
        if is_prime[x]:
            is_prime[x * x::x] = bytearray(len(is_prime[x * x::x]))
    
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    n = len(primes)
    
    if n == 0:
        return []
    if n == 1:
        return primes
    
    # dp[i] is a list of states (last_gap, length, prev_idx, prev_state_idx)
    # prev_idx is the index j in primes
    # prev_state_idx is the index of the state in dp[j] that was extended
    # We prune states such that if state A has gap <= state B's gap and length >= state B's length, 
    # state B is dominated and removed.
    
    dp = []
    best_len_global = 1
    best_i_global = 0
    best_k_global = 0
    
    for i in range(n):
        if time.time() - t0 > DEADLINE:
            break
            
        p_i = primes[i]
        candidates = []
        
        # Base case: sequence of length 1 ending at primes[i]
        # last_gap = 0, length = 1, no prev
        candidates.append((0, 1, -1, -1))
        
        # Extend from previous primes
        for j in range(i):
            p_j = primes[j]
            gap = p_i - p_j
            
            # If gap is 0, skip (primes are distinct)
            if gap == 0:
                continue
                
            # For each state in dp[j], check if we can extend
            # We iterate over the pruned list of dp[j]
            for k, state in enumerate(dp[j]):
                g_j, L_j, _, _ = state
                if gap >= g_j:
                    # New state for i
                    candidates.append((gap, L_j + 1, j, k))
                    if L_j + 1 > best_len_global:
                        best_len_global = L_j + 1
                        best_i_global = i
                        best_k_global = len(candidates) - 1 # Tentative, will be fixed after pruning if this state survives
                    
        # Prune candidates
        # Sort by gap ascending, then by length descending
        # This allows us to easily find the max length for a given gap or smaller
        candidates.sort(key=lambda x: (x[0], -x[1]))
        
        pruned = []
        max_len_so_far = 0
        
        for c in candidates:
            gap, length, pj, ps = c
            # If current length is greater than the max length seen so far for smaller gaps,
            # it is not dominated.
            if length > max_len_so_far:
                pruned.append(c)
                max_len_so_far = length
            # If length <= max_len_so_far, it is dominated by a previous state 
            # with smaller (or equal) gap and >= length
        
        dp.append(pruned)
        
        # Update global best if this prime has a better state than the current global best
        # We need to check the best state in pruned for i
        if pruned:
            # The last element in pruned has the max length because we sorted by gap asc, 
            # and we only added if length increased. 
            # Wait, sorting by gap asc means the last element has the largest gap.
            # But we prune based on max_len_so_far.
            # The state with the absolute max length in pruned might not be the last one?
            # No, max_len_so_far is monotonically increasing as we iterate through sorted candidates.
            # So the last element in pruned has the maximum length for this prime i.
            last_state = pruned[-1]
            if last_state[1] > best_len_global:
                best_len_global = last_state[1]
                best_i_global = i
                best_k_global = len(pruned) - 1
            elif last_state[1] == best_len_global:
                 # Tie-breaking not strictly necessary for score, but let's keep the first found or last?
                 # Doesn't matter for validity.
                 pass

    # Reconstruct the best sequence found
    if best_i_global == -1 or not dp:
        return [primes[0]] if n > 0 else []
        
    seq_indices = []
    curr_i = best_i_global
    curr_k = best_k_global
    
    # Ensure curr_k is valid for curr_i
    if curr_i < len(dp) and curr_k < len(dp[curr_i]):
        while curr_i != -1:
            seq_indices.append(curr_i)
            if curr_i == -1:
                break
            state = dp[curr_i][curr_k]
            curr_i = state[2]
            curr_k = state[3]
    else:
        # Fallback if indices got messed up (shouldn't happen with careful tracking)
        return [primes[0]] if n > 0 else []

    seq_indices.reverse()
    result = [primes[idx] for idx in seq_indices]
    
    return result

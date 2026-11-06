def construct():
    import time
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin
    
    # 1. Sieve of Eratosthenes
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, val in enumerate(is_prime) if val]
    N = len(primes)
    
    if N == 0:
        return []
    
    # states[i] is a list of tuples: (gap, length, prev_prime_idx, prev_state_idx)
    # gap: the gap between primes[i] and primes[prev_prime_idx]
    # length: length of the subsequence ending at primes[i] with this gap
    # prev_prime_idx: index of the previous prime in the subsequence
    # prev_state_idx: index of the state in states[prev_prime_idx]
    
    states = [[] for _ in range(N)]
    
    # Base case: subsequence of length 1 at primes[0]
    # Gap is 0 (conceptually, no previous prime)
    states[0].append((0, 1, -1, -1))
    
    best_len = 1
    best_end_idx = 0
    best_state_idx = 0
    
    for i in range(1, N):
        if time.time() > DEADLINE:
            break
            
        pi = primes[i]
        candidates = []
        
        # Option 1: Start new subsequence at i
        # Gap 0, Length 1
        candidates.append((0, 1, -1, -1))
        
        # Option 2: Extend from any j < i
        for j in range(i):
            if time.time() > DEADLINE:
                break
            pj = primes[j]
            gap = pi - pj
            
            # If gap is 0, it's the same prime, but primes are distinct, so gap > 0
            # We can optimize: if gap is very small, it might not extend many states
            # But we just check all states in states[j]
            
            for k, state in enumerate(states[j]):
                g_j, l_j, _, _ = state
                if gap >= g_j:
                    # Can extend
                    candidates.append((gap, l_j + 1, j, k))
        
        if not candidates:
            states[i] = []
            continue
            
        # Prune candidates
        # Sort by gap, then by length (descending? No, we just filter)
        # We want to keep states where length is strictly increasing as gap increases.
        candidates.sort(key=lambda x: (x[0], -x[1]))
        
        pruned = []
        max_len_so_far = 0
        
        for c in candidates:
            gap, length, p_idx, s_idx = c
            if length > max_len_so_far:
                pruned.append(c)
                max_len_so_far = length
                if length > best_len:
                    best_len = length
                    best_end_idx = i
                    best_state_idx = len(pruned) - 1
        states[i] = pruned

    # Reconstruct sequence
    if best_len == 1:
        return [primes[best_end_idx]]
        
    seq_indices = []
    curr_i = best_end_idx
    curr_k = best_state_idx
    
    while curr_i != -1:
        seq_indices.append(curr_i)
        if curr_k == -1:
            break
        state = states[curr_i][curr_k]
        _, _, prev_i, prev_k = state
        curr_i = prev_i
        curr_k = prev_k
        
    seq_indices.reverse()
    
    return [primes[idx] for idx in seq_indices]

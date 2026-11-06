"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # 1. Generate primes up to 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
            
    primes = [p for p in range(2, limit + 1) if is_prime[p]]
    N = len(primes)
    
    if N == 0:
        return []
    if N == 1:
        return primes
    
    # dp[i] is a list of (gap, length) tuples, representing non-dominated states
    # ending at prime i.
    # A state (g1, l1) dominates (g2, l2) if g1 <= g2 and l1 >= l2.
    # We maintain the list sorted by gap increasing. Since it's Pareto optimal,
    # lengths must be strictly decreasing as gaps increase.
    
    dp = [[] for _ in range(N)]
    # parent[i][k] = (prev_prime_idx, prev_state_idx)
    # We store parent info parallel to dp[i]
    parents = [None for _ in range(N)]
    
    for j in range(N):
        if time.time() > deadline:
            break
            
        # Initialize with the state of just the prime itself (length 1, gap 0)
        # Actually, a single prime has no gap constraint for the "next" step 
        # except that the next gap must be >= 0.
        # Let's define the state for a single prime p_j as (0, 1).
        # When extending from i to j, the new gap is p_j - p_i.
        # The condition is p_j - p_i >= last_gap_of_i.
        
        new_states = []
        new_parents = []
        
        # State for starting at j
        # (gap=0, length=1)
        # Is it better to start with gap 0?
        # If we have a sequence ... p_i, p_j, the gap is p_j - p_i.
        # If the sequence is just p_j, there is no previous gap.
        # For the purpose of extending to p_k, the constraint is p_k - p_j >= gap(prev, j).
        # If j is the start, there is no constraint on the next gap other than being positive (which it is).
        # So we can model the "last gap" of a length-1 sequence as 0.
        
        candidates = [(0, 1, -1, -1)] # (gap, length, prev_idx, prev_state_idx)
        
        for i in range(j):
            if time.time() > deadline:
                break
            gap = primes[j] - primes[i]
            # For each state in dp[i], check if we can extend
            for k, (g_prev, l_prev) in enumerate(dp[i]):
                if gap >= g_prev:
                    # We can extend
                    new_len = l_prev + 1
                    # The new state at j has gap 'gap' and length 'new_len'
                    # Parent is i, state k
                    candidates.append((gap, new_len, i, k))
        
        # Now we have a list of candidate states for j.
        # We need to filter out dominated states.
        # Sort by gap ascending, then by length descending.
        # Iterate and keep only those with length greater than any previous length seen.
        
        # Remove duplicates? Not strictly necessary for correctness if we handle dominance,
        # but reduces size.
        # Sort candidates: primary key gap (asc), secondary key length (desc)
        candidates.sort(key=lambda x: (x[0], -x[1]))
        
        filtered_states = []
        filtered_parents = []
        max_len_so_far = 0
        
        for gap, length, prev_idx, prev_state_idx in candidates:
            if length > max_len_so_far:
                # This state is not dominated by any previous state with smaller/equal gap
                # because its length is strictly larger.
                # Note: If gap is same as previous, and length is larger, it dominates previous.
                # But we sorted by gap asc, length desc.
                # So if gap is same, the first one has the largest length.
                # Subsequent ones with same gap have smaller length, so length <= max_len_so_far (which is the first one's length).
                # So they will be skipped. Correct.
                
                filtered_states.append((gap, length))
                filtered_parents.append((prev_idx, prev_state_idx))
                max_len_so_far = length
        
        dp[j] = filtered_states
        parents[j] = filtered_parents
    
    # Find the best state
    best_len = 0
    best_j = -1
    best_k = -1
    
    for j in range(N):
        for k, (g, l) in enumerate(dp[j]):
            if l > best_len:
                best_len = l
                best_j = j
                best_k = k
    
    # Reconstruct
    if best_j == -1:
        return []
        
    seq = []
    j = best_j
    k = best_k
    while j != -1:
        seq.append(primes[j])
        if k == -1:
            break
        prev_idx, prev_state_idx = parents[j][k]
        j = prev_idx
        k = prev_state_idx
        
    seq.reverse()
    return seq

# EVOLVE-BLOCK-END

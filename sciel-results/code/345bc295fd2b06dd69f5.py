import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # Sieve of Eratosthenes to get primes up to 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    primes = [p for p in range(2, limit + 1) if is_prime[p]]
    n = len(primes)
    
    # Segment tree for range maximum query over gap values
    # Gaps range from 1 to 10000
    max_gap = 10000
    size = 1
    while size < max_gap + 1:
        size *= 2
    
    # Tree stores (length, prime_index, state_idx) for each gap position
    # -1 means no valid state
    tree_len = [-1] * (2 * size)
    tree_prime_idx = [-1] * (2 * size)
    tree_state_idx = [-1] * (2 * size)
    
    def update(pos, length, prime_idx, state_idx):
        """Update position pos in the segment tree"""
        pos += size
        if length > tree_len[pos]:
            tree_len[pos] = length
            tree_prime_idx[pos] = prime_idx
            tree_state_idx[pos] = state_idx
        while pos > 1:
            pos //= 2
            left = 2 * pos
            right = 2 * pos + 1
            # Choose the child with larger length; break ties by higher prime_idx
            if tree_len[left] > tree_len[right] or (tree_len[left] == tree_len[right] and tree_prime_idx[left] > tree_prime_idx[right]):
                if tree_len[left] > tree_len[pos] or (tree_len[left] == tree_len[pos] and tree_prime_idx[left] > tree_prime_idx[pos]):
                    tree_len[pos] = tree_len[left]
                    tree_prime_idx[pos] = tree_prime_idx[left]
                    tree_state_idx[pos] = tree_state_idx[left]
            else:
                if tree_len[right] > tree_len[pos] or (tree_len[right] == tree_len[pos] and tree_prime_idx[right] > tree_prime_idx[pos]):
                    tree_len[pos] = tree_len[right]
                    tree_prime_idx[pos] = tree_prime_idx[right]
                    tree_state_idx[pos] = tree_state_idx[right]
    
    def query(l, r):
        """Query max length in range [l, r] inclusive"""
        if l > r:
            return (-1, -1, -1)
        l += size
        r += size
        best_len = -1
        best_prime_idx = -1
        best_state_idx = -1
        while l <= r:
            if l % 2 == 1:
                if tree_len[l] > best_len or (tree_len[l] == best_len and tree_prime_idx[l] > best_prime_idx):
                    best_len = tree_len[l]
                    best_prime_idx = tree_prime_idx[l]
                    best_state_idx = tree_state_idx[l]
                l += 1
            if r % 2 == 0:
                if tree_len[r] > best_len or (tree_len[r] == best_len and tree_prime_idx[r] > best_prime_idx):
                    best_len = tree_len[r]
                    best_prime_idx = tree_prime_idx[r]
                    best_state_idx = tree_state_idx[r]
                r -= 1
            l //= 2
            r //= 2
        return (best_len, best_prime_idx, best_state_idx)
    
    # dp[i] is a list of non-dominated states ending at primes[i]
    # State: (length, last_gap, prev_prime_idx, prev_state_idx)
    dp = [None] * n
    dp[0] = [(1, 0, -1, -1)]
    
    best_len = 1
    best_i = 0
    best_state_idx = 0
    
    for i in range(1, n):
        if time.time() > deadline:
            break
            
        candidates = [(1, 0, -1, -1)]  # Start new sequence at i
        
        for j in range(i):
            gap = primes[i] - primes[j]
            # Query segment tree for best state with last_gap <= gap
            q_len, q_prime_idx, q_state_idx = query(0, gap)
            
            if q_len > 0:
                # Found a valid predecessor at prime q_prime_idx with state q_state_idx
                new_state = (q_len + 1, gap, q_prime_idx, q_state_idx)
                candidates.append(new_state)
        
        # Prune candidates to keep only non-dominated states
        # Sort by length descending, then by gap ascending
        candidates.sort(key=lambda x: (-x[0], x[1]))
        
        non_dom = []
        min_gap_encountered = float('inf')
        for cand in candidates:
            L, G, pi, si = cand
            if G < min_gap_encountered:
                non_dom.append(cand)
                min_gap_encountered = G
        
        dp[i] = non_dom
        
        # Update global best
        current_best_len = non_dom[0][0]
        if current_best_len > best_len:
            best_len = current_best_len
            best_i = i
            best_state_idx = 0
        
        # Update segment tree with states ending at primes[i]
        for state_idx, (L, G, pi, si) in enumerate(non_dom):
            if G <= max_gap:
                update(G, L, i, state_idx)
    
    # Reconstruct the subsequence
    seq = []
    i = best_i
    state_idx = best_state_idx
    while i != -1:
        L, G, prev_i, prev_state_idx = dp[i][state_idx]
        seq.append(primes[i])
        i = prev_i
        state_idx = prev_state_idx
        
    seq.reverse()
    return seq

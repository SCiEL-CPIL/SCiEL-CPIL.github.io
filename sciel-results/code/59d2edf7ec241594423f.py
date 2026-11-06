import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

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
    
    # dp[i] is a list of tuples (length, last_gap, prev_idx, prev_state_idx)
    # representing non-dominated states ending at primes[i]
    dp = [None] * n
    
    # Initialize dp[0]
    dp[0] = [(1, 0, -1, -1)]
    
    best_len = 1
    best_i = 0
    best_state_idx = 0
    
    for i in range(1, n):
        if time.time() > deadline:
            break
            
        candidates = []
        
        # Also consider starting a new sequence at i (length 1)
        # This is implicitly handled if we don't find any extension, 
        # but we must ensure (1, 0, -1, -1) is a candidate.
        # Actually, the loop below only adds extensions. 
        # A sequence of length 1 has no previous gap, so last_gap=0 is a convention.
        # But we can't extend a length 1 sequence from a previous prime j 
        # because there is no gap before the first element of the subsequence.
        # The gap constraint applies to successive gaps IN the subsequence.
        # So p_j -> p_i is the first gap. There is no constraint on the first gap.
        # So any prime j < i can be the predecessor of i, forming a sequence of length L_j + 1.
        # The new gap is p_i - p_j.
        # The previous last gap was G_j. The condition is G_j <= p_i - p_j.
        # This is exactly what the loop does.
        
        # What if we start a new sequence at i?
        # That corresponds to a sequence of length 1.
        # We should always include (1, 0, -1, -1) as a candidate for dp[i].
        candidates.append((1, 0, -1, -1))
        
        for j in range(i):
            gap = primes[i] - primes[j]
            # Check all states in dp[j]
            for k, (L, G, pi, si) in enumerate(dp[j]):
                if G <= gap:
                    candidates.append((L + 1, gap, j, k))
        
        # Prune candidates
        # Sort by length descending, then gap ascending
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
        # The first element in non_dom has the maximum length
        current_best_len = non_dom[0][0]
        if current_best_len > best_len:
            best_len = current_best_len
            best_i = i
            best_state_idx = 0

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

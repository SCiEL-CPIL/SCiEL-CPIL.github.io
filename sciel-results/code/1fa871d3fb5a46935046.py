import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s limit, 10s margin
    
    # Sieve of Eratosthenes
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, flag in enumerate(is_prime) if flag]
    n = len(primes)
    
    # DP: dp[i] = (max_len, min_last_gap, prev_idx)
    # We want to maximize length, and for each length minimize the last gap to allow future extensions
    # Actually, we need to keep track of multiple states per index i because different (length, gap) pairs may be non-dominated.
    # But for efficiency, we can use the fact that for a fixed i, if we have two states (l1, g1) and (l2, g2) with l1 >= l2 and g1 <= g2, then (l2, g2) is dominated.
    # So we keep only the Pareto frontier of (length, gap) for each i.
    
    # Let's use a list of (length, gap, prev_idx) for each i, keeping only non-dominated states.
    states = [[] for _ in range(n)]
    states[0] = [(1, 0, -1)]
    
    max_len = 1
    max_idx = 0
    
    for i in range(1, n):
        if time.time() > deadline:
            break
            
        pi = primes[i]
        new_states = []
        
        for j in range(i):
            gap = pi - primes[j]
            
            for l, g, prev_idx in states[j]:
                if gap >= g:
                    new_states.append((l + 1, gap, j))
        
        if not new_states:
            new_states.append((1, 0, -1))
        
        # Reduce to non-dominated: sort by length desc, then gap asc; keep only those with strictly decreasing gap
        new_states.sort(key=lambda x: (-x[0], x[1]))
        reduced = []
        min_gap = float('inf')
        for l, g, prev_idx in new_states:
            if g < min_gap:
                reduced.append((l, g, prev_idx))
                min_gap = g
        
        states[i] = reduced
        
        if reduced:
            curr_max_len = reduced[0][0]
            if curr_max_len > max_len:
                max_len = curr_max_len
                max_idx = i
    
    # Reconstruct
    best_state = None
    for s in states[max_idx]:
        if s[0] == max_len:
            best_state = s
            break
            
    if not best_state:
        return primes[:max_len]
    
    seq = []
    curr_idx = max_idx
    while curr_idx != -1:
        seq.append(primes[curr_idx])
        prev_idx = best_state[2]
        curr_len = best_state[0]
        
        if prev_idx == -1:
            break
            
        next_best_state = None
        for s in states[prev_idx]:
            if s[0] == curr_len - 1:
                next_best_state = s
                break
                
        if next_best_state is None:
            break
            
        best_state = next_best_state
        curr_idx = prev_idx
        
    seq.reverse()
    return seq

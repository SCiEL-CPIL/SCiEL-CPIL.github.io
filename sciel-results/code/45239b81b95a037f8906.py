import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin (90s limit)
    
    # 1. Generate primes <= 10000
    limit = 10000
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for x in range(2, int(limit ** 0.5) + 1):
        if is_prime[x]:
            is_prime[x * x::x] = bytearray(len(is_prime[x * x::x]))
    
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    n = len(primes)
    
    if n == 0:
        return []
    if n == 1:
        return primes
    
    # 2. DP with state lists
    # For each prime i, dp[i] is a list of (last_gap, length, prev_prime_idx, prev_state_idx)
    # We prune: sort by gap ascending, keep only states where length is strictly increasing
    # This means for any gap g, the best length achievable is from the last state with gap <= g
    
    dp = []  # dp[i] = list of pruned states for prime i
    
    best_len = 1
    best_i = 0
    best_state_idx = 0
    
    for i in range(n):
        if time.time() > deadline:
            break
        
        pi = primes[i]
        candidates = []
        
        # Base case: sequence of length 1 ending at primes[i], gap = 0
        candidates.append((0, 1, -1, -1))
        
        # Extend from all previous primes j < i
        for j in range(i):
            gap = pi - primes[j]
            # For each state in dp[j], check if gap >= state's last_gap
            # If so, we can extend: new_length = state_length + 1
            # We want the maximum length for states with last_gap <= gap
            # Since dp[j] is sorted by gap and pruned (length increasing),
            # we can binary search for the last state with gap <= our gap
            
            states_j = dp[j]
            if not states_j:
                continue
            
            # Binary search: find rightmost index where states_j[idx][0] <= gap
            lo, hi = 0, len(states_j)
            while lo < hi:
                mid = (lo + hi) // 2
                if states_j[mid][0] <= gap:
                    lo = mid + 1
                else:
                    hi = mid
            # lo is now the count of states with gap <= our gap
            # The best length among those is states_j[lo-1][1] (last one, since length is increasing)
            if lo > 0:
                best_state = states_j[lo - 1]
                new_len = best_state[1] + 1
                candidates.append((gap, new_len, j, lo - 1))
        
        # Prune candidates: sort by gap ascending, keep only strictly increasing lengths
        candidates.sort(key=lambda x: x[0])
        
        pruned = []
        max_len_so_far = 0
        for c in candidates:
            gap, length, pj, ps = c
            if length > max_len_so_far:
                pruned.append(c)
                max_len_so_far = length
            # If length <= max_len_so_far, dominated by earlier state with smaller gap
        
        dp.append(pruned)
        
        # Update global best
        if pruned:
            last_state = pruned[-1]
            if last_state[1] > best_len:
                best_len = last_state[1]
                best_i = i
                best_state_idx = len(pruned) - 1
    
    # 3. Reconstruct sequence
    if best_len <= 1:
        return [primes[best_i]]
    
    seq_indices = []
    curr_i = best_i
    curr_k = best_state_idx
    
    while curr_i != -1:
        seq_indices.append(curr_i)
        gap, length, pj, ps = dp[curr_i][curr_k]
        curr_i = pj
        curr_k = ps
    
    seq_indices.reverse()
    result = [primes[idx] for idx in seq_indices]
    
    return result

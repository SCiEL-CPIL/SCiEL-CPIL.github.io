import sys
import time

def construct():
    start_time = time.time()
    limit = 90  # seconds
    
    # Sieve of Eratosthenes to find primes up to 10000
    N_LIMIT = 10000
    is_prime = [True] * (N_LIMIT + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(N_LIMIT**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N_LIMIT + 1, i):
                is_prime[j] = False
    
    primes = [i for i, prime in enumerate(is_prime) if prime]
    n_primes = len(primes)
    
    # dp[i] is a list of tuples (gap, length, prev_index)
    # Sorted by gap. For each gap, length is strictly increasing (Pareto frontier).
    dp = [None] * n_primes
    
    max_len = 1
    best_end_idx = 0
    best_state = None
    
    for i in range(n_primes):
        p_i = primes[i]
        candidates = []
        
        # Option 1: Start a new sequence with just p_i
        # State: last_gap=0, length=1, prev_index=-1
        candidates.append((0, 1, -1))
        
        # Option 2: Extend sequences ending at primes[j] for j < i
        for j in range(i):
            p_j = primes[j]
            gap = p_i - p_j
            
            states_j = dp[j]
            if not states_j:
                continue
            
            # Binary search for the largest gap <= current gap in states_j
            # states_j is sorted by gap.
            # We want the rightmost index where gap_val <= gap
            lo, hi = 0, len(states_j) - 1
            best_idx = -1
            while lo <= hi:
                mid = (lo + hi) // 2
                if states_j[mid][0] <= gap:
                    best_idx = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            
            if best_idx != -1:
                g, l, prev = states_j[best_idx]
                # Since states_j is Pareto sorted (increasing gap -> increasing length),
                # the state at best_idx has the maximum length for gaps <= gap.
                candidates.append((gap, l + 1, j))
        
        # Prune candidates to Pareto frontier
        # Sort by gap
        candidates.sort(key=lambda x: x[0])
        
        pruned = []
        max_l_so_far = 0
        for g, l, prev in candidates:
            if l > max_l_so_far:
                pruned.append((g, l, prev))
                max_l_so_far = l
                if l > max_len:
                    max_len = l
                    best_end_idx = i
                    best_state = (g, l, prev)
        
        dp[i] = pruned
        
        # Time check (safety)
        if time.time() - start_time > limit - 5:
            break
    
    # Reconstruct the sequence
    if best_state is None:
        return [primes[0]]
    
    seq_indices = []
    curr_idx = best_end_idx
    curr_state = best_state
    
    while curr_idx != -1:
        seq_indices.append(curr_idx)
        # Find the state in dp[curr_idx] that corresponds to curr_state's prev
        # Actually, we stored prev_index in the state.
        # The state in dp[curr_idx] is (gap, len, prev).
        # We need to find the state in dp[prev] that was used.
        # But we only stored the prev_index.
        # We need to trace back.
        # The state we are at is (g, l, prev).
        # The previous state was the one in dp[prev] that generated this.
        # It was the state with the largest length such that its gap <= g_prev?
        # No, the transition was:
        # state_j (g_j, l_j, prev_j) -> state_i (g_i, l_i, j) where g_i = p_i - p_j and g_i >= g_j.
        # We picked the state_j with max l_j subject to g_j <= g_i.
        # So to go back from state_i to state_j, we look at dp[j].
        # We need to find the state in dp[j] that has length l_j = l_i - 1.
        # Is it unique?
        # In dp[j], lengths are strictly increasing with gaps.
        # So there is exactly one state in dp[j] with length l_j.
        # We can find it by searching for length l_j in dp[j].
        
        prev_idx = curr_state[2]
        if prev_idx == -1:
            break
        
        target_len = curr_state[1] - 1
        # Find state in dp[prev_idx] with length == target_len
        # dp[prev_idx] is sorted by gap, and length increases with gap.
        # We can binary search by length? Or just linear scan since list is small.
        # Or store index of length?
        # Linear scan is fine.
        found_prev_state = None
        for state in dp[prev_idx]:
            if state[1] == target_len:
                found_prev_state = state
                break
        
        if found_prev_state is None:
            # This shouldn't happen if logic is correct
            break
            
        curr_idx = prev_idx
        curr_state = found_prev_state
        
    seq_indices.reverse()
    
    return [primes[idx] for idx in seq_indices]

if __name__ == "__main__":
    print(construct())

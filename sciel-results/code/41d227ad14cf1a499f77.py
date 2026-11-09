"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import bisect

def construct():
    deadline = time.time() + 85.0
    from math import isqrt
    
    # Generate all primes <= 10000 using sieve
    sieve = [True] * 10001
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(10000) + 1):
        if sieve[i]:
            for j in range(i*i, 10001, i):
                sieve[j] = False
    primes = [p for p in range(2, 10001) if sieve[p]]
    n = len(primes)
    
    # Pareto Frontier DP
    # frontiers[i] is a list of (last_gap, length, parent_idx, parent_state_idx)
    # Sorted by last_gap. Lengths are strictly increasing.
    frontiers = [None] * n
    frontiers[0] = [(0, 1, -1, -1)]
    
    max_len_found = 1
    max_idx_found = 0
    max_state_idx_found = 0
    
    for i in range(1, n):
        if time.time() > deadline - 5.0:
            break
        pi = primes[i]
        current_states = []
        
        for j in range(i):
            gap = pi - primes[j]
            fj = frontiers[j]
            if not fj:
                continue
            
            # Binary search: find rightmost state with gap <= current gap
            # Frontiers are sorted by gap, lengths strictly increasing
            lo, hi = 0, len(fj) - 1
            best_len = -1
            best_parent_idx = -1
            best_parent_state_idx = -1
            
            while lo <= hi:
                mid = (lo + hi) // 2
                g, l, p_idx, p_s_idx = fj[mid]
                if g <= gap:
                    best_len = l
                    best_parent_idx = p_idx
                    best_parent_state_idx = p_s_idx
                    lo = mid + 1
                else:
                    hi = mid - 1
            
            if best_len != -1:
                new_len = best_len + 1
                new_state = (gap, new_len, j, best_parent_state_idx)
                current_states.append(new_state)
        
        if time.time() > deadline - 5.0:
            break
        
        if not current_states:
            frontiers[i] = []
            continue
            
        # Sort by gap
        current_states.sort(key=lambda x: x[0])
        
        pruned = []
        max_len_so_far = -1
        for state in current_states:
            g, l, p_idx, p_s_idx = state
            if l > max_len_so_far:
                pruned.append(state)
                max_len_so_far = l
        
        frontiers[i] = pruned
        
        if max_len_so_far > max_len_found:
            max_len_found = max_len_so_far
            max_idx_found = i
            max_state_idx_found = len(pruned) - 1

    # Reconstruct
    result = []
    idx = max_idx_found
    state_idx = max_state_idx_found
    while idx != -1:
        result.append(primes[idx])
        state = frontiers[idx][state_idx]
        p_idx = state[2]
        p_s_idx = state[3]
        idx = p_idx
        state_idx = p_s_idx
    result.reverse()
    
    return result
# EVOLVE-BLOCK-END

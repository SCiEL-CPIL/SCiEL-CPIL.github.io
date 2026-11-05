import time
import sys
import bisect

def construct():
    start_time = time.time()
    LIMIT = 10000
    TIME_LIMIT = 85.0  # Safety margin for 90s kill
    
    # 1. Generate primes
    is_prime = [True] * (LIMIT + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(LIMIT**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, LIMIT + 1, i):
                is_prime[j] = False
            
    primes = [i for i, prime in enumerate(is_prime) if prime]
    n = len(primes)
    
    best_seq = [primes[0]] if primes else []
    best_len = len(best_seq)
    
    # Recursive backtracking with limited branching factor
    sys.setrecursionlimit(10000)
    K = 10  # Branching factor
    
    def dfs(p_prev, g_prev, current_seq):
        nonlocal best_len, best_seq
        
        curr_len = len(current_seq)
        
        # Pruning: if even with min gap we can't beat best_len
        min_gap = max(1, g_prev)
        max_additional = (LIMIT - p_prev) // min_gap
        if curr_len + max_additional <= best_len:
            return
            
        if time.time() - start_time > TIME_LIMIT:
            return

        # Determine minimum value for next prime
        if g_prev == 0:
            min_next_val = p_prev + 1
        else:
            min_next_val = p_prev + g_prev
            
        # Find the index of the first prime >= min_next_val
        idx = bisect.bisect_left(primes, min_next_val)
        
        if idx >= n:
            return
            
        # Try the first K candidates
        for k in range(K):
            j = idx + k
            if j >= n:
                break
                
            p_next = primes[j]
            g_next = p_next - p_prev
            
            # Check if this branch can beat best
            new_min_gap = max(1, g_next)
            new_max_additional = (LIMIT - p_next) // new_min_gap
            
            if curr_len + 1 + new_max_additional <= best_len:
                continue
                
            # Append to current_seq
            current_seq.append(p_next)
            
            # Update best if longer
            if curr_len + 1 > best_len:
                best_len = curr_len + 1
                best_seq = current_seq[:]
                
            # Recurse
            dfs(p_next, g_next, current_seq)
            
            # Pop from current_seq
            current_seq.pop()
            
            # Time check
            if time.time() - start_time > TIME_LIMIT:
                return

    # Start DFS from each prime as the first element
    for i in range(n):
        p_start = primes[i]
        
        # Bound: max possible length starting here
        if (LIMIT - p_start) < best_len:
            break
            
        dfs(p_start, 0, [p_start])
        
        if time.time() - start_time > TIME_LIMIT:
            break

    return best_seq


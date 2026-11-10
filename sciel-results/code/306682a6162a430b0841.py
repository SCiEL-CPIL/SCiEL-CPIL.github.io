import time
import sys

def construct():
    # Deadline guard: 90s limit, use 85s to be safe
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 1000000
    MAX_N = 1000001
    
    # Sieve for Largest Prime Factor
    lpf = [1] * (MAX_N + 1)
    
    # Standard sieve to find largest prime factor
    # We iterate p from 2 to MAX_N
    # If lpf[p] is 1, p is prime
    # Then for all multiples j of p, set lpf[j] = p
    # Since we go in increasing order, the last prime that divides j is the largest.
    
    for p in range(2, MAX_N + 1):
        if lpf[p] == 1:
            # p is prime
            for j in range(p, MAX_N + 1, p):
                lpf[j] = p
                
    # Check time
    if time.time() > deadline:
        # Fallback to seed if sieve took too long (unlikely)
        return 10000

    best_n = 10000
    min_val = float('inf')
    
    # Search for n in [10000, 1000000]
    # We need lpf[n] and lpf[n+1]
    for n in range(10000, 1000001):
        # Check time periodically to avoid hard kill, though loop is fast
        if n % 100000 == 0 and time.time() > deadline:
            break
            
        val_n = lpf[n]
        val_n1 = lpf[n+1]
        
        # The largest prime factor of n(n+1) is max(lpf[n], lpf[n+1])
        current_max = val_n if val_n > val_n1 else val_n1
        
        if current_max < min_val:
            min_val = current_max
            best_n = n
            
    return best_n

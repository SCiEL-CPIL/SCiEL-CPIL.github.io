"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    N = 1000000
    max_val = N + 1  # we need lpf up to 1000001
    
    # Sieve to compute largest prime factor for each number
    lpf = list(range(max_val + 1))  # lpf[i] = largest prime factor of i
    for i in range(2, int(max_val**0.5) + 1):
        if lpf[i] == i:  # i is prime
            for j in range(i * i, max_val + 1, i):
                lpf[j] = i
    # For primes, lpf[p] = p (already set)
    # For composites, lpf was set to the smallest prime factor during sieve... 
    # Wait, that's wrong. The standard sieve sets the smallest prime factor.
    # I need the LARGEST prime factor.
    
    # Let me redo: use a different approach.
    # For each number, find its largest prime factor.
    # Approach: for each prime p, mark all multiples of p with p as a candidate.
    # The largest such p for each number is the largest prime factor.
    
    lpf = [0] * (max_val + 1)
    lpf[1] = 1
    for i in range(2, max_val + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, max_val + 1, i):
                lpf[j] = i  # since we go in increasing order, the last assignment is the largest prime
    
    # Now scan for best n
    best_n = 10000
    best_val = max(lpf[10000], lpf[10001])
    
    for n in range(10000, N + 1):
        if time.time() > deadline:
            break
        
        # Optimization: if lpf[n] alone is >= best_val, skip
        if lpf[n] >= best_val:
            continue
            
        v2 = lpf[n + 1]
        if v2 >= best_val:
            continue
            
        v = lpf[n]
        if v2 > v:
            v = v2
            
        if v < best_val:
            best_val = v
            best_n = n
            
    return best_n
# EVOLVE-BLOCK-END

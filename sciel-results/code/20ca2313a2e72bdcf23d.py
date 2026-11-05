import time

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin
    
    # 1. Sieve for Largest Prime Factor (LPF)
    # We need lpf for numbers up to 1,000,001
    max_n = 1000000
    max_val = max_n + 1
    
    # Initialize lpf array
    lpf = [0] * (max_val + 1)
    
    # Generate primes and fill lpf
    # We only need to check primes up to sqrt(max_val) to find all prime factors?
    # No, the sieve method below works by marking multiples.
    # If we iterate i from 2 to max_val, and if lpf[i] == 0, it's a prime.
    # Then for all multiples j of i, set lpf[j] = i.
    # Since we go i in increasing order, the last assignment is the largest prime factor.
    
    for i in range(2, max_val + 1):
        if lpf[i] == 0:
            # i is a prime
            for j in range(i, max_val + 1, i):
                lpf[j] = i
                
    # 2. Search for the best n
    best_n = 10000
    best_val = float('inf')
    
    for n in range(10000, max_n + 1):
        if time.time() > deadline:
            break
            
        lpf_n = lpf[n]
        lpf_np1 = lpf[n + 1]
        
        current_val = lpf_n if lpf_n > lpf_np1 else lpf_np1
        
        if current_val < best_val:
            best_val = current_val
            best_n = n
            
    return best_n

# EVOLVE-BLOCK-END

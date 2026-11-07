"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin before 60s limit
    
    n = 256
    result = [1]
    
    # For each prime p <= n, include the largest power of p that is <= n
    # This ensures pairwise coprimality since each prime factor is used by exactly one number
    for p in range(2, n + 1):
        if time.time() > deadline:
            break
        
        # Check if p is prime
        is_prime = True
        if p < 2:
            is_prime = False
        else:
            for d in range(2, int(p**0.5) + 1):
                if p % d == 0:
                    is_prime = False
                    break
        
        if not is_prime:
            continue
        
        # Find largest power of p <= n
        power = p
        while power * p <= n:
            power *= p
        
        result.append(power)
    
    # Ensure all are Python ints and return as list
    return [int(x) for x in result]
# EVOLVE-BLOCK-END

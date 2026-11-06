import time

def construct():
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    num_primes = len(primes)
    
    # Max value for n is 1,000,000. We check n+24, so max index is 1,000,024.
    max_n = 1000000
    max_val = max_n + 24
    min_n = 1000
    
    # masks[i] will hold the bitmask of which primes from 'primes' divide i
    masks = [0] * (max_val + 1)
    
    # Precompute masks
    for i, p in enumerate(primes):
        bit = 1 << i
        for m in range(p, max_val + 1, p):
            masks[m] |= bit
            
    best_score = num_primes + 1
    best_n = min_n
    start_time = time.time()
    
    # Iterate through valid n
    # We can optimize the inner loop by checking if adding more bits exceeds current best
    # But 24 ORs is very fast.
    
    for n in range(min_n, max_n + 1):
        # Check time limit every 10000 steps
        if (n - min_n) % 10000 == 0:
            if time.time() - start_time > 80:
                break
                
        current_or = 0
        # Unroll the loop for speed? Or just a small loop.
        # A small loop of 24 is fine.
        for k in range(1, 25):
            current_or |= masks[n + k]
            
        # Count bits
        score = current_or.bit_count()
        
        if score < best_score:
            best_score = score
            best_n = n
            
    return best_n

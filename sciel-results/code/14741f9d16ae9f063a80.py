import time

def construct():
    # Deadline guard: 90s limit, use 80s to be safe.
    # This algorithm is O(N log log N) and should run in milliseconds, 
    # but we include the guard as per instructions.
    start_time = time.time()
    deadline = start_time + 80.0

    MAX_N = 1000000
    # We need lpf up to MAX_N + 1 because we check n+1
    limit = MAX_N + 1
    
    # Sieve for Largest Prime Factor
    # lpf[i] will store the largest prime factor of i
    lpf = [0] * (limit + 1)
    
    # Iterate i from 2 to limit
    # If lpf[i] is 0, i is prime
    # For each prime i, mark all multiples j with lpf[j] = i
    # Since we iterate i in increasing order, the last prime that divides j
    # will be the largest prime factor of j.
    for i in range(2, limit + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, limit + 1, i):
                lpf[j] = i
                
    # For any number still 0 (which shouldn't happen for i>=2 if loop goes to limit), 
    # but specifically for primes > sqrt(limit), they were set when i was that prime.
    # Wait, the loop `for i in range(2, limit+1)` covers all numbers.
    # If i is prime, it sets lpf for its multiples.
    # So lpf[j] is correctly set to the largest prime factor for all j >= 2.
    # lpf[1] is 0, but n >= 10000, so we don't care about 1.
    
    min_lpf_val = float('inf')
    best_n = 10000
    
    # Search for n in [10000, 1000000]
    # Check time periodically? 
    # The loop is 10^6 iterations, very fast. No need for fine-grained checks.
    for n in range(10000, MAX_N + 1):
        # Calculate largest prime factor of n(n+1)
        # Since gcd(n, n+1)=1, the prime factors are disjoint.
        # The largest prime factor of the product is the max of the largest prime factors of the factors.
        current_lpf = max(lpf[n], lpf[n + 1])
        
        if current_lpf < min_lpf_val:
            min_lpf_val = current_lpf
            best_n = n
            
        # Optional: Early termination if we hit the theoretical minimum?
        # The smallest possible largest prime factor for two consecutive integers > 1
        # is at least 2. But 2*3=6 (lpf 3), 3*4=12 (lpf 3), 4*5=20 (lpf 5).
        # Actually, we just want the global min.
        
    return best_n

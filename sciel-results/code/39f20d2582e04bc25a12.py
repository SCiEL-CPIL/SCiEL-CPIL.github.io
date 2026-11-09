import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin
    
    max_n = 1000000
    min_n = 10000
    
    # Sieve for largest prime factor
    lpf = [0] * (max_n + 2)
    
    for i in range(2, int(max_n**0.5) + 1):
        if lpf[i] == 0: # i is prime
            for j in range(i, max_n + 2, i):
                lpf[j] = i
                
    # Note: The above loop only marks prime factors up to sqrt(max_n).
    # For composite numbers with a prime factor > sqrt(max_n), that factor must be the number itself (if prime) or the number divided by small factors.
    # Wait, the standard sieve for smallest prime factor (spf) works like this:
    # spf = list(range(max_n+2))
    # for i in range(2, int(max_n**0.5)+1):
    #   if spf[i] == i:
    #     for j in range(i*i, max_n+2, i):
    #       if spf[j] == j: spf[j] = i
    # Then lpf can be derived or computed directly.
    
    # Let's do a direct lpf sieve.
    lpf = [0] * (max_n + 2)
    for i in range(2, max_n + 2):
        if lpf[i] == 0:
            for j in range(i, max_n + 2, i):
                lpf[j] = i
                
    # This is O(N log log N) but with a larger constant?
    # Actually, the inner loop runs N/i times. Sum is N log log N.
    # This is fine for 10^6.
    
    best_n = min_n
    best_val = float('inf')
    
    for n in range(min_n, max_n + 1):
        # Check time
        if time.time() > deadline:
            break
            
        v1 = lpf[n]
        v2 = lpf[n+1]
        current_val = max(v1, v2)
        
        if current_val < best_val:
            best_val = current_val
            best_n = n
            
    return best_n

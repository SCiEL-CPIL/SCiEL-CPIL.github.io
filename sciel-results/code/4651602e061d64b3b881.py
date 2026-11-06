import time

def construct():
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Max n is 1000000, so max x is 1000000 + 24 = 1000024
    max_x = 1000024
    
    # Precompute the 23-smooth part for each x in [1, max_x]
    # K[x] is the factor of x composed of primes < 24
    K = [0] * (max_x + 1)
    for x in range(1, max_x + 1):
        k = 1
        temp = x
        for p in primes:
            while temp % p == 0:
                k *= p
                temp //= p
        K[x] = k

    # Sliding window to find n minimizing distinct K values in [n+1, n+24]
    # Window size is 24
    window_size = 24
    
    # Initialize window for n=1000: [1001, 1024]
    freq = {}
    distinct = 0
    min_distinct = 25 # Start high
    best_n = 1000
    
    # Helper to add a value
    def add_val(v):
        nonlocal distinct
        if v not in freq:
            freq[v] = 1
            distinct += 1
        else:
            freq[v] += 1
            
    # Helper to remove a value
    def remove_val(v):
        nonlocal distinct
        freq[v] -= 1
        if freq[v] == 0:
            del freq[v]
            distinct -= 1

    # Init window for n=1000
    for i in range(1001, 1001 + window_size):
        add_val(K[i])
    
    if distinct < min_distinct:
        min_distinct = distinct
        best_n = 1000

    # Slide window
    # n goes from 1001 to 1000000
    # When n increases by 1, we remove K[n+1] and add K[n+25]
    # For n=1001, remove K[1001], add K[1025]
    
    start_time = time.time()
    deadline = 55.0 # Safety margin
    
    for n in range(1001, 1000001):
        # Time check
        if time.time() - start_time > deadline:
            break
        
        # Remove outgoing element: n_prev + 1 = (n-1) + 1 = n
        # Wait, previous window was for n-1: [(n-1)+1, (n-1)+24] = [n, n+23]
        # Current window for n: [n+1, n+24]
        # So we remove K[n] and add K[n+24]
        
        out_val = K[n]
        in_val = K[n + 24]
        
        remove_val(out_val)
        add_val(in_val)
        
        if distinct < min_distinct:
            min_distinct = distinct
            best_n = n

    return best_n

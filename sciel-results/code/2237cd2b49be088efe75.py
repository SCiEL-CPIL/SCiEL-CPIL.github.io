import time

def construct():
    start_time = time.time()
    deadline = 50  # Leave 10s margin
    
    # Primes strictly below 24
    S = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    MAX_N = 1000000
    WINDOW_SIZE = 24
    MAX_VAL = MAX_N + WINDOW_SIZE
    
    # Precompute smooth parts
    # A[i] is the 23-smooth part of i
    A = [1] * (MAX_VAL + 1)
    
    # Optimization: We can compute this slightly faster
    # But simple loop is likely fine.
    for i in range(1, MAX_VAL + 1):
        x = i
        val = 1
        for p in S:
            while x % p == 0:
                val *= p
                x //= p
            if x == 1:
                break
        A[i] = val
        
    # Check time
    if time.time() - start_time > deadline:
        return 1000

    # Sliding window to find n minimizing distinct smooth parts in [n+1, n+24]
    # n ranges from 1000 to MAX_N
    # Window for n is A[n+1] ... A[n+24]
    
    best_n = 1000
    best_distinct = float('inf')
    
    # Initialize window for n=1000
    # Indices: 1001 to 1024
    counts = {}
    distinct = 0
    
    for i in range(1001, 1001 + WINDOW_SIZE):
        v = A[i]
        if v not in counts:
            counts[v] = 1
            distinct += 1
        else:
            counts[v] += 1
            
    best_distinct = distinct
    best_n = 1000
    
    # Slide window
    # n goes from 1001 to MAX_N
    # When n increments by 1:
    # Remove A[n] (which was the first element of previous window, index n)
    # Add A[n + WINDOW_SIZE] (new last element, index n + 24)
    
    for n in range(1001, MAX_N + 1):
        # Remove A[n]
        v_out = A[n]
        counts[v_out] -= 1
        if counts[v_out] == 0:
            distinct -= 1
            # del counts[v_out] # optional, keeps dict small
        
        # Add A[n + WINDOW_SIZE]
        v_in = A[n + WINDOW_SIZE]
        if v_in not in counts:
            counts[v_in] = 1
            distinct += 1
        else:
            counts[v_in] += 1
            
        if distinct < best_distinct:
            best_distinct = distinct
            best_n = n
            
        if time.time() - start_time > deadline:
            break
            
    return best_n

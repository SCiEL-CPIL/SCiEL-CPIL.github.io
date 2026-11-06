import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # We want to maximize the number of pairs.
    # Number of powerful numbers K ~ 2.17 * sqrt(L)
    # Cost of counting is O(K^2)
    # For L = 10^7, K ~ 6800, K^2 ~ 4.6 * 10^7. 
    # Python loop ~ 1-2 seconds.
    # For L = 5 * 10^7, K ~ 15000, K^2 ~ 2.2 * 10^8.
    # Python loop ~ 5-10 seconds.
    # Let's aim for L = 2 * 10^7 to be safe and fast.
    
    L = 2 * 10**7
    
    # 1. Generate powerful numbers <= L
    powerful_set = set()
    max_y = int(L ** (1/3)) + 1
    
    # Precompute square-free numbers up to max_y
    # y is square-free if no p^2 divides y
    # Simple sieve-like check or trial division
    # Since max_y is small (~270 for 2*10^7), trial division is fine.
    
    sq_free = []
    for y in range(1, max_y + 1):
        is_sq_free = True
        temp_y = y
        d = 2
        while d * d <= temp_y:
            if temp_y % (d * d) == 0:
                is_sq_free = False
                break
            d += 1
        if is_sq_free:
            sq_free.append(y)
            
    for y in sq_free:
        y_cubed = y ** 3
        if y_cubed > L:
            continue
        max_x = int(math.isqrt(L // y_cubed))
        for x in range(1, max_x + 1):
            val = x * x * y_cubed
            powerful_set.add(val)
            
    P = sorted(powerful_set)
    K = len(P)
    
    # Check time
    if time.time() > deadline - 5:
        # Fallback: return a small result if generation was unexpectedly slow
        # But generation is O(L^(1/3) * L^(1/6)) which is small.
        pass

    # 2. Count sums
    # Using dictionary for sparse sums
    sum_counts = {}
    
    P_list = P
    K_val = K
    
    # Optimization: local variables
    for i in range(K_val):
        pi = P_list[i]
        # j starts from i
        # We don't need to break on sum > limit because we are just counting, 
        # and we want to find the global max N, which could be up to 2L.
        # However, if we are interested in N <= some limit, we could break.
        # But the task doesn't limit N, only N <= 10^18.
        # Our N will be <= 2L = 4*10^7.
        for j in range(i, K_val):
            s = pi + P_list[j]
            sum_counts[s] = sum_counts.get(s, 0) + 1
            
            # Time check inside loop is expensive. 
            # K=15000 -> 1.1*10^8 ops. 
            # If this takes 10s, we are fine.
            
    # 3. Find max
    max_count = 0
    best_N = 0
    for s, c in sum_counts.items():
        if c > max_count:
            max_count = c
            best_N = s
            
    # 4. Retrieve pairs
    pairs = []
    powerful_set_lookup = powerful_set
    for a in P_list:
        b = best_N - a
        if b < a:
            continue
        if b in powerful_set_lookup:
            pairs.append([a, b])
            
    return {'N': best_N, 'pairs': pairs}

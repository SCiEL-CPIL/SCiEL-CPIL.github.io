import time
import sys

def construct():
    start_time = time.time()
    limit = 1000000
    max_n = limit
    min_n = 1000
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Precompute small_part for range 1001 to 1000024
    # We need sp(x) for x in [1001, 1000024]
    # sp(x) = product of p^k for p in primes such that p^k divides x
    
    # Efficient calculation:
    # Initialize sp array with 1s
    size = max_n + 25
    sp = [1] * size
    
    # For each prime p, multiply sp[j] by p as long as p divides j
    # This is O(N log N) roughly
    for p in primes:
        for j in range(p, size, p):
            # Count how many times p divides j
            temp = j
            while temp % p == 0:
                sp[j] *= p
                temp //= p
    
    best_n = min_n
    best_val = float('inf')
    
    # Search
    # To speed up, we can check every 10th n first?
    # Or just linear scan.
    # 10^6 iterations.
    # Each iteration: 24 lookups, set creation.
    
    # Optimization: Use a fixed size set or just count distincts.
    # Since values can be large, we can't use a simple boolean array.
    # But we can store the set.
    
    # Let's run a linear scan until time limit
    deadline = start_time + 80  # Leave 10s margin
    
    for n in range(min_n, max_n + 1):
        if time.time() > deadline:
            break
        
        # Extract sp(n+1) ... sp(n+24)
        # Using a set
        s = set()
        for i in range(1, 25):
            s.add(sp[n + i])
        
        val = len(s)
        if val < best_val:
            best_val = val
            best_n = n
            
        # Early termination if we find the theoretical minimum?
        # Theoretical minimum is likely > 1.
        # If best_val == 2, it's very good.
        # If best_val == 1, it's impossible (as 2 must appear).
        
    return best_n

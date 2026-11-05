import time
from math import isqrt

def construct():
    start_time = time.time()
    timeout = 100  # seconds, leave margin for 120s limit
    
    # We will try to find the best N for a given limit M on powerful numbers.
    # The number of powerful numbers up to M is approx 0.65 * M^(2/3).
    # We want to maximize the count of pairs a+b=N.
    # The cost is O(K^2) where K is the number of powerful numbers.
    # K^2 should be < 10^8 for reasonable speed in Python (approx 1-5s).
    # So K < 10000.
    # M^(2/3) < 10000/0.65 => M < (15000)^1.5 => M < ~1.8 * 10^6.
    # Let's try M = 2 * 10^6.
    
    M = 2_000_000
    
    # Generate powerful numbers up to M
    powerful_set = set()
    # v^3 <= M => v <= M^(1/3)
    max_v = int(M ** (1/3)) + 1
    
    for v in range(1, max_v + 1):
        # Check if v is square-free? 
        # Actually, the form u^2 * v^3 covers all powerful numbers uniquely 
        # if v is square-free. 
        # If v is not square-free, say v = w^2 * k, then u^2 w^6 k^3 = (u w^3)^2 k^3.
        # So we can just iterate all v and u, and use a set to handle duplicates.
        # Or enforce v square-free to be efficient.
        
        # Let's just iterate all v, it's simpler and M is small.
        v_cubed = v ** 3
        if v_cubed > M:
            break
        max_u = isqrt(M // v_cubed)
        for u in range(1, max_u + 1):
            p = u * u * v_cubed
            powerful_set.add(p)
            
    powerful_list = sorted(powerful_set)
    K = len(powerful_list)
    
    # Count sums
    # We want to find N with max count.
    # N = a + b with a <= b.
    # We can use a dictionary.
    counts = {}
    
    # Optimization: 
    # If K is large, this loop is slow.
    # For M=2e6, K ~ 10000. K^2/2 = 50M.
    # This might take 10-20s.
    
    # Let's add a check for time.
    # If we are running out of time, we might need to reduce M or stop.
    # But we haven't started the heavy loop yet.
    
    # We can try to be smarter.
    # Instead of counting all, can we just sample? No, we need exact count for the winner.
    
    # Let's run the loop.
    for i in range(K):
        a = powerful_list[i]
        # b must be >= a
        # a + b <= 2*M
        # b <= 2*M - a
        # We can binary search the end index?
        # Or just break if a + powerful_list[j] > 2*M.
        
        # Inner loop
        # To speed up, we can use a local variable for the list and counts
        pl = powerful_list
        for j in range(i, K):
            b = pl[j]
            s = a + b
            if s > 2 * M:
                break
            if s in counts:
                counts[s] += 1
            else:
                counts[s] = 1
                
    # Find max
    best_N = 0
    best_count = 0
    for n, c in counts.items():
        if c > best_count:
            best_count = c
            best_N = n
            
    # Construct pairs
    pairs = []
    # We need to list the pairs for best_N
    # This is another O(K) scan
    for a in powerful_list:
        b = best_N - a
        if b < a:
            continue
        if b in powerful_set:
            pairs.append([a, b])
            
    # Ensure we return valid format
    return {'N': best_N, 'pairs': pairs}

import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    # We want to find M with many k such that k*sigma(k) = M.
    # Strategy: Sieve sigma(k) for k up to N, then bucket by M = k*sigma(k).
    
    # Start with a reasonable N. 10^6 is safe and fast.
    N = 10**6
    
    # Sieve for sigma
    # sigma[k] = sum of divisors of k
    sigma = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            sigma[j] += i
            
    # Bucket k by M = k * sigma[k]
    M_map = defaultdict(list)
    for k in range(1, N + 1):
        M = k * sigma[k]
        if M <= 10**18:
            M_map[M].append(k)
            
    # Find the M with the most k's
    best_M = 0
    best_ks = []
    for M, ks in M_map.items():
        if len(ks) > len(best_ks):
            best_M = M
            best_ks = ks
            
    # If we have time, try a larger N?
    # 10^6 might only give 3 or 4.
    # Let's check if we can go higher.
    # Sieve up to 10^7 might be too slow in pure Python loops for the inner loop.
    # The inner loop `for j in range(i, N+1, i): sigma[j] += i` is O(N log N).
    # For N=10^6, sum(1/i) ~ log(10^6) ~ 14. So ~14 * 10^6 operations. Very fast.
    # For N=10^7, ~16 * 10^7 operations. Might take 5-10s in Python.
    # Let's try to extend to 5*10^6 if time permits.
    
    if time.time() < deadline - 20:
        N2 = 5 * 10**6
        # We can extend the sieve or just run a new one.
        # Running a new one is simpler but duplicates work.
        # Let's just try a second pass with a different range or larger N.
        # Actually, let's just stick to N=10^6 for now to ensure validity, 
        # but 10^6 might only find the seed's 3 solutions or slightly more.
        
        # Let's try to be smarter. 
        # The seed found 336, 372, 434. These are small.
        # There might be larger clusters.
        
        # Let's try N=2*10^6 to be safe and fast.
        if N < 2 * 10**6:
             N = 2 * 10**6
             sigma = [0] * (N + 1)
             for i in range(1, N + 1):
                 for j in range(i, N + 1, i):
                     sigma[j] += i
                     
             M_map = defaultdict(list)
             for k in range(1, N + 1):
                 M = k * sigma[k]
                 if M <= 10**18:
                     M_map[M].append(k)
                     
             for M, ks in M_map.items():
                 if len(ks) > len(best_ks):
                     best_M = M
                     best_ks = ks

    # Ensure types are correct
    # M must be int
    # k must be list of ints
    
    # Fallback if nothing found (should not happen)
    if not best_ks:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    return {'M': int(best_M), 'k': [int(k) for k in best_ks]}

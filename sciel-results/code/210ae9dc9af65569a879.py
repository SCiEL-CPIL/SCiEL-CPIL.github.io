import sys

# Increase recursion limit just in case, though depth is 12
sys.setrecursionlimit(2000)

def construct():
    n = 12
    lcm_period = 27720  # LCM of 1..12
    
    best_m = 0
    best_L = 0
    
    # Precompute divisors logic is simple, but we can precompute nothing global 
    # that depends on m and L easily without loops.
    
    for m in range(0, lcm_period):
        # We want to find the minimum L for this m.
        # If we find a match for L, that is f(12, m).
        # We want to maximize this f(12, m).
        
        found_match = False
        for L in range(1, 145):
            # Precompute available indices for each divisor i+1
            # avail[i] will store indices j in 0..L-1 such that (m+1+j) % (i+1) == 0
            avail = [[] for _ in range(n)]
            
            # Optimization: Only check j if needed? 
            # Looping L up to 144 and n=12 is 1728 checks per L.
            # 144 * 1728 ~ 250k ops per m.
            # 27720 m's * 250k ~ 7 billion ops. Too slow in Python.
            
            # We need a faster way to determine if a match exists.
            # Or a faster way to compute avail.
            
            # Let's stick to the current structure but optimize the inner loop.
            # The current code does the modulo check inside DFS.
            # Precomputing avail lists is better.
            
            # However, 7 billion operations is definitely too slow for 60s.
            # We need a different strategy.
            
            # Strategy: 
            # 1. Note that the condition depends on m mod (i+1).
            # 2. For a fixed L, the set of valid j for each i is determined by m mod (i+1).
            # 3. We can group m's by their residues modulo 27720? No, m IS the residue.
            
            # Is there a way to jump L?
            # No, we need the MIN L.
            
            # Can we parallelize? No.
            
            # Can we use NumPy?
            # For a fixed m and L, we can create a matrix of booleans.
            # But creating matrices for 27720 * 144 cases is heavy.
            
            # Let's look at the constraints. 60 seconds.
            # 27720 * 144 = ~4 million (m, L) pairs.
            # For each pair, we need to check if a matching exists.
            # If the matching check is O(1) or very small, we are fine.
            
            # A simple DFS for N=12 is very fast if the graph is dense.
            # The graph is defined by divisibility.
            # Average degree: For i=1, all L are valid. For i=12, L/12 are valid.
            # The bottleneck is the precomputation of `avail` or the DFS itself.
            
            # Let's try to optimize the Python loop.
            # Instead of precomputing `avail` lists (which involves appending), 
            # we can just pass m and L to a C-optimized function? No.
            
            # Let's try to reduce the number of (m, L) checks.
            # We are looking for MAX f(m).
            # f(m) is the min L.
            # If for a given m, we fail to find a match for L=1, we try L=2, etc.
            # Most m's will have small f(m).
            # Only "bad" m's will have large f(m).
            # We can stop early if current best_L is high?
            # No, we don't know if a future m will be higher.
            
            # However, we can prune L search?
            # If we know that for this m, a match is impossible for L < K, we skip.
            # How to know K?
            
            # Let's just implement the precomputation efficiently.
            # Using list comprehensions might be faster than manual loops.
            
            pass # Placeholder for logic below

        # The previous thought process identified the bottleneck.
        # Let's rewrite the core loop with precomputation and efficient DFS.
        
        pass
    return best_m

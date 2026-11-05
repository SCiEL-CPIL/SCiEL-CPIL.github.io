import time
import random
from math import gcd
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0  # Leave 5s margin for 90s limit
    
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute GCD table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
            
    best_set = set()
    best_size = 0
    
    # Heuristic: Pre-compute potential conflicts? 
    # A triple (a,b,c) is bad if gcd(a,b)==gcd(a,c)==gcd(b,c).
    
    def is_safe_addition(candidate, current_set):
        # Check if adding candidate to current_set creates a bad triple
        # The bad triple must consist of candidate and two elements from current_set
        # Let the two elements be u, v from current_set.
        # We need gcd(candidate, u) == gcd(candidate, v) == gcd(u, v)
        
        # Optimization: Group current_set elements by their GCD with candidate?
        # Let g_c_u = gcd(candidate, u).
        # If g_c_u == g_c_v, then we must check if gcd(u, v) == g_c_u.
        
        # Bucket elements in current_set by their GCD with candidate
        buckets = {}
        for u in current_set:
            g = gcd_table[candidate][u]
            if g not in buckets:
                buckets[g] = []
            buckets[g].append(u)
            
        # For each bucket with >1 elements, check if any pair within the bucket has GCD equal to the bucket key
        for g, members in buckets.items():
            if len(members) < 2:
                continue
            # Check all pairs in members
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    u = members[i]
                    v = members[j]
                    if gcd_table[u][v] == g:
                        return False
        return True

    # Randomized Greedy with Restarts
    # We can also try different orderings.
    
    for _ in range(1000): # Max restarts, will be limited by time
        if time.time() > deadline:
            break
            
        random.shuffle(elements)
        current_set = set()
        
        for x in elements:
            if time.time() > deadline:
                break
            if is_safe_addition(x, current_set):
                current_set.add(x)
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
            # If we found a very large set, we might stop early? 
            # But let's try to maximize.
            
    # Local Search / Refinement
    # Try to add any remaining elements if possible (in case we missed some due to order)
    # Or try to swap.
    
    # Simple refinement: try to add any element from [1..128] not in best_set
    # This is already done by the greedy, but the order matters.
    # Let's do a few passes of "try to add anything that fits" on the best_set
    
    for _ in range(10):
        if time.time() > deadline:
            break
        added = False
        for x in range(1, N + 1):
            if x not in best_set:
                if is_safe_addition(x, best_set):
                    best_set.add(x)
                    added = True
        if not added:
            break

    return list(best_set)
# EVOLVE-BLOCK-END

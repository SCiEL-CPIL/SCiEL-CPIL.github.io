import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Try n=21 first
    for n in [21, 20, 19]:
        if time.time() > deadline - 5:
            break
            
        # Try to find a valid graph with this n
        # We will try random graphs
        # For n=21, total edges possible = 210
        # Random regular graphs or p=0.5
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
                
            # Generate random graph
            edges = []
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        edges.append([i, j])
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Check validity
            if is_valid(adj, n):
                return {'n': n, 'edges': edges}
        
        # If no valid graph found for this n within attempts, try smaller n
    # Fallback: return a known small valid graph if time runs out
    # The empty graph on 5 vertices is valid.
    return {'n': 5, 'edges': []}

def is_valid(adj, n):
    """
    Returns True if no induced subgraph of size >= 6 is regular.
    adj is a list of bitmasks.
    """
    # Iterate over all subsets of size k >= 6
    # We can iterate over all masks and check size
    # To be faster, iterate by k
    
    # Precompute bit counts? 
    # Python's int.bit_count() is fast.
    
    # Iterate k from 6 to n
    for k in range(6, n + 1):
        # Generate all combinations of size k?
        # itertools.combinations is okay for small k, but we need to do it many times.
        # It might be faster to iterate all masks and filter by bit_count.
        # 2^21 is 2 million masks. 
        # Doing this for each random graph attempt might be slow if we do many attempts.
        # Let's optimize: 
        # If we find a regular subset, return False immediately.
        
        # Strategy: Iterate all masks. 
        # But we can skip masks with bit_count < 6.
        
        # Let's try iterating combinations for k=6 first.
        # If no regular k=6, try k=7, etc.
        
        # For k=6, C(21,6) = 54k. Fast.
        # For k=7, C(21,7) = 116k.
        # ...
        # Total is 2^21.
        
        # Let's just iterate all masks from 0 to 2^n - 1?
        # No, that's 2 million iterations. In Python, 2 million iterations of bit_count and degree check might take ~1-2 seconds.
        # With 50 attempts, that's 50-100 seconds. Too slow.
        
        # We need a faster check or fewer attempts.
        # Maybe we can check only k=6 and k=7? 
        # Is it possible to have no regular k=6 or k=7, but have a regular k=8?
        # Yes. But it's less likely? 
        # Actually, if a graph has no regular induced subgraph of size 6, it's a strong condition.
        # Let's try checking k=6 first. If valid for k=6, then check k=7, etc.
        # We can break early if we find a violation.
        
        # Let's use itertools.combinations for each k.
        
        from itertools import combinations
        
        for k in range(6, n + 1):
            for combo in combinations(range(n), k):
                # Check if induced subgraph on combo is regular
                # Degrees:
                # deg[v] = popcount(adj[v] & mask)
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                
                # Calculate degrees
                # Since k is small, we can just loop
                degrees = []
                for v in combo:
                    deg = (adj[v] & mask).bit_count()
                    degrees.append(deg)
                
                # Check if all degrees are equal
                if len(set(degrees)) == 1:
                    return False
            
            # If we finish a k without finding a regular subgraph, continue to next k
            # Optimization: If time is running out, we might stop checking larger k?
            # No, we must be sure. But if we have time, we check all.
            # If we run out of time during checking, we should assume it's invalid?
            # Better to have a time check inside the loop.
            if time.time() > time.time() + 100: # Placeholder
                 pass
                
    return True

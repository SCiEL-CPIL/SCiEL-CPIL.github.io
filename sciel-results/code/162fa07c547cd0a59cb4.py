import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    time_limit = 110.0 # Safety margin
    
    # 1. Calculate Moore Bound
    if g % 2 == 0:
        # Bipartite
        n_min = 2
        term = 1
        for i in range(g//2 - 1):
            n_min += 2 * term
            term *= (k-1)
    else:
        # Non-bipartite
        n_min = 1
        term = 1
        for i in range((g-3)//2 + 1): # i=0 to (g-3)/2
             n_min += k * term
             term *= (k-1)
             
    # Adjust n_min to satisfy parity if k is odd?
    # Sum of degrees = n*k must be even.
    if (n_min * k) % 2 != 0:
        n_min += 1
        
    best_n = None
    best_edges = None
    
    # Search range
    # For small k,g, the cage is very close to Moore bound.
    # For larger k,g, random search might fail to find it quickly.
    # We will iterate n upwards.
    
    max_search_n = n_min + 1000 # Cap the search
    
    for n in range(n_min, max_search_n + 1):
        if time.time() - start_time > time_limit:
            break
            
        # Check if n*k is even
        if (n * k) % 2 != 0:
            continue
            
        # Try to find a graph of size n
        # Number of attempts depends on n and time
        attempts = 5000
        if n > 100: attempts = 1000
        if n > 500: attempts = 100
        
        found = False
        for _ in range(attempts):
            if time.time() - start_time > time_limit:
                break
                
            edges = generate_random_regular(k, n)
            if edges is None:
                continue
                
            if check_girth(edges, n, k, g):
                best_n = n
                best_edges = edges
                found = True
                break
                
        if found:
            break
            
    if best_n is None:
        # Fallback to LUW if random search failed
        # Re-implement LUW briefly or return a trivial valid graph if possible?
        # The task requires valid graph. LUW is guaranteed.
        return luw_construct(k, g)
        
    return best_n, best_edges

import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Try to find a smaller graph
    # Moore bound for k=4, g=10 is ~243.
    # Try N from 244 to 320 step 2 (must be even? No, 4-regular implies sum degrees = 4N, always even. N can be odd? No, 4N is even, so N can be odd. But simple graph constraints... usually even N is easier for random gen? No, 4-regular graphs exist for any N >= 5.
    # However, bipartite requires even N. Non-bipartite can be odd.
    # Let's try even N first as they are more common in cages.
    
    best_n = None
    best_edges = None
    
    for n in range(244, 321, 2): # Try up to 320
        if time.time() > deadline - 10: break
        
        # Try to construct a valid graph of size n
        result = search_for_cage(n, k, g, time_limit=5.0)
        if result is not None:
            best_n, best_edges = result
            # Since we iterate n increasing, the first valid one is the smallest.
            # We can return immediately?
            # Yes, we want minimal n.
            return best_n, best_edges
        
    # If no valid graph found in range, fallback to seed
    return construct_fallback(k, g)

def search_for_cage(n, k, g, time_limit):
    # Generate random k-regular graph
    # Use configuration model
    # ...
    # Local search
    # ...
    # Check girth
    # ...
    pass

def construct_fallback(k, g):
    # Seed code
    pass

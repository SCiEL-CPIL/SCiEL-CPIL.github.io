import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    # Safety margin: 110s out of 120s
    deadline = start_time + 110
    
    # If the specific case 4,10, try to improve
    if k == 4 and g == 10:
        # Try to find a graph with n=320
        # If we find one, it's likely optimal or close
        result = try_local_search(320, 4, 10, deadline)
        if result:
            return result
            
    # Fallback to the seed algorithm (Lazebnik-Ustimenko-Woldar)
    return seed_construct(k, g)

def seed_construct(k, g):
    # ... (copy seed code) ...
    pass

def try_local_search(n, k, g, deadline):
    # Initialize random k-regular graph
    # ...
    pass

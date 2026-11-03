# EVOLVE-BLOCK-START
import numpy as np
from collections import deque
import time
import random

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Tries a greedy search first, falls back to LUW construction.
    """
    if k == 2:
        n = max(g, 3)
        edges = [(i, (i + 1) % n) for i in range(n)]
        return n, edges
    
    if k == 1:
        return 2, [(0, 1)]
        
    # Try greedy search
    try:
        start_time = time.time()
        n_search, edges_search = construct_greedy(k, g, time_limit=110)
        if n_search is not None:
            return n_search, edges_search
    except Exception:
        pass

    # Fallback to LUW
    return construct_luw(k, g)

def construct_luw(k, g):
    # ... (copy of existing LUW code) ...
    pass

def construct_greedy(k, g, time_limit=110):
    # ... (new search code) ...
    return None, None
# EVOLVE-BLOCK-END

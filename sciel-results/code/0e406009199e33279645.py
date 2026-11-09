import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Try n=18
    result_18 = search_coloring(18, deadline - time.time())
    if result_18 is not None:
        return {'n': 18, 'colors': result_18}
        
    # Try n=17
    remaining = deadline - time.time()
    if remaining < 10:
        # Not enough time, return seed or best small
        return {'n': 9, 'colors': [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]}
        
    result_17 = search_coloring(17, remaining)
    if result_17 is not None:
        return {'n': 17, 'colors': result_17}
        
    # Fallback
    return {'n': 9, 'colors': [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]}

def search_coloring(n, time_limit):
    # Precompute cycles
    # ...
    # Search loop
    # ...
    pass

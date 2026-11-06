import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Try from largest to smallest
    for n in [26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11]:
        if time.time() > deadline:
            break
            
        # Allocate time based on n
        # Give more time to larger n
        time_limit = 10.0 if n > 20 else 5.0
        # But for 26, we should give it the bulk of the time
        if n == 26:
             time_limit = deadline - time.time() - 5.0 # Leave 5s for fallbacks
             if time_limit < 10: time_limit = 10
             
        result = find_coloring(n, time_limit)
        if result is not None:
            return result
            
    # Fallback: Return a known valid small n
    # n=11 is always valid with simple coloring
    return {'n': 11, 'colors': [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]}

def find_coloring(n, time_limit):
    # ... SA implementation ...
    pass

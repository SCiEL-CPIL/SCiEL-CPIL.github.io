import time

_start_time = None
_DEADLINE = 330 * 0.95

def priority(vector, n):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    if time.time() - _start_time > _DEADLINE:
        return 0.0

    sum_sq = 0
    sum_coords = 0
    for v in vector:
        sum_sq += v * v
        sum_coords += v
        
    # Primary: Maximize sum of squares (favors 2s > 1s > 0s)
    # Secondary: Maximize sum of coordinates
    return sum_sq + sum_coords * 0.001

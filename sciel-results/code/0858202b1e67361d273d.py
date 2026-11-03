import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = 2, 3, 5
    deadline = time.time() + 140  # 10s margin
    
    # Try ranks from 27 down to some minimum
    for rank in range(27, 14, -1):
        if time.time() > deadline - 10:
            break
        result = _try_rank(n, m, p, rank, deadline)
        if result is not None:
            return result
    
    # Fallback to rank 28
    return _rank_28()

def _rank_28():
    # ... existing code ...
    pass

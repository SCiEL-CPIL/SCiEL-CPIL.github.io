import numpy as np
import time

def construct():
    start_time = time.time()
    n, m, p = 2, 3, 6
    
    # Pre-computed rank 33 solution
    # ... (copy from current program) ...
    
    # Try to find lower rank
    for rank in [32, 31, 30, 29, 28]:
        if time.time() - start_time > 120:
            break
        success = _try_als_decomposition(n, m, p, rank, max_iters=50, restarts=5)
        if success:
            # Retrieve the solution
            return _get_solution()
            
    return _build_rank33()

def _try_als_decomposition(n, m, p, rank, max_iters, restarts):
    # Implement ALS
    pass

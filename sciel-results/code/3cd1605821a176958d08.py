<<<<<<< SEARCH
import time
import numpy as np
from itertools import product
from scipy.optimize import minimize, differential_evolution

DIMENSIONS = [3, 3, 4]
DEADLINE = 115.0  # seconds, with safety margin before 120s

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    # Schoolbook rank is 36
    schoolbook_rank = n * m * p
    
    # Try to find a better decomposition
    best_rank = schoolbook_rank
    best_U = None
    best_V = None
    best_W = None
    
    # We'll try ranks from 25 to 35 (36 is schoolbook)
    # The minimum possible rank for 3x3x4 is at least 27 (lower bound)
    # Let's try a few candidate ranks
    
    for rank in range(27, 36):
        if time.time() - start_time > DEADLINE - 30:
            break
            
        # Try to find a decomposition of this rank
        result = _try_find_decomposition(n, m, p, rank, time.time() - start_time)
        if result is not None:
            U, V, W = result
            # Verify
            if _verify_decomposition(U, V, W, n, m, p):
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W
                break  # Found a valid decomposition, stop searching lower ranks
    
    if best_U is None:
        # Return schoolbook
        return _schoolbook_decomposition(n, m, p)
    else:
        return {"U": best_U, "V": best_V, "W": best_W}
=======
import time
import numpy as np
from itertools import product
from scipy.optimize import minimize, differential_evolution

DIMENSIONS = [3, 3, 4]
DEADLINE = 115.0  # seconds, with safety margin before 120s

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    # Schoolbook rank is 36
    schoolbook_rank = n * m * p
    
    # Try to find a better decomposition
    best_rank = schoolbook_rank
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from 27 to 35 (36 is schoolbook)
    # The minimum possible rank for 3x3x4 is at least 27 (lower bound)
    
    for rank in range(27, 36):
        if time.time() - start_time > DEADLINE - 30:
            break
            
        # Try to find a decomposition of this rank with multiple restarts
        result = _try_find_decomposition(n, m, p, rank, time.time() - start_time)
        if result is not None:
            U, V, W = result
            # Verify
            if _verify_decomposition(U, V, W, n, m, p):
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W
                break  # Found a valid decomposition, stop searching lower ranks
    
    if best_U is None:
        # Return schoolbook
        return _schoolbook_decomposition(n, m, p)
    else:
        return {"U": best_U, "V": best_V, "W": best_W}
>>>>>>> REPLACE


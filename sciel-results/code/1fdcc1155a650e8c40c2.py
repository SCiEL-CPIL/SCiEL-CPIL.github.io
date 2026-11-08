import time
import numpy as np

_DIMS = (3, 4, 7)

def construct():
    _t0 = time.time()
    n, m, p = _DIMS
    
    # 1. Find max 2x2x2 Strassen blocks
    # ... (Bitmask MIS) ...
    
    # 2. Identify uncovered
    # 3. Greedily pack 2x2x1, 2x1x2, 1x2x2 blocks in uncovered
    # 4. Add schoolbook for rest
    
    # Construct matrices
    # Verify
    # Return

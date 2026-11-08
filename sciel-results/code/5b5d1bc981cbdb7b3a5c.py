import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s margin
    
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # ... (rest of the code) ...
    
    if time.time() > deadline:
        # Return best found so far (which is the full construction if it finished)
        pass
        
    return {"U": U, "V": V, "W": W}

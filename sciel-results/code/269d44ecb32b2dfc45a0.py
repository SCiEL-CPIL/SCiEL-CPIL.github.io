import numpy as np
import time

def construct():
    n, m, p = 3, 4, 4
    rank_target = 43
    
    # ... (generate valid/invalid terms)
    
    start_time = time.time()
    deadline = 110
    
    # Use the rank 44 solution as initialization for rank 43
    U44, V44, W44 = _build_rank_44(n, m, p)
    
    # Initialize rank 43 by taking first 43 columns of rank 44
    U = U44[:, :43].copy()
    V = V44[:, :43].copy()
    W = W44[:, :43].copy()
    
    # Add noise
    U += np.random.randn(*U.shape) * 0.5
    V += np.random.randn(*V.shape) * 0.5
    W += np.random.randn(*W.shape) * 0.5
    
    # Gradient Descent
    # ...
    
    # If fails, return rank 44

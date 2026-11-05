def _find_cw_3x3x3(time_limit=5):
    """Try to find a rank-23 decomposition of the 3×3×3 tensor."""
    import numpy as np
    from scipy.optimize import minimize
    
    start = time.time()
    
    # Target: all-ones tensor of size 3×3×3
    T_target = np.ones((3, 3, 3))
    
    rank = 23
    # Initialize with schoolbook + random
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    
    # Schoolbook: 9 terms
    for i in range(3):
        for j in range(3):
            U[i*3+j, i] = 1.0  # Wait, this is wrong

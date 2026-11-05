import numpy as np

def construct():
    n, m, p = 3, 3, 5
    
    # The tensor for matrix multiplication C = AB is the all-ones tensor.
    # C_ik = sum_j A_ij B_jk.
    # The coefficient of A_ij B_jk in C_ik is 1 for all valid i, j, k.
    # Thus, T_ijk = 1.
    # The rank of the all-ones tensor is 1, achieved by U=1, V=1, W=1.
    
    rank = 1
    
    U = np.ones((n * m, rank))
    V = np.ones((m * p, rank))
    W = np.ones((p * n, rank))
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

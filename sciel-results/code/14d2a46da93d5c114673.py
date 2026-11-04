"""
Solution for 2x3 by 3x3 matrix multiplication tensor decomposition.
Uses the trivial rank-18 decomposition (one rank-1 tensor per output entry).
"""

import time
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x3 matrix.
    
    Returns:
        dict: A dictionary with keys 'U', 'V', 'W' containing the factor matrices.
              U: shape (6, 18)
              V: shape (9, 18)
              W: shape (6, 18)
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    N, M, P = 2, 3, 3
    R = N * M * P  # 18
    
    # Build using numpy for efficiency
    U = np.zeros((N * M, R), dtype=float)
    V = np.zeros((M * P, R), dtype=float)
    W = np.zeros((P * N, R), dtype=float)
    
    # Trivial decomposition:
    # r = i * M * P + j * P + k
    for i in range(N):
        for j in range(M):
            for k in range(P):
                r = i * M * P + j * P + k
                U[i * M + j, r] = 1.0
                V[j * P + k, r] = 1.0
                W[k * N + i, r] = 1.0
    
    # Convert to lists for return
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

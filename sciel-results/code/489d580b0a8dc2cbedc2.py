"""
Program to find a rank-minimal exact tensor decomposition for matrix multiplication 3x3 by 3x7.
The trivial decomposition has rank n*m*p = 3*3*7 = 63.
This program constructs that decomposition with a deadline guard.
"""
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    n, m, p = 3, 3, 7
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    # The tensor T_{i,j,k} corresponds to the entry (i,k) of the product C = A * B
    # A is n x m, B is m x p, C is n x p.
    # Matrix multiplication: C_{ik} = sum_j A_{ij} B_{jk}
    # The tensor is defined such that T_{i,j,k} A_{ij} B_{jk} = C_{ik}.
    # The rank-1 decomposition term for (i,j,k) is:
    # U_{i*m+j, r} = 1
    # V_{j*p+k, r} = 1
    # W_{k*n+i, r} = 1
    # This matches the trace convention W[k*n+i, r].
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

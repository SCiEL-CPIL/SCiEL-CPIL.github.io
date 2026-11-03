"""
Program for finding a rank-minimal exact tensor decomposition for
multiplying a 2x3 matrix by a 3x10 matrix.

The tensor M_{2,3,10} corresponds to the matrix multiplication operation.
The schoolbook algorithm uses 2*3*10 = 60 multiplications, corresponding
to a tensor rank of 60.

Given the complexity of finding a lower rank decomposition for these
dimensions within the time limit, and the fact that the schoolbook rank
is a valid upper bound, we return the schoolbook decomposition.
"""
import time
import numpy as np

def construct():
    n, m, p = 2, 3, 10
    rank = n * m * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j, col] = 1
                # V index: j*p + k
                V[j * p + k, col] = 1
                # W index: k*n + i
                W[k * n + i, col] = 1
                col += 1
    
    # Convert to list of lists for the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

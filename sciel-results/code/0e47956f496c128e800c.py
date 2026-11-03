import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    """
    Returns the schoolbook decomposition for matrix multiplication of 3x3 by 3x7.
    Rank = 3 * 3 * 7 = 63.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U maps input A indices (i, j) to column col
                U[i * m + j, col] = 1
                # V maps input B indices (j, k) to column col
                V[j * p + k, col] = 1
                # W maps output C indices (k, i) to column col
                # Convention: W[k*n + i, col]
                W[k * n + i, col] = 1
                col += 1
                
    # Convert to lists of lists as per interface requirement (though numpy arrays are often accepted, 
    # the seed used lists. The prompt says "factor matrices of shapes... with integer/half-integer entries".
    # Returning numpy arrays is usually fine if the evaluator handles it, but to be safe and match seed style:
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

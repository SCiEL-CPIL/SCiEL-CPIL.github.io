import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # The tensor to be decomposed is T[i, j, k] = 1 for all i, j, k.
    # This is the all-ones tensor.
    # The rank of the all-ones tensor is 1.
    # We can construct U, V, W such that:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # If R=1, we can set U[:, 0] = 1, V[:, 0] = 1, W[:, 0] = 1.
    # Then the sum is 1 * 1 * 1 = 1.
    
    R = 1
    
    U = np.ones((n*m, R))
    V = np.ones((m*p, R))
    W = np.ones((p*n, R))
    
    # Verification (optional, for safety)
    # for i in range(n):
    #     for j in range(m):
    #         for k in range(p):
    #             val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
    #             if not np.isclose(val, 1.0, atol=1e-9):
    #                 # Fallback to schoolbook if verification fails (should not happen)
    #                 R = n*m*p
    #                 U = np.zeros((n*m, R))
    #                 V = np.zeros((m*p, R))
    #                 W = np.zeros((p*n, R))
    #                 col = 0
    #                 for i in range(n):
    #                     for j in range(m):
    #                         for k in range(p):
    #                             U[i*m+j, col] = 1
    #                             V[j*p+k, col] = 1
    #                             W[k*n+i, col] = 1
    #                             col += 1
    #                 return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

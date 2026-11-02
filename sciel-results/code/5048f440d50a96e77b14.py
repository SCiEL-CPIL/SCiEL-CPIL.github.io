import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank) = (12, rank)
    # V: (m*p, rank) = (16, rank)
    # W: (p*n, rank) = (12, rank)
    
    # Total rank is 4 * 11 = 44
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Blocks of J (middle index) and K (output index)
    # J is size 2, K is size 2
    # 4 combinations
    blocks = [
        ([0, 1], [0, 1]),
        ([0, 1], [2, 3]),
        ([2, 3], [0, 1]),
        ([2, 3], [2, 3])
    ]
    
    col_offset = 0
    for J, K in blocks:
        j0, j1 = J
        k0, k1 = K
        
        # Strassen 7 terms
        # P1
        U[0*m + j0, col_offset] += 1
        U[1*m + j0, col_offset] += 1
        U[0*m + j1, col_offset] += 1
        U[1*m + j1, col_offset] += 1
        
        V[j0*p + k0, col_offset] += 1
        V[j0*p + k1, col_offset] += 1
        V[j1*p + k0, col_offset] += 1
        V[j1*p + k1, col_offset] += 1
        
        W[k0*n + 0, col_offset] += 1
        W[k1*n + 1, col_offset] += 1
        col_offset += 1
        
        # P2
        U[1*m + j0, col_offset] += 1
        U[1*m + j1, col_offset] += 1
        
        V[j0*p + k0, col_offset] += 1
        
        W[k0*n + 1, col_offset] += 1
        W[k1*n + 1, col_offset] -= 1
        col_offset += 1
        
        # P3
        U[0*m + j0, col_offset] += 1
        
        V[j0*p + k1, col_offset] += 1
        V[j1*p + k1, col_offset] -= 1
        
        W[k1*n + 0, col_offset] += 1
        W[k1*n + 1, col_offset] += 1
        col_offset += 1
        
        # P4
        U[1*m + j1, col_offset] += 1
        
        V[j1*p + k0, col_offset] += 1
        V[j0*p + k0, col_offset] -= 1
        
        W[k0*n + 0, col_offset] += 1
        W[k0*n + 1, col_offset] += 1
        col_offset += 1
        
        # P5
        U[0*m + j0, col_offset] += 1
        U[0*m + j1, col_offset] += 1
        
        V[j1*p + k1, col_offset] += 1
        
        W[k0*n + 0, col_offset] -= 1
        W[k1*n + 0, col_offset] += 1
        col_offset += 1
        
        # P6
        U[1*m + j0, col_offset] += 1
        U[0*m + j0, col_offset] -= 1
        
        V[j0*p + k0, col_offset] += 1
        V[j0*p + k1, col_offset] += 1
        
        W[k1*n + 1, col_offset] += 1
        col_offset += 1
        
        # P7
        U[0*m + j1, col_offset] += 1
        U[1*m + j1, col_offset] -= 1
        
        V[j1*p + k0, col_offset] += 1
        V[j1*p + k1, col_offset] += 1
        
        W[k0*n + 0, col_offset] += 1
        col_offset += 1
        
        # Trivial 4 terms for row i=2
        # T1: A[2, j0] B[j0, k0] -> C[2, k0]
        U[2*m + j0, col_offset] += 1
        V[j0*p + k0, col_offset] += 1
        W[k0*n + 2, col_offset] += 1
        col_offset += 1
        
        # T2: A[2, j1] B[j1, k0] -> C[2, k0]
        U[2*m + j1, col_offset] += 1
        V[j1*p + k0, col_offset] += 1
        W[k0*n + 2, col_offset] += 1
        col_offset += 1
        
        # T3: A[2, j0] B[j0, k1] -> C[2, k1]
        U[2*m + j0, col_offset] += 1
        V[j0*p + k1, col_offset] += 1
        W[k1*n + 2, col_offset] += 1
        col_offset += 1
        
        # T4: A[2, j1] B[j1, k1] -> C[2, k1]
        U[2*m + j1, col_offset] += 1
        V[j1*p + k1, col_offset] += 1
        W[k1*n + 2, col_offset] += 1
        col_offset += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

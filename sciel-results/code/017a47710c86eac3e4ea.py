"""
Matrix multiplication 2,3,4 tensor decomposition.
Returns the schoolbook decomposition with rank 24.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    """
    Construct the factor matrices U, V, W for the matrix multiplication tensor
    of dimensions n=2, m=3, p=4.
    
    The tensor is defined by T[i,j,k,l] = delta_{j,l} * delta_{k,i} (conceptually),
    but specifically for the decomposition:
    C_{ik} = sum_{j} A_{ij} B_{jk}
    
    We express this as:
    C_{ik} = sum_{r} U_{(i,j), r} V_{(j,k), r} W_{(i,k), r} ??? No.
    
    The standard decomposition is:
    C_{ik} = sum_{j} A_{ij} B_{jk}
    
    This can be written as a sum of rank-1 tensors:
    T = sum_{r=1}^{nmp} u^{(r)} \otimes v^{(r)} \otimes w^{(r)}
    
    Where:
    u^{(r)} corresponds to the entry A_{i,j}
    v^{(r)} corresponds to the entry B_{j,k}
    w^{(r)} corresponds to the entry C_{i,k}
    
    Specifically, for each triplet (i, j, k):
    u has a 1 at index i*m + j
    v has a 1 at index j*p + k
    w has a 1 at index k*n + i (trace convention W[k*n+i, r])
    
    Rank is n*m*p = 2*3*4 = 24.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=4
    rank = n * m * p      # 24
    
    # Shapes:
    # U: (n*m, rank) -> (6, 24)
    # V: (m*p, rank) -> (12, 24)
    # W: (p*n, rank) -> (8, 24)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index for A[i,j] in U
                u_idx = i * m + j
                # Index for B[j,k] in V
                v_idx = j * p + k
                # Index for C[i,k] in W (trace convention: row index is k*n + i)
                w_idx = k * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                
                col += 1
                
    return {"U": U, "V": V, "W": W}

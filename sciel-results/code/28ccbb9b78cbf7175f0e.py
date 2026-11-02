"""
Program to find a rank-minimal exact tensor decomposition for matrix multiplication 3x3 by 3x7.
The trivial decomposition has rank n*m*p = 3*3*7 = 63.
This program constructs that decomposition.
"""
import numpy as np

def construct():
    n, m, p = 3, 3, 7
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    # The tensor T_{i,j,k} corresponds to the entry (i,k) of the product C = A * B
    # A is n x m, B is m x p, C is n x p.
    # The standard basis vectors for the tensor decomposition are:
    # e_i^U (for A), e_j^V (for B), e_k^W (for C)
    # Wait, the interface defines:
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    # The tensor is T_{i,j,k} = delta_{i,k} * delta_{j,j}? No.
    # Matrix multiplication: C_{ik} = sum_j A_{ij} B_{jk}
    # The tensor T has indices i, j, k. T_{i,j,k} = 1 if this term contributes to C_{ik}.
    # Actually, the tensor is defined such that T_{i,j,k} A_{ij} B_{jk} = C_{ik}.
    # So T_{i,j,k} = delta_{i,i} delta_{j,j} delta_{k,k}? No.
    # The tensor is the identity on the multiplication map.
    # T_{i,j,k} = 1 if we are computing C_{i,k} using A_{i,j} and B_{j,k}.
    # So for each i, j, k, there is a term A_{ij} B_{jk} that contributes to C_{ik}.
    # The rank-1 decomposition term for (i,j,k) is:
    # U_{i*m+j, r} = 1
    # V_{j*p+k, r} = 1
    # W_{k*n+i, r} = 1
    # This matches the current program's logic.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

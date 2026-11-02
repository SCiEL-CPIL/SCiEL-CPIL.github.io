import numpy as np

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    # Standard canonical decomposition
    # Rank = n * m * p
    rank = n * m * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Iterate over all indices i, j, k
    # The tensor entry T[i, j, k] is 1 if the term contributes to C[i, k] via A[i, j] and B[j, k]
    # In the decomposition C = sum_r U_r * V_r * W_r
    # C[i, k] = sum_r U[i, j_r] * V[j_r, k_r] * W[k_r, i] ? No.
    # The standard form is:
    # C_{ik} = sum_{j} A_{ij} B_{jk}
    # We can write this as sum_{i,j,k} delta_{ij, A} delta_{jk, B} delta_{ik, C}
    # The rank is n*m*p.
    # Each term corresponds to a specific (i, j, k) triple?
    # No, the standard decomposition has terms indexed by (i, j, k) where the term is:
    # U is basis vector e_i (in A space? No, A is n x m)
    
    # Let's use the explicit definition:
    # Term r corresponds to (i, j, k)
    # U_r is the basis vector for A_{ij} -> index i*m + j
    # V_r is the basis vector for B_{jk} -> index j*p + k
    # W_r is the basis vector for C_{ik} -> index k*n + i
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U component
                U[i * m + j, idx] = 1.0
                # V component
                V[j * p + k, idx] = 1.0
                # W component
                W[k * n + i, idx] = 1.0
                idx += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

"""Tensor decomposition for 3x3 by 3x5 matrix multiplication."""
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The multiplication tensor T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need U (n*m x r), V (m*p x r), W (p*n x r) such that:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Schoolbook decomposition gives rank n*m*p = 3*3*5 = 45
    # Let's try to find a lower rank decomposition.
    
    # Known lower bounds and constructions:
    # For 3x3 * 3x3, the best known is 23 (Laderman) or possibly lower.
    # For 3x3 * 3x5, we can try various approaches.
    
    # Let's try to construct a decomposition of rank r where r < 45.
    # One approach: use the fact that we can sometimes combine terms.
    
    # For now, let's return the schoolbook decomposition which is guaranteed correct.
    # rank = 45
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1.0
                V[j * p + k, column] = 1.0
                W[k * n + i, column] = 1.0
                column += 1
    
    # Convert to lists
    U_list = [[float(U[i, j]) for j in range(rank)] for i in range(n * m)]
    V_list = [[float(V[i, j]) for j in range(rank)] for i in range(m * p)]
    W_list = [[float(W[i, j]) for j in range(rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

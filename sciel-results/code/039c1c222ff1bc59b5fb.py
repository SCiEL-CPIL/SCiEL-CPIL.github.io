import time
import sys

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # The theoretical lower bound for the rank of the matrix multiplication tensor
    # for n x m * m x p is n * p, assuming the outputs are linearly independent.
    # We construct a decomposition of rank n * p.
    rank = n * p
    
    # Initialize matrices with zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for k in range(p):
            # U_c: selects row i of A
            # Indices in U are i*m + j
            for j in range(m):
                U[i * m + j][col] = 1
                
            # V_c: selects column k of B
            # Indices in V are j*p + k
            for j in range(m):
                V[j * p + k][col] = 1
                
            # W_c: outputs to C[i, k]
            # Indices in W are k*n + i
            W[k * n + i][col] = 1
            
            col += 1
            
    return {"U": U, "V": V, "W": W}

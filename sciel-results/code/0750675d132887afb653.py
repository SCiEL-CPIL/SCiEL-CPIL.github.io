"""Rewritten program for matrix multiplication 4,4,7."""

DIMENSIONS = [4, 4, 7]

def construct():
    """
    Construct the tensor decomposition for matrix multiplication A (n x m) * B (m x p) = C (n x p).
    Here n=4, m=4, p=7.
    The decomposition is of the form T = sum_{r=1}^{Rank} U_r \otimes V_r \otimes W_r.
    We use the schoolbook method where each term corresponds to one product term a_{i,j} * b_{j,k}.
    Rank = n * m * p = 4 * 4 * 7 = 112.
    
    U has shape (n*m, rank). Index row = i*m + j.
    V has shape (m*p, rank). Index row = j*p + k.
    W has shape (p*n, rank). Index row = k*n + i (Trace convention W[k*n+i, r]).
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][column] = 1
                # V index: j*p + k
                V[j * p + k][column] = 1
                # W index: k*n + i
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}

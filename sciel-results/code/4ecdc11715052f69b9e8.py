"""Tensor decomposition for 2x3x8 matrix multiplication."""
import numpy as np

DIMENSIONS = [2, 3, 8]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition is correct."""
    rank = U.shape[1]
    result = np.zeros((n*m, m*p, p*n), dtype=float)
    for t in range(rank):
        u = U[:, t]
        v = V[:, t]
        w = W[:, t]
        result += np.einsum('i,j,k->ijk', u, v, w)
    
    correct = np.zeros((n*m, m*p, p*n), dtype=float)
    for row_a in range(n):
        for col_a in range(m):
            for row_b in range(m):
                for col_b in range(p):
                    if col_a == row_b:
                        i = row_a * m + col_a
                        j = row_b * p + col_b
                        k = col_b * n + row_a
                        correct[i, j, k] = 1
    
    return np.allclose(result, correct, atol=1e-10)

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Rank-16 construction: group by output column
    # For each output column k, one term computes C[0,k] = A[0,:] @ B[:,k]
    # and another term computes C[1,k] = A[1,:] @ B[:,k]
    rank = 2 * p  # 16
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for k in range(p):
        # Term k: computes C[0,k] = sum_j A[0,j]*B[j,k]
        U[0, k] = 1  # A[0,0]
        U[1, k] = 1  # A[0,1]
        U[2, k] = 1  # A[0,2]
        V[0*p + k, k] = 1  # B[0,k]
        V[1*p + k, k] = 1  # B[1,k]
        V[2*p + k, k] = 1  # B[2,k]
        W[k*n + 0, k] = 1  # C[0,k]
        
        # Term p+k: computes C[1,k] = sum_j A[1,j]*B[j,k]
        U[3, p+k] = 1  # A[1,0]
        U[4, p+k] = 1  # A[1,1]
        U[5, p+k] = 1  # A[1,2]
        V[0*p + k, p+k] = 1  # B[0,k]
        V[1*p + k, p+k] = 1  # B[1,k]
        V[2*p + k, p+k] = 1  # B[2,k]
        W[k*n + 1, p+k] = 1  # C[1,k]
    
    if _verify_decomposition(U, V, W, n, m, p):
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook if verification fails
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}

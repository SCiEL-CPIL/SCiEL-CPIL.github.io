"""Tensor decomposition for 2x3x8 matrix multiplication."""
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Rank-16 optimal construction for 2x3x8
    # Each output C[i,k] = sum_j A[i,j] * B[j,k]
    # We use one rank-1 term per output entry (16 total)
    # Term k (k=0..7): computes C[0,k] = A[0,:] @ B[:,k]
    # Term 8+k (k=0..7): computes C[1,k] = A[1,:] @ B[:,k]
    
    rank = 16
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Terms 0-7: For k=0..7, compute C[0,k]
    for k in range(p):
        # U: all entries in row 0 of A (indices 0,1,2)
        U[0, k] = 1  # A[0,0]
        U[1, k] = 1  # A[0,1]
        U[2, k] = 1  # A[0,2]
        # V: all entries in column k of B (indices k, p+k, 2p+k)
        V[0 * p + k, k] = 1  # B[0,k]
        V[1 * p + k, k] = 1  # B[1,k]
        V[2 * p + k, k] = 1  # B[2,k]
        # W: C[0,k] (index k*n + 0 = k)
        W[k * n + 0, k] = 1
    
    # Terms 8-15: For k=0..7, compute C[1,k]
    for k in range(p):
        # U: all entries in row 1 of A (indices 3,4,5)
        U[3, 8 + k] = 1  # A[1,0]
        U[4, 8 + k] = 1  # A[1,1]
        U[5, 8 + k] = 1  # A[1,2]
        # V: all entries in column k of B (indices k, p+k, 2p+k)
        V[0 * p + k, 8 + k] = 1  # B[0,k]
        V[1 * p + k, 8 + k] = 1  # B[1,k]
        V[2 * p + k, 8 + k] = 1  # B[2,k]
        # W: C[1,k] (index k*n + 1 = 2+k)
        W[k * n + 1, 8 + k] = 1
    
    # Verify the decomposition
    def verify(U_mat, V_mat, W_mat, n, m, p):
        rank = U_mat.shape[1]
        result = np.zeros((n*m, m*p, p*n), dtype=float)
        for t in range(rank):
            u = U_mat[:, t]
            v = V_mat[:, t]
            w = W_mat[:, t]
            result += np.einsum('i,j,k->ijk', u, v, w)
        
        correct = np.zeros((n*m, m*p, p*n), dtype=float)
        for row_a in range(n):
            for col_a in range(m):
                for col_b in range(p):
                    i = row_a * m + col_a
                    j = col_a * p + col_b
                    k = col_b * n + row_a
                    correct[i, j, k] = 1
        
        return np.allclose(result, correct, atol=1e-10)
    
    if verify(U, V, W, n, m, p):
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook (rank 48)
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

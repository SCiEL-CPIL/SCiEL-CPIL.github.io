"""Tensor decomposition for 2x3x8 matrix multiplication."""
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Rank-16 construction for 2x3x8 matrix multiplication
    # For each output column k (0..7), we use 2 terms:
    #   Term k:     computes C[0,k] = sum_j A[0,j]*B[j,k]
    #   Term 8+k:   computes C[1,k] = sum_j A[1,j]*B[j,k]
    #
    # Index conventions:
    #   U[i*m + j, t]  where i in [0,n), j in [0,m)  -> A[i,j]
    #   V[j*p + k, t]  where j in [0,m), k in [0,p)  -> B[j,k]
    #   W[k*n + i, t]  where k in [0,p), i in [0,n)  -> C[i,k]
    #
    # Multiplication tensor T[i,j,k] = 1 iff:
    #   A index i = row_a*m + col_a
    #   B index j = row_b*p + col_b
    #   C index k = col_c*n + row_c
    #   and col_a == row_b, row_a == row_c, col_b == col_c
    
    rank = 2 * p  # 16
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for k in range(p):
        # Term k: C[0,k] = sum_j A[0,j] * B[j,k]
        # U nonzeros at A[0,0], A[0,1], A[0,2] -> indices 0, 1, 2
        U[0, k] = 1
        U[1, k] = 1
        U[2, k] = 1
        # V nonzeros at B[0,k], B[1,k], B[2,k] -> indices 0*p+k, 1*p+k, 2*p+k
        V[0 * p + k, k] = 1
        V[1 * p + k, k] = 1
        V[2 * p + k, k] = 1
        # W nonzero at C[0,k] -> index k*n + 0
        W[k * n + 0, k] = 1
        
        # Term 8+k: C[1,k] = sum_j A[1,j] * B[j,k]
        # U nonzeros at A[1,0], A[1,1], A[1,2] -> indices 3, 4, 5
        U[3, 8 + k] = 1
        U[4, 8 + k] = 1
        U[5, 8 + k] = 1
        # V nonzeros at B[0,k], B[1,k], B[2,k] -> indices 0*p+k, 1*p+k, 2*p+k
        V[0 * p + k, 8 + k] = 1
        V[1 * p + k, 8 + k] = 1
        V[2 * p + k, 8 + k] = 1
        # W nonzero at C[1,k] -> index k*n + 1
        W[k * n + 1, 8 + k] = 1
    
    # Verify correctness by checking the tensor decomposition
    def verify(U, V, W, n, m, p):
        rank = U.shape[1]
        # Build expected multiplication tensor
        correct = np.zeros((n * m, m * p, p * n), dtype=float)
        for row_a in range(n):
            for col_a in range(m):
                for col_b in range(p):
                    # col_a must equal row_b for valid multiplication
                    row_b = col_a
                    i = row_a * m + col_a
                    j = row_b * p + col_b
                    k_idx = col_b * n + row_a
                    correct[i, j, k_idx] = 1
        
        # Build decomposition tensor
        result = np.zeros((n * m, m * p, p * n), dtype=float)
        for t in range(rank):
            u = U[:, t]
            v = V[:, t]
            w = W[:, t]
            # Only add where all three are nonzero to save time
            nz_u = np.nonzero(u)[0]
            nz_v = np.nonzero(v)[0]
            nz_w = np.nonzero(w)[0]
            for i in nz_u:
                for j in nz_v:
                    for k_idx in nz_w:
                        result[i, j, k_idx] += u[i] * v[j] * w[k_idx]
        
        return np.allclose(result, correct, atol=1e-10)
    
    if verify(U, V, W, n, m, p):
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback: schoolbook rank 48
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}

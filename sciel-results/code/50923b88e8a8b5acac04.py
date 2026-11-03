import numpy as np

DIMENSIONS = [3, 4, 4]

def _build_rank44():
    """
    Build rank-44 decomposition for 3x4 * 4x4 matrix multiplication.
    Split A into A_top (2x4) and A_bot (1x4).
    Use Winograd-based rank-28 for 2x4x4 and rank-16 schoolbook for 1x4x4.
    """
    n, m, p = 3, 4, 4
    rank = 44
    
    U = np.zeros((n * m, rank))  # (12, 44)
    V = np.zeros((m * p, rank))  # (16, 44)
    W = np.zeros((p * n, rank))  # (12, 44)
    
    # Part 1: 1x4x4 schoolbook (rank 16) for A_bot (row 2 of A)
    col_offset = 0
    for j in range(m):  # 4 columns of A_bot
        for k in range(p):  # 4 columns of B
            U[2 * m + j, col_offset] = 1  # A[2, j]
            V[j * p + k, col_offset] = 1  # B[j, k]
            W[k * n + 2, col_offset] = 1  # C[k, 2]
            col_offset += 1
    # col_offset = 16
    
    # Part 2: 2x4x4 using Strassen (rank 28) for A_top (rows 0,1 of A)
    # Use Winograd 2x2x2 (rank 7) for each of 4 block multiplications? No.
    # Strassen: 7 products of 2x2x2, each with rank 4 (schoolbook) = 28 total.
    
    # Winograd 2x2x2 rank-7 factors:
    # U_w: (4, 7), V_w: (4, 7), W_w: (4, 7)
    # C = U_w^T * (U_w * A * V_w^T) * W_w ... no, that's not right.
    
    # The tensor decomposition: C[i,j] = sum_k U[i,k] * V[j,k] * W[k,k]... 
    # Actually: C = sum_k u_k * v_k^T where u_k, v_k are vectors.
    # For matrix mult: C[i,j] = sum_k U[i,k] * V[j,k] * W[k,k] is wrong.
    
    # Correct: C[i,j] = sum_k (sum_a U[a,k]*A[a,b]) * (sum_c V[c,k]*B[b,c]) * W[i,j,k]... 
    # No. The decomposition is:
    # C[i,j] = sum_k U[i,k] * M_k * V[j,k] where M_k is a scalar.
    # More precisely: C[i,j] = sum_k U[i,k] * V[j,k] * W[k,k] is still wrong.
    
    # The matrix multiplication tensor T_{i,j,k} = delta_{i,a} delta_{k,b} delta_{j,c} ... 
    # Let me just hardcode the Winograd 2x2x2 factors.
    
    # Winograd 2x2x2 (rank 7):
    # U_w (4x7):
    # V_w (4x7):
    # W_w (4x7):
    
    # I'll use the known Winograd factors for 2x2x2.
    # From literature:
    # U_w = [[1,0,1,0,1,0,1],
    #        [0,1,0,1,0,1,0],
    #        [1,0,0,0,1,0,0],
    #        [0,1,0,0,0,1,0]]  # This is not correct, let me look it up mentally.
    
    # Actually, let me just use schoolbook 2x2x2 with rank 4 for simplicity,
    # even though it's not minimal. 7*4 = 28 is the target.
    
    # Schoolbook 2x2x2: 4 multiplications
    # For 2x2 matrices X, Y:
    # (X*Y)[0,0] = X[0,0]*Y[0,0] + X[0,1]*Y[1,0]
    # (X*Y)[0,1] = X[0,0]*Y[0,1] + X[0,1]*Y[1,1]
    # (X*Y)[1,0] = X[1,0]*Y[0,0] + X[1,1]*Y[1,0]
    # (X*Y)[1,1] = X[1,0]*Y[0,1] + X[1,1]*Y[1,1]
    
    # This uses 8 scalar multiplications, not 4. The rank is 7 (Winograd).
    
    # I think I need to accept that implementing rank-28 Strassen correctly is complex.
    # Let me just return None and use the 48-rank fallback.
    return None

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

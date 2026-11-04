import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify the tensor decomposition reconstructs matrix multiplication exactly."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    # Check zero constraints
    import random
    random.seed(42)
    for _ in range(2000):
        a = random.randint(0, n*m-1)
        b = random.randint(0, m*p-1)
        g = random.randint(0, p*n-1)
        i = a // m; j = a % m
        jp = b // p; k = b % p
        kn = g // n; ip = g % n
        if i != ip or j != jp or k != kn:
            val = 0.0
            for r in range(rank):
                val += U[a, r] * V[b, r] * W[g, r]
            if abs(val) > tol:
                return False
    return True

def _strassen_2x2():
    """
    Strassen's 7-multiplication decomposition for 2x2 matrix multiplication.
    Returns U, V, W for the 2x2x2 tensor with rank 7.
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((n*m, rank))  # 4x7
    V = np.zeros((m*p, rank))  # 4x7
    W = np.zeros((p*n, rank))  # 4x7
    
    # M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)*B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3 = A00*(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11*(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00+A01)*B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[3, 4] = 1
    
    # M6 = (A10-A00)*(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1; W[2, 5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[1, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def _build_block_strassen(n, m, p):
    """
    Build a decomposition using Strassen for 2x2 blocks and schoolbook for the rest.
    For 4x5x6:
    - Partition rows of A (4 rows) into 2 blocks of 2
    - Partition columns of A / rows of B (5 cols/rows) into blocks: 2+2+1
    - Partition columns of B (6 cols) into blocks: 2+2+2
    
    For each pair of blocks (block_i of A, block_j of B), if both are 2x2, use Strassen.
    Otherwise use schoolbook.
    """
    # Block partitions
    # A is 4x5: row blocks [0:2], [2:4]; col blocks [0:2], [2:4], [4:5]
    # B is 5x6: row blocks [0:2], [2:4], [4:5]; col blocks [0:2], [2:4], [4:6]
    
    a_row_blocks = [(0, 2), (2, 4)]  # 2 blocks of size 2
    a_col_blocks = [(0, 2), (2, 4), (4, 5)]  # 2 blocks of size 2, 1 of size 1
    b_row_blocks = [(0, 2), (2, 4), (4, 5)]  # same as a_col_blocks
    b_col_blocks = [(0, 2), (2, 4), (4, 6)]  # 3 blocks of size 2
    
    # We'll build the decomposition by iterating over all block pairs
    # and accumulating the rank contributions
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    for ai, (a_r0, a_r1) in enumerate(a_row_blocks):
        a_row_size = a_r1 - a_r0
        for bi, (a_c0, a_c1) in enumerate(a_col_blocks):
            a_col_size = a_c1 - a_c0
            # This block of A is a_row_size x a_col_size
            # The corresponding block of B is a_col_size x b_col_size
            for cj, (b_c0, b_c1) in enumerate(b_col_blocks):
                b_col_size = b_c1 - b_c0
                # The result block C is a_row_size x b_col_size
                # We need to compute A[a_r0:a_r1, a_c0:a_c1] @ B[a_c0:a_c1, b_c0:b_c1]
                
                # If both blocks are 2x2 (in the relevant dimensions), use Strassen
                # A block is a_row_size x a_col_size
                # B block is a_col_size x b_col_size
                # For Strassen, we need a_col_size == 2 (the shared dimension)
                # and a_row_size == 2, b_col_size == 2 for full 2x2x2
                
                if a_row_size == 2 and a_col_size == 2 and b_col_size == 2:
                    # Use Strassen for this 2x2x2 block
                    U_s, V_s, W_s = _strassen_2x2()
                    
                    # Map the Strassen indices to global indices
                    # U_s has shape (4, 7): indices (i_local, j_local) where i_local in [0,1], j_local in [0,1]
                    # Global: row = a_r0 + i_local, col = a_c0 + j_local
                    U_g = np.zeros((n * m, 7))
                    for i_local in range(2):
                        for j_local in range(2):
                            U_g[(a_r0 + i_local) * m + (a_c0 + j_local), :] = U_s[i_local * 2 + j_local, :]
                    
                    # V_s has shape (4, 7): indices (j_local, k_local) where j_local in [0,1], k_local in [0,1]
                    # Global: row = a_c0 + j_local, col = b_c0 + k_local
                    V_g = np.zeros((m * p, 7))
                    for j_local in range(2):
                        for k_local in range(2):
                            V_g[(a_c0 + j_local) * p + (b_c0 + k_local), :] = V_s[j_local * 2 + k_local, :]
                    
                    # W_s has shape (4, 7): indices (k_local, i_local) where k_local in [0,1], i_local in [0,1]
                    # Global: row = b_c0 + k_local, col = a_r0 + i_local
                    W_g = np.zeros((p * n, 7))
                    for k_local in range(2):
                        for i_local in range(2):
                            W_g[(b_c0 + k_local) * n + (a_r0 + i_local), :] = W_s[k_local * 2 + i_local, :]
                    
                    U_parts.append(U_g)
                    V_parts.append(V_g)
                    W_parts.append(W_g)
                else:
                    # Use schoolbook for this block
                    # The block multiplication is a_row_size x a_col_size @ a_col_size x b_col_size
                    # Number of terms: a_row_size * a_col_size * b_col_size
                    rank_block = a_row_size * a_col_size * b_col_size
                    
                    U_g = np.zeros((n * m, rank_block))
                    V_g = np.zeros((m * p, rank_block))
                    W_g = np.zeros((p * n, rank_block))
                    
                    col = 0
                    for i_local in range(a_row_size):
                        for j_local in range(a_col_size):
                            for k_local in range(b_col_size):
                                U_g[(a_r0 + i_local) * m + (a_c0 + j_local), col] = 1
                                V_g[(a_c0 + j_local) * p + (b_c0 + k_local), col] = 1
                                W_g[(b_c0 + k_local) * n + (a_r0 + i_local), col] = 1
                                col += 1
                    
                    U_parts.append(U_g)
                    V_parts.append(V_g)
                    W_parts.append(W_g)
    
    # Concatenate all parts
    U = np.hstack(U_parts)
    V = np.hstack(V_parts)
    W = np.hstack(W_parts)
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as baseline
    best_U, best_V, best_W = _build_schoolbook(n, m, p)
    best_rank = best_U.shape[1]
    
    # Try block Strassen
    try:
        U_bs, V_bs, W_bs = _build_block_strassen(n, m, p)
        if U_bs.shape[1] < best_rank:
            # Verify
            if _verify_decomposition(U_bs, V_bs, W_bs, n, m, p):
                best_U, best_V, best_W = U_bs, V_bs, W_bs
                best_rank = best_U.shape[1]
    except Exception as e:
        pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

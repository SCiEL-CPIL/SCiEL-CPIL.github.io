"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x6 matrix.
    Uses block decomposition with Strassen's rank-7 algorithm for 2x2x2 blocks.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # Must finish within 120s

    # Strassen's 7-term decomposition for 2x2x2 matrix multiplication
    # C = A * B where A, B, C are 2x2 matrices
    # The 7 bilinear forms:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # Strassen factors for 2x2x2:
    # U factors (for A indices): row = i*2 + j, col = term (0-6)
    strassen_U = np.array([
        # A11 (i=0,j=0): M1, M3, M5, M6
        [1, 0, 1, 0, 1, -1, 0],
        # A12 (i=0,j=1): M5, M7
        [0, 0, 0, 0, 1, 0, 1],
        # A21 (i=1,j=0): M2, M4, M6
        [0, 1, 0, 1, 0, 1, 0],
        # A22 (i=1,j=1): M1, M2, M4, M7
        [1, 1, 0, 1, 0, 0, -1],
    ], dtype=float)

    # V factors (for B indices): row = j*2 + k, col = term (0-6)
    strassen_V = np.array([
        # B11 (j=0,k=0): M1, M2, M4, M6
        [1, 1, 0, -1, 0, 1, 0],
        # B12 (j=0,k=1): M3, M6
        [1, 0, 1, 0, 0, 1, 0],
        # B21 (j=1,k=0): M4, M6, M7
        [-1, 0, 0, 1, 0, 0, 1],
        # B22 (j=1,k=1): M1, M3, M5, M7
        [1, 0, -1, 0, 1, 0, 1],
    ], dtype=float)

    # W factors (for C indices): row = k*2 + i, col = term (0-6)
    # C11 (k=0, i=0): M1, M4, M5, M7
    # C12 (k=0, i=1): M3, M5
    # C21 (k=1, i=0): M2, M4
    # C22 (k=1, i=1): M1, M2, M3, M6
    strassen_W = np.array([
        # C11 (k=0,i=0): row = 0*2+0 = 0
        [1, 0, 0, 1, -1, 0, 1],
        # C12 (k=0,i=1): row = 0*2+1 = 1
        [0, 0, 1, 0, 1, 0, 0],
        # C21 (k=1,i=0): row = 1*2+0 = 2
        [0, 1, 0, 1, 0, 0, 0],
        # C22 (k=1,i=1): row = 1*2+1 = 3
        [1, -1, 1, 0, 0, 1, 0],
    ], dtype=float)

    def verify_strassen_2x2x2():
        """Verify Strassen decomposition for 2x2x2."""
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    val = np.dot(strassen_U[i*2+j], strassen_V[j*2+k] * strassen_W[k*2+i])
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True

    # Build the 4x4x6 decomposition using block structure
    # We decompose 4x4x6 into blocks of 2x2x2
    # A is 4x4: split into 2x2 blocks: A[0:2,0:2], A[0:2,2:4], A[2:4,0:2], A[2:4,2:4]
    # B is 4x6: split into 2x2 blocks: B[0:2,0:2], B[0:2,2:4], B[0:2,4:6], B[2:4,0:2], B[2:4,2:4], B[2:4,4:6]
    # C is 4x6: split into 2x2 blocks: C[0:2,0:2], C[0:2,2:4], C[0:2,4:6], C[2:4,0:2], C[2:4,2:4], C[2:4,4:6]
    #
    # C[i_block, k_block] = sum_j_block A[i_block, j_block] * B[j_block, k_block]
    #
    # We have 2x3 = 6 output blocks, each computed from 2 input blocks
    # Total: 6 blocks * 2 j-blocks * 7 Strassen terms = 84 rank terms

    # Build the full decomposition
    total_rank = 0
    # Number of (i_block, j_block, k_block) combinations
    n_blocks = 2  # 4/2
    m_blocks = 2  # 4/2
    p_blocks = 3  # 6/2

    # Each (i_blk, j_blk, k_blk) triple contributes 7 terms
    total_rank = n_blocks * m_blocks * p_blocks * 7

    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)

    col = 0
    for i_blk in range(n_blocks):
        for j_blk in range(m_blocks):
            for k_blk in range(p_blocks):
                for term in range(7):
                    # U index: A[i_blk*2 + i_local, j_blk*2 + j_local]
                    for i_local in range(2):
                        for j_local in range(2):
                            a_idx = (i_blk * 2 + i_local) * m + (j_blk * 2 + j_local)
                            U[a_idx, col] = strassen_U[i_local * 2 + j_local, term]
                    
                    # V index: B[j_blk*2 + j_local, k_blk*2 + k_local]
                    for j_local in range(2):
                        for k_local in range(2):
                            b_idx = (j_blk * 2 + j_local) * p + (k_blk * 2 + k_local)
                            V[b_idx, col] = strassen_V[j_local * 2 + k_local, term]
                    
                    # W index: C[i_blk*2 + i_local, k_blk*2 + k_local] -> k*n + i
                    for i_local in range(2):
                        for k_local in range(2):
                            c_idx = (k_blk * 2 + k_local) * n + (i_blk * 2 + i_local)
                            W[c_idx, col] = strassen_W[k_local * 2 + i_local, term]
                    
                    col += 1

    # Verify the decomposition
    def check_decomposition(U, V, W, tol=1e-6):
        """Check if the decomposition is exact (within tolerance)."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    max_err = check_decomposition(U, V, W)
    
    if max_err > 1e-6:
        # Fall back to schoolbook if Strassen construction has errors
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

    # Convert to lists with rounded values (snap to integers)
    def snap_to_half_int(val):
        """Snap a value to the nearest half-integer."""
        return round(val * 2) / 2.0

    U_list = [[snap_to_half_int(val) for val in row] for row in U.tolist()]
    V_list = [[snap_to_half_int(val) for val in row] for row in V.tolist()]
    W_list = [[snap_to_half_int(val) for val in row] for row in W.tolist()]

    # Final verification
    final_err = check_decomposition(np.array(U_list), np.array(V_list), np.array(W_list))
    if final_err > 1e-6:
        # Fall back to schoolbook
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
        U_list = [[round(val, 10) for val in row] for row in U.tolist()]
        V_list = [[round(val, 10) for val in row] for row in V.tolist()]
        W_list = [[round(val, 10) for val in row] for row in W.tolist()]

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

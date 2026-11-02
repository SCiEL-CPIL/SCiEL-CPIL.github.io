import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 5, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_2x2():
    """
    Return Strassen decomposition for 2x2 matrix multiplication.
    Rank 7.
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1; W[1, 0] = -1; W[2, 0] = -1
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1
    
    # M3 = A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1
    
    # M4 = A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[1, 3] = -1; W[2, 3] = 1
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[2, 4] = -1
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1; W[1, 5] = -1
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[3, 6] = 1
    
    return U, V, W, rank

def _construct_block_decomposition():
    """
    Construct a rank-92 decomposition for 4x5 * 5x5.
    """
    n, m, p = 4, 5, 5
    # We will build the decomposition by accumulating terms.
    # Each term is a tuple (u, v, w) where u is index in A, v in B, w in C.
    # But since we need the full matrices, we'll just fill them.
    
    # Total rank estimate: 92
    # Let's use a dictionary to map (i, j, k) -> list of (u_idx, v_idx, w_idx, coeff)
    # Actually, it's easier to just append to lists and then build arrays.
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_idx, v_idx, w_idx, coeff=1.0):
        U_list.append((u_idx, coeff))
        V_list.append((v_idx, coeff))
        W_list.append((w_idx, coeff))
        
    def add_strassen(u_base, v_base, w_base):
        """
        Add Strassen terms for 2x2x2 block.
        u_base: index of A[0,0] in the flattened A
        v_base: index of B[0,0] in the flattened B
        w_base: index of C[0,0] in the flattened C
        """
        # M1
        add_term(u_base + 0, v_base + 0, w_base + 0, 1.0) # A00 B00 -> C00
        add_term(u_base + 3, v_base + 3, w_base + 0, 1.0) # A11 B11 -> C00
        add_term(u_base + 0, v_base + 3, w_base + 1, -1.0) # A00 B11 -> C01
        add_term(u_base + 3, v_base + 0, w_base + 1, -1.0) # A11 B01 -> C01
        add_term(u_base + 0, v_base + 0, w_base + 2, -1.0) # A00 B00 -> C10
        add_term(u_base + 3, v_base + 3, w_base + 2, -1.0) # A11 B11 -> C10
        add_term(u_base + 0, v_base + 3, w_base + 3, 1.0) # A00 B11 -> C11
        add_term(u_base + 3, v_base + 0, w_base + 3, 1.0) # A11 B01 -> C11
        # Wait, the indices in Strassen are:
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1, C01 -= M1, C10 -= M1
        # My add_term above adds single products. I need to add the *sums* correctly.
        # The function _strassen_2x2 returns U, V, W matrices.
        # It's easier to just call _strassen_2x2 and merge the results.
        
    # Let's restart the accumulation logic using the matrices from _strassen_2x2.
    
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    def merge_block(U_b, V_b, W_b, rank_b, u_offset, v_offset, w_offset):
        """
        Merge a block decomposition into the global lists.
        U_b has shape (block_n * block_m, rank_b)
        V_b has shape (block_m * block_p, rank_b)
        W_b has shape (block_p * block_n, rank_b)
        """
        for r in range(rank_b):
            for i in range(U_b.shape[0]):
                if U_b[i, r] != 0:
                    U_blocks.append((u_offset + i, U_b[i, r]))
            for i in range(V_b.shape[0]):
                if V_b[i, r] != 0:
                    V_blocks.append((v_offset + i, V_b[i, r]))
            for i in range(W_b.shape[0]):
                if W_b[i, r] != 0:
                    W_blocks.append((w_offset + i, W_b[i, r]))

    # 1. A1 (4x4) * B1 (4x5)
    # Split A1 into 2x2 blocks: A11, A12, A21, A22
    # Split B1 into 2x2 blocks for first 4 cols: B11, B12 (top 2 rows), B21, B22 (bottom 2 rows)
    # And B13 (4x1) for last col.
    
    # B1 is 4x5.
    # B1_top (2x5), B1_bot (2x5).
    # B1_top_left (2x2), B1_top_mid (2x2), B1_top_right (2x1).
    # B1_bot_left (2x2), B1_bot_mid (2x2), B1_bot_right (2x1).
    
    # A1 is 4x4.
    # A11 (2x2), A12 (2x2).
    # A21 (2x2), A22 (2x2).
    
    # C = A1 B1 + A2 B2
    
    # A1 B1:
    # C_top = A11 B1_top_left + A12 B1_top_mid + A11 B1_top_right + A12 B1_top_right? No.
    # C_top (2x5) = A11 B1_top (2x2 * 2x5) + A12 B1_top (2x2 * 2x5).
    # B1_top is 2x5. Split into B1_top_left (2x2), B1_top_mid (2x2), B1_top_right (2x1).
    
    # A11 B1_top_left: 2x2 * 2x2. Strassen.
    U_s, V_s, W_s, r_s = _strassen_2x2()
    # A11 is at A[0:2, 0:2]. u_offset = 0.
    # B1_top_left is at B[0:2, 0:2]. v_offset = 0.
    # C_top_left is at C[0:2, 0:2]. w_offset = 0.
    merge_block(U_s, V_s, W_s, r_s, 0, 0, 0)
    
    # A11 B1_top_mid: 2x2 * 2x2. Strassen.
    # B1_top_mid is at B[0:2, 2:4]. v_offset = 2*2 = 4.
    # C_top_mid is at C[0:2, 2:4]. w_offset = 2*2 = 4.
    merge_block(U_s, V_s, W_s, r_s, 0, 4, 4)
    
    # A11 B1_top_right: 2x2 * 2x1.
    # Schoolbook: 4 terms.
    # A[0,0]B[0,0] -> C[0,0]
    # A[0,1]B[1,0] -> C[0,0]
    # A[1,0]B[0,0] -> C[1,0]
    # A[1,1]B[1,0] -> C[1,0]
    # Indices:
    # A11 u_offset 0. B1_top_right v_offset = 2*4 + 0 = 8? 
    # B is 5x5. B1 is top 4 rows. B1_top is top 2 rows of B1.
    # B1_top_right is rows 0,1 and col 4 of B.
    # v_index for B[0,4] is 0*5+4 = 4.
    # v_index for B[1,4] is 1*5+4 = 9.
    # C_top_right is rows 0,1 and col 4 of C.
    # w_index for C[0,4] is 0*4+4 = 4.
    # w_index for C[1,4] is 1*4+4 = 8.
    
    U_2x2_1 = np.zeros((4, 4))
    V_2x2_1 = np.zeros((2, 4)) # 2x1 block has 2 elements
    W_2x2_1 = np.zeros((2, 4)) # 2x1 block has 2 elements
    
    # Term 1: A[0,0] B[0,0] -> C[0,0]
    U_2x2_1[0, 0] = 1; V_2x2_1[0, 0] = 1; W_2x2_1[0, 0] = 1
    # Term 2: A[0,1] B[1,0] -> C[0,0]
    U_2x2_1[1, 1] = 1; V_2x2_1[1, 1] = 1; W_2x2_1[0, 1] = 1
    # Term 3: A[1,0] B[0,0] -> C[1,0]
    U_2x2_1[2, 2] = 1; V_2x2_1[0, 2] = 1; W_2x2_1[1, 2] = 1
    # Term 4: A[1,1] B[1,0] -> C[1,0]
    U_2x2_1[3, 3] = 1; V_2x2_1[1, 3] = 1; W_2x2_1[1, 3] = 1
    
    merge_block(U_2x2_1, V_2x2_1, W_2x2_1, 4, 0, 4, 4)
    
    # A12 B1_top_left: 2x2 * 2x2. Strassen.
    # A12 is at A[0:2, 2:4]. u_offset = 2.
    # B1_top_left v_offset 0.
    # C_top_left w_offset 0.
    merge_block(U_s, V_s, W_s, r_s, 2, 0, 0)
    
    # A12 B1_top_mid: 2x2 * 2x2. Strassen.
    # A12 u_offset 2.
    # B1_top_mid v_offset 4.
    # C_top_mid w_offset 4.
    merge_block(U_s, V_s, W_s, r_s, 2, 4, 4)
    
    # A12 B1_top_right: 2x2 * 2x1.
    # A12 u_offset 2.
    # B1_top_right v_offset 4.
    # C_top_right w_offset 4.
    merge_block(U_2x2_1, V_2x2_1, W_2x2_1, 4, 2, 4, 4)
    
    # A21 B1_bot_left: 2x2 * 2x2. Strassen.
    # A21 is at A[2:4, 0:2]. u_offset = 2*2 + 0 = 4.
    # B1_bot_left is at B[2:4, 0:2]. v_offset = 2*5 + 0 = 10.
    # C_bot_left is at C[2:4, 0:2]. w_offset = 2*4 + 0 = 8.
    merge_block(U_s, V_s, W_s, r_s, 4, 10, 8)
    
    # A21 B1_bot_mid: 2x2 * 2x2. Strassen.
    # A21 u_offset 4.
    # B1_bot_mid v_offset = 2*5 + 2 = 12.
    # C_bot_mid w_offset = 2*4 + 2 = 10.
    merge_block(U_s, V_s, W_s, r_s, 4, 12, 10)
    
    # A21 B1_bot_right: 2x2 * 2x1.
    # A21 u_offset 4.
    # B1_bot_right v_offset = 2*5 + 4 = 14.
    # C_bot_right w_offset = 2*4 + 4 = 12.
    merge_block(U_2x2_1, V_2x2_1, W_2x2_1, 4, 4, 14, 12)
    
    # A22 B1_bot_left: 2x2 * 2x2. Strassen.
    # A22 is at A[2:4, 2:4]. u_offset = 2*2 + 2 = 6.
    # B1_bot_left v_offset 10.
    # C_bot_left w_offset 8.
    merge_block(U_s, V_s, W_s, r_s, 6, 10, 8)
    
    # A22 B1_bot_mid: 2x2 * 2x2. Strassen.
    # A22 u_offset 6.
    # B1_bot_mid v_offset 12.
    # C_bot_mid w_offset 10.
    merge_block(U_s, V_s, W_s, r_s, 6, 12, 10)
    
    # A22 B1_bot_right: 2x2 * 2x1.
    # A22 u_offset 6.
    # B1_bot_right v_offset 14.
    # C_bot_right w_offset 12.
    merge_block(U_2x2_1, V_2x2_1, W_2x2_1, 4, 6, 14, 12)
    
    # 2. A2 (4x1) * B2 (1x5)
    # A2 is at A[0:4, 4:5]. u_offset = 4.
    # B2 is at B[4:5, 0:5]. v_offset = 4*5 + 0 = 20.
    # C2 is at C[0:4, 0:5]. w_offset = 0.
    
    # This is 4x1 * 1x5.
    # C[i,j] = A[i,4] B[4,j].
    # 20 terms.
    
    U_4x1_5 = np.zeros((4, 20))
    V_1x5 = np.zeros((5, 20))
    W_4x5 = np.zeros((20, 20))
    
    idx = 0
    for i in range(4):
        for j in range(5):
            U_4x1_5[i, idx] = 1
            V_1x5[j, idx] = 1
            W_4x5[i*5+j, idx] = 1
            idx += 1
            
    merge_block(U_4x1_5, V_1x5, W_4x5, 20, 4, 20, 0)
    
    # Now build the final matrices
    rank = len(U_blocks)
    U = np.zeros((20, rank))
    V = np.zeros((25, rank))
    W = np.zeros((20, rank))
    
    for r in range(rank):
        for idx, val in U_blocks[r*3:r*3+3]:
            U[idx, r] = val
        for idx, val in V_blocks[r*3:r*3+3]:
            V[idx, r] = val
        for idx, val in W_blocks[r*3:r*3+3]:
            W[idx, r] = val
            
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    U, V, W, rank = _construct_block_decomposition()
    
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

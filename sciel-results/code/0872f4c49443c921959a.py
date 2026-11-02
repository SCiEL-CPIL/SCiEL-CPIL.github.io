import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def _build_strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    Indexing: U[i*2+j, c], V[i*2+j, c], W[i*2+j, c]
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # Strassen steps
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U entries (A part)
    # col 0 (M1): A00, A11
    U[0][0] = 1; U[3][0] = 1
    # col 1 (M2): A10, A11
    U[2][1] = 1; U[3][1] = 1
    # col 2 (M3): A00
    U[0][2] = 1
    # col 3 (M4): A11
    U[3][3] = 1
    # col 4 (M5): A00, A01
    U[0][4] = 1; U[1][4] = 1
    # col 5 (M6): A10, A00
    U[2][5] = 1; U[0][5] = -1
    # col 6 (M7): A01, A11
    U[1][6] = 1; U[3][6] = -1
    
    # V entries (B part)
    # col 0 (M1): B00, B11
    V[0][0] = 1; V[3][0] = 1
    # col 1 (M2): B00
    V[0][1] = 1
    # col 2 (M3): B01, B11
    V[1][2] = 1; V[3][2] = -1
    # col 3 (M4): B10, B00
    V[2][3] = 1; V[0][3] = -1
    # col 4 (M5): B11
    V[3][4] = 1
    # col 5 (M6): B00, B01
    V[0][5] = 1; V[1][5] = 1
    # col 6 (M7): B10, B11
    V[2][6] = 1; V[3][6] = 1
    
    # W entries (C part)
    # col 0 (M1): C00, C11
    W[0][0] = 1; W[3][0] = 1
    # col 1 (M2): C10, C11
    W[2][1] = 1; W[3][1] = -1
    # col 2 (M3): C01, C11
    W[1][2] = 1; W[3][2] = 1
    # col 3 (M4): C00, C10
    W[0][3] = 1; W[2][3] = 1
    # col 4 (M5): C00, C01
    W[0][4] = -1; W[1][4] = 1
    # col 5 (M6): C11
    W[3][5] = 1
    # col 6 (M7): C00
    W[0][6] = 1
    
    return U, V, W

def _build_trivial(n, m, p):
    """
    Returns U, V, W for n x m x p matrix multiplication using schoolbook.
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _combine_blocks(blocks, n, m, p):
    """
    Combines a list of block decompositions into a single decomposition for n x m x p.
    blocks: list of tuples (U_sub, V_sub, W_sub, i_off, j_off, k_off)
    i_off, j_off, k_off are the starting indices for the block in the global matrices.
    n, m, p are the global dimensions.
    """
    total_rank = sum(len(U_sub) for U_sub, V_sub, W_sub, _, _, _ in blocks)
    # Note: len(U_sub) is n_sub * m_sub? No, U_sub is (n_sub*m_sub, rank_sub).
    # So rank_sub is number of columns.
    total_rank = sum(U_sub.shape[1] if isinstance(U_sub, np.ndarray) else len(U_sub[0]) for U_sub, V_sub, W_sub, _, _, _ in blocks)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for U_sub, V_sub, W_sub, i_off, j_off, k_off in blocks:
        rank_sub = U_sub.shape[1] if isinstance(U_sub, np.ndarray) else len(U_sub[0])
        n_sub = U_sub.shape[0] if isinstance(U_sub, np.ndarray) else len(U_sub)
        # We need to infer m_sub, p_sub from V_sub, W_sub dimensions?
        # V_sub is (m_sub * p_sub, rank_sub)
        # W_sub is (p_sub * n_sub, rank_sub)
        # It's easier to just iterate over the rows of the sub-matrices and map them.
        
        for r in range(n_sub):
            # Map local row r to global row
            # Local row index in U_sub corresponds to (i_local, j_local)
            # But we don't know m_sub directly from U_sub shape alone unless we know n_sub.
            # Actually, the input `blocks` should provide enough info or we can infer.
            # Let's assume the caller provides the sub-dimensions or we can deduce them.
            # Better: Pass n_sub, m_sub, p_sub in the block tuple.
            pass
            
    # Revised approach: Pass sub-dimensions in the block tuple.
    # blocks: (U, V, W, n_sub, m_sub, p_sub, i_off, j_off, k_off)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for U_sub, V_sub, W_sub, n_sub, m_sub, p_sub, i_off, j_off, k_off in blocks:
        rank_sub = U_sub.shape[1] if isinstance(U_sub, np.ndarray) else len(U_sub[0])
        
        for c in range(rank_sub):
            for i_local in range(n_sub):
                for j_local in range(m_sub):
                    u_idx = i_local * m_sub + j_local
                    val_u = U_sub[u_idx][c]
                    if val_u != 0:
                        i_global = i_off + i_local
                        j_global = j_off + j_local
                        U[i_global * m + j_global][col_offset + c] = val_u
                        
            for j_local in range(m_sub):
                for k_local in range(p_sub):
                    v_idx = j_local * p_sub + k_local
                    val_v = V_sub[v_idx][c]
                    if val_v != 0:
                        j_global = j_off + j_local
                        k_global = k_off + k_local
                        V[j_global * p + k_global][col_offset + c] = val_v
                        
            for k_local in range(p_sub):
                for i_local in range(n_sub):
                    w_idx = k_local * n_sub + i_local
                    val_w = W_sub[w_idx][c]
                    if val_w != 0:
                        i_global = i_off + i_local
                        k_global = k_off + k_local
                        W[k_global * n + i_global][col_offset + c] = val_w
                        
        col_offset += rank_sub
        
    return U, V, W

def construct():
    start_time = time.time()
    
    # 1. Build 2x2x2 Strassen (Rank 7)
    U_222, V_222, W_222 = _build_strassen_2x2x2()
    
    # 2. Build 2x2x1 Trivial (Rank 4)
    U_221, V_221, W_221 = _build_trivial(2, 2, 1)
    
    # 3. Build 2x1x2 Trivial (Rank 4)
    U_212, V_212, W_212 = _build_trivial(2, 1, 2)
    
    # 4. Build 2x1x1 Trivial (Rank 2)
    U_211, V_211, W_211 = _build_trivial(2, 1, 1)
    
    # Combine to 2x3x3 (Rank 7+4+4+2 = 17)
    # A (2x3) split into A_L (2x2, cols 0-1) and A_R (2x1, col 2)
    # B (3x3) split into B_T (2x3, rows 0-1) and B_B (1x3, row 2)
    # B_T split into B_TL (2x2, cols 0-1) and B_TR (2x1, col 2)
    # B_B split into B_BL (1x2, cols 0-1) and B_BR (1x1, col 2)
    
    # Block 1: A_L (2x2) * B_TL (2x2) -> C_LL (2x2) at (0,0)
    # i_off=0, j_off=0, k_off=0
    block_1 = (U_222, V_222, W_222, 2, 2, 2, 0, 0, 0)
    
    # Block 2: A_L (2x2) * B_TR (2x1) -> C_LR (2x1) at (0,2)
    # i_off=0, j_off=0, k_off=2
    block_2 = (U_221, V_221, W_221, 2, 2, 1, 0, 0, 2)
    
    # Block 3: A_R (2x1) * B_BL (1x2) -> C_RL (2x2) at (2,0)
    # i_off=2, j_off=0, k_off=0
    block_3 = (U_212, V_212, W_212, 2, 1, 2, 2, 0, 0)
    
    # Block 4: A_R (2x1) * B_BR (1x1) -> C_RR (2x1) at (2,2)
    # i_off=2, j_off=0, k_off=2
    block_4 = (U_211, V_211, W_211, 2, 1, 1, 2, 0, 2)
    
    U_233, V_233, W_233 = _combine_blocks([block_1, block_2, block_3, block_4], 2, 3, 3)
    
    # Combine to 2x3x6 (Rank 17 + 17 = 34)
    # Split B (3x6) into B_L (3x3, cols 0-2) and B_R (3x3, cols 3-5)
    # Split C (2x6) into C_L (2x3, cols 0-2) and C_R (2x3, cols 3-5)
    
    # Block 5: 2x3x3 at (0,0,0)
    block_5 = (U_233, V_233, W_233, 2, 3, 3, 0, 0, 0)
    
    # Block 6: 2x3x3 at (0,3,3)
    block_6 = (U_233, V_233, W_233, 2, 3, 3, 0, 3, 3)
    
    U_236, V_236, W_236 = _combine_blocks([block_5, block_6], 2, 3, 6)
    
    # Combine to 2x2x6 (Rank 7*3 = 21)
    # Split B (2x6) into three 2x2 blocks
    # Block 7: 2x2x2 at (0,0,0)
    block_7 = (U_222, V_222, W_222, 2, 2, 2, 0, 0, 0)
    # Block 8: 2x2x2 at (0,2,2)
    block_8 = (U_222, V_222, W_222, 2, 2, 2, 0, 2, 2)
    # Block 9: 2x2x2 at (0,4,4)
    block_9 = (U_222, V_222, W_222, 2, 2, 2, 0, 4, 4)
    
    U_226, V_226, W_226 = _combine_blocks([block_7, block_8, block_9], 2, 2, 6)
    
    # Combine to 2x5x6 (Rank 34 + 21 = 55)
    # Split A (2x5) into A_T (2x3, cols 0-2) and A_B (2x2, cols 3-4)
    # Split B (5x6) into B_T (3x6, rows 0-2) and B_B (2x6, rows 3-4)
    
    # Block 10: 2x3x6 at (0,0,0)
    block_10 = (U_236, V_236, W_236, 2, 3, 6, 0, 0, 0)
    
    # Block 11: 2x2x6 at (3,3,0) -> Wait, A_B is rows 3-4 of A? No.
    # A is 2x5. i is 0..1, j is 0..4.
    # A_T uses j=0,1,2. A_B uses j=3,4.
    # So i_off=0 for A_B? No, i is the row of A. A is 2x5.
    # The split is on the middle dimension (j).
    # So A_T is A[0:2, 0:3] and A_B is A[0:2, 3:5].
    # The i_offset for both is 0.
    # The j_offset for A_T is 0. For A_B is 3.
    
    # B_T is B[0:3, 0:6]. B_B is B[3:5, 0:6].
    # i_offset for B_T is 0. For B_B is 3.
    # j_offset for both is 0.
    
    # C is 2x6. i_offset 0, k_offset 0 for both.
    
    block_10 = (U_236, V_236, W_236, 2, 3, 6, 0, 0, 0)
    block_11 = (U_226, V_226, W_226, 2, 2, 6, 0, 3, 0)
    
    U, V, W = _combine_blocks([block_10, block_11], 2, 5, 6)
    
    return {"U": U, "V": V, "W": W}

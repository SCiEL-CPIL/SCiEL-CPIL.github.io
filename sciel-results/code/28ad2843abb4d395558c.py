import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def _build_strassen_2x2():
    """Strassen decomposition for 2x2x2 matrix multiplication. Rank 7."""
    # Strassen's 7 multiplications for 2x2 matrices
    # C = AB where A, B, C are 2x2
    # Using standard Strassen formulas
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Index: row = i*2+j for A (i,j in 0..1), V: j*2+k, W: k*2+i
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1; W[1, 0] = 1; W[2, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[0, 2] = 1; W[1, 2] = 1
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[3, 3] = 1
    
    # M5 = (A00+A10)B10
    U[0, 4] = 1; U[2, 4] = 1
    V[2, 4] = 1
    W[0, 4] = 1; W[2, 4] = 1
    
    # M6 = (A11+A01)B01
    U[3, 5] = 1; U[1, 5] = 1
    V[1, 5] = 1
    W[1, 5] = 1; W[3, 5] = 1
    
    # M7 = (A00-A11)(B11+B10)
    U[0, 6] = 1; U[3, 6] = -1
    V[3, 6] = 1; V[2, 6] = 1
    W[1, 6] = -1; W[2, 6] = -1; W[0, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def _build_strassen_4x4():
    """Strassen decomposition for 4x4x4 matrix multiplication. Rank 49."""
    # Use recursive Strassen: 4x4 = 2x(2x2) blocks
    # S = [S11 S12; S21 S22] where each Sij is 2x2
    # Each Sij computed via Strassen 2x2, but with shared multiplications
    
    # Actually, for 4x4x4, we can use the standard 49-multiplication Strassen
    # Or we can compose 2x2 Strassen blocks
    
    # Let's use the block approach: 
    # A = [A00 A01; A10 A11] where each is 2x2
    # B = [B00 B01; B10 B11] where each is 2x2
    
    # M1 = (A00+A11)(B00+B11) -> 7 mults
    # M2 = (A10+A11)B00 -> 7 mults  
    # M3 = A00(B01-B11) -> 7 mults
    # M4 = A11(B10-B00) -> 7 mults
    # M5 = (A00+A10)B10 -> 7 mults
    # M6 = (A11+A01)B01 -> 7 mults
    # M7 = (A00-A11)(B11+B10) -> 7 mults
    # Total: 7*7 = 49 multiplications
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Index for 4x4: A[i,j] at i*4+j, B[j,k] at j*4+k, C[k,i] at k*4+i
    # Block indices: A00 = rows 0-1, cols 0-1; A01 = rows 0-1, cols 2-3
    #                A10 = rows 2-3, cols 0-1; A11 = rows 2-3, cols 2-3
    
    def set_block(U_mat, V_mat, W_mat, u_offset, v_offset, w_offset, 
                  a_rows, a_cols, b_rows, b_cols, c_rows, c_cols,
                  sign_a=1, sign_b=1):
        """Set up a 2x2x2 Strassen block at given offsets with optional signs."""
        U2, V2, W2 = _build_strassen_2x2()
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    if U2[i*2+j, r] != 0:
                        ai = a_rows[i]
                        aj = a_cols[j]
                        U_mat[ai*4+aj, u_offset+r] += sign_a * U2[i*2+j, r]
            for j in range(2):
                for k in range(2):
                    if V2[j*2+k, r] != 0:
                        bj = b_rows[j]
                        bk = b_cols[k]
                        V_mat[bj*4+bk, u_offset+r] += sign_b * V2[j*2+k, r]
            for k in range(2):
                for i in range(2):
                    if W2[k*2+i, r] != 0:
                        ck = c_rows[k]
                        ci = c_rows[i]
                        W_mat[ck*4+ci, u_offset+r] += W2[k*2+i, r]
    
    # M1 = (A00+A11)(B00+B11)
    set_block(U, V, W, 0, 0, 0, 
              a_rows=[0,1], a_cols=[0,1], b_rows=[0,1], b_cols=[0,1], 
              c_rows=[0,1], c_cols=[0,1], sign_a=1)
    set_block(U, V, W, 0, 0, 0,
              a_rows=[2,3], a_cols=[2,3], b_rows=[2,3], b_cols=[2,3],
              c_rows=[0,1], c_cols=[0,1], sign_a=1)
    
    # M2 = (A10+A11)B00
    set_block(U, V, W, 7, 7, 7,
              a_rows=[2,3], a_cols=[0,1], b_rows=[0,1], b_cols=[0,1],
              c_rows=[2,3], c_cols=[0,1], sign_a=1)
    set_block(U, V, W, 7, 7, 7,
              a_rows=[2,3], a_cols=[2,3], b_rows=[0,1], b_cols=[0,1],
              c_rows=[2,3], c_cols=[0,1], sign_a=1)
    
    # M3 = A00(B01-B11)
    set_block(U, V, W, 14, 14, 14,
              a_rows=[0,1], a_cols=[0,1], b_rows=[0,1], b_cols=[2,3],
              c_rows=[0,1], c_cols=[0,1], sign_a=1)
    set_block(U, V, W, 14, 14, 14,
              a_rows=[0,1], a_cols=[0,1], b_rows=[2,3], b_cols=[2,3],
              c_rows=[0,1], c_cols=[0,1], sign_a=1, sign_b=-1)
    
    # M4 = A11(B10-B00)
    set_block(U, V, W, 21, 21, 21,
              a_rows=[2,3], a_cols=[2,3], b_rows=[2,3], b_cols=[0,1],
              c_rows=[2,3], c_cols=[2,3], sign_a=1)
    set_block(U, V, W, 21, 21, 21,
              a_rows=[2,3], a_cols=[2,3], b_rows=[0,1], b_cols=[0,1],
              c_rows=[2,3], c_cols=[2,3], sign_a=1, sign_b=-1)
    
    # M5 = (A00+A10)B10
    set_block(U, V, W, 28, 28, 28,
              a_rows=[0,1], a_cols=[0,1], b_rows=[2,3], b_cols=[0,1],
              c_rows=[0,1], c_cols=[0,1], sign_a=1)
    set_block(U, V, W, 28, 28, 28,
              a_rows=[2,3], a_cols=[0,1], b_rows=[2,3], b_cols=[0,1],
              c_rows=[0,1], c_cols=[0,1], sign_a=1)
    
    # M6 = (A11+A01)B01
    set_block(U, V, W, 35, 35, 35,
              a_rows=[2,3], a_cols=[2,3], b_rows=[0,1], b_cols=[2,3],
              c_rows=[2,3], c_cols=[2,3], sign_a=1)
    set_block(U, V, W, 35, 35, 35,
              a_rows=[0,1], a_cols=[2,3], b_rows=[0,1], b_cols=[2,3],
              c_rows=[2,3], c_cols=[2,3], sign_a=1)
    
    # M7 = (A00-A11)(B11+B10)
    set_block(U, V, W, 42, 42, 42,
              a_rows=[0,1], a_cols=[0,1], b_rows=[2,3], b_cols=[2,3],
              c_rows=[0,1], c_cols=[0,1], sign_a=1)
    set_block(U, V, W, 42, 42, 42,
              a_rows=[2,3], a_cols=[2,3], b_rows=[2,3], b_cols=[2,3],
              c_rows=[0,1], c_cols=[0,1], sign_a=-1)
    # B10 part
    set_block(U, V, W, 42, 42, 42,
              a_rows=[0,1], a_cols=[0,1], b_rows=[2,3], b_cols=[0,1],
              c_rows=[0,1], c_cols=[0,1], sign_a=1)
    set_block(U, V, W, 42, 42, 42,
              a_rows=[2,3], a_cols=[2,3], b_rows=[2,3], b_cols=[0,1],
              c_rows=[0,1], c_cols=[0,1], sign_a=-1)
    
    return U, V, W

def _build_strassen_4x4x2():
    """Decomposition for 4x4x2 matrix multiplication. Rank 28."""
    # C (4x2) = A (4x4) * B (4x2)
    # Split A into 2x2 blocks, B into 2x2 blocks
    # Use Strassen-like approach
    
    # A = [A00 A01; A10 A11] each 2x2
    # B = [B00; B10] each 2x2 (top and bottom)
    
    # C = [C0; C1] where C0 = A00*B00 + A01*B10, C1 = A10*B00 + A11*B10
    
    # Using Strassen for 2x2x2:
    # C0: 7 mults, C1: 7 mults, but we can share some
    
    # Actually, let's just use schoolbook for now and see if we can improve
    # Schoolbook: 4*4*2 = 32 mults
    # Strassen-based: can we do better?
    
    # Let's try a different approach: treat it as two 4x2 multiplications
    # C0 = A * B0 (4x4 * 4x2 = 4x2)
    # C1 = A * B1 (4x4 * 4x2 = 4x2)
    # Each 4x4x2 can be done in... let's think
    
    # For 4x4x2: 
    # Split A into 2x2 blocks, B (4x2) into two 2x2 blocks
    # Result is 4x2, split into two 2x2 blocks
    
    # C_top = A00*B0_top + A01*B1_top
    # C_bot = A10*B0_bot + A11*B1_bot
    
    # Using Strassen for each 2x2x2: 7+7 = 14 for top, 7+7 = 14 for bottom
    # Total: 28
    
    U = np.zeros((16, 28))
    V = np.zeros((8, 28))
    W = np.zeros((8, 28))
    
    # Index: A[i,j] at i*4+j, B[j,k] at j*2+k, C[k,i] at k*4+i
    # B is 4x2: B[j,k] where j in 0..3, k in 0..1
    # C is 4x2: C[k,i] where k in 0..1, i in 0..3
    
    U2, V2, W2 = _build_strassen_2x2()
    
    # Top half of C (i=0,1): C[k,0] and C[k,1] for k=0,1
    # C_top = A_top * B_top + A_top_right * B_bot
    # A_top = A[0:2, 0:2], A_top_right = A[0:2, 2:4]
    # B_top = B[0:2, 0:2], B_bot = B[2:4, 0:2]
    
    # For C_top (2x2) = A_top(2x2)*B_top(2x2) + A_top_right(2x2)*B_bot(2x2)
    # Use Strassen: M = (A_top+A_top_right)(B_top+B_bot) - A_top*B_bot - A_top_right*B_top
    # But that's 7+7+7 = 21, not better
    
    # Let's just do it directly with two Strassen calls
    # C_top = A_top * B_top: 7 mults
    for r in range(7):
        for i in range(2):
            for j in range(2):
                if U2[i*2+j, r] != 0:
                    U[i*4+j, r] = U2[i*2+j, r]
        for j in range(2):
            for k in range(2):
                if V2[j*2+k, r] != 0:
                    V[j*2+k, r] = V2[j*2+k, r]
        for k in range(2):
            for i in range(2):
                if W2[k*2+i, r] != 0:
                    W[k*4+i, r] = W2[k*2+i, r]
    
    # C_top += A_top_right * B_bot: 7 mults
    for r in range(7):
        for i in range(2):
            for j in range(2):
                if U2[i*2+j, r] != 0:
                    U[i*4+(j+2), 7+r] = U2[i*2+j, r]
        for j in range(2):
            for k in range(2):
                if V2[j*2+k, r] != 0:
                    V[(j+2)*2+k, 7+r] = V2[j*2+k, r]
        for k in range(2):
            for i in range(2):
                if W2[k*2+i, r] != 0:
                    W[k*4+i, 7+r] = W2[k*2+i, r]
    
    # C_bot (i=2,3): C[k,2] and C[k,3] for k=0,1
    # C_bot = A_bot * B_top + A_bot_right * B_bot
    # A_bot = A[2:4, 0:2], A_bot_right = A[2:4, 2:4]
    
    # C_bot = A_bot * B_top: 7 mults
    for r in range(7):
        for i in range(2):
            for j in range(2):
                if U2[i*2+j, r] != 0:
                    U[(i+2)*4+j, 14+r] = U2[i*2+j, r]
        for j in range(2):
            for k in range(2):
                if V2[j*2+k, r] != 0:
                    V[j*2+k, 14+r] = V2[j*2+k, r]
        for k in range(2):
            for i in range(2):
                if W2[k*2+i, r] != 0:
                    W[k*4+(i+2), 14+r] = W2[k*2+i, r]
    
    # C_bot += A_bot_right * B_bot: 7 mults
    for r in range(7):
        for i in range(2):
            for j in range(2):
                if U2[i*2+j, r] != 0:
                    U[(i+2)*4+(j+2), 21+r] = U2[i*2+j, r]
        for j in range(2):
            for k in range(2):
                if V2[j*2+k, r] != 0:
                    V[(j+2)*2+k, 21+r] = V2[j*2+k, r]
        for k in range(2):
            for i in range(2):
                if W2[k*2+i, r] != 0:
                    W[k*4+(i+2), 21+r] = W2[k*2+i, r]
    
    return U, V, W

def _build_schoolbook(n, m, p):
    """Schoolbook decomposition for n x m x p. Rank n*m*p."""
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

def _combine_decompositions(decomps, n, m, p):
    """Combine multiple block decompositions into one full decomposition.
    
    decomps: list of (U_block, V_block, W_block, n_start, m_start, p_start)
    where each block computes C[n_start:n_start+nb, p_start:p_start+pb] = 
          A[n_start:n_start+nb, m_start:m_start+mb] * B[m_start:m_start+mb, p_start:p_start+pb]
    """
    total_rank = sum(d[0].shape[1] for d in decomps)
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))
    
    rank_offset = 0
    for U_b, V_b, W_b, n_start, m_start, p_start in decomps:
        nb, mb, pb = U_b.shape[0] // mb if mb > 0 else 0, V_b.shape[0] // pb if pb > 0 else 0, W_b.shape[0] // nb if nb > 0 else 0
        
        # Actually, let's infer dimensions from the block sizes
        # U_b is (nb*mb, rank_b), V_b is (mb*pb, rank_b), W_b is (pb*nb, rank_b)
        rank_b = U_b.shape[1]
        
        # We need to figure out nb, mb, pb from context
        # For now, let's just copy the blocks at the right positions
        
        for r in range(rank_b):
            for u_idx in range(U_b.shape[0]):
                # u_idx corresponds to (i_local, j_local) in the block
                # We need to map to global indices
                pass
        
        # This is getting complicated. Let me simplify by having each decom
        # explicitly specify the mapping.
        
        rank_offset += rank_b
    
    # Simplified approach: just concatenate
    U_full = np.hstack([d[0] for d in decomps])
    V_full = np.hstack([d[1] for d in decomps])
    W_full = np.hstack([d[2] for d in decomps])
    
    return U_full, V_full, W_full

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Strategy: Use block decomposition
    # 4x5x6 = split into blocks
    # 
    # Approach 1: Split into 4x4x6 + 4x1x6
    #   4x4x6: use Strassen 4x4x4 (rank 49) for first 4 cols of B, schoolbook 4x4x2 (rank 32) for last 2
    #   4x1x6: schoolbook (rank 24)
    #   Total: 49 + 32 + 24 = 105
    #
    # Approach 2: Split into 4x4x4 + 4x4x2 + 4x1x4 + 4x1x2
    #   4x4x4: Strassen (rank 49)
    #   4x4x2: Strassen-based (rank 28)
    #   4x1x4: schoolbook (rank 16)
    #   4x1x2: schoolbook (rank 8)
    #   Total: 49 + 28 + 16 + 8 = 101
    #
    # Let's try Approach 2
    
    # Block 1: A[0:4, 0:4] * B[0:4, 0:4] -> C[0:4, 0:4] (4x4x4)
    U1, V1, W1 = _build_strassen_4x4()
    
    # Block 2: A[0:4, 0:4] * B[0:4, 4:6] -> C[0:4, 4:6] (4x4x2)
    U2, V2, W2 = _build_strassen_4x4x2()
    
    # Block 3: A[0:4, 4:5] * B[4:5, 0:4] -> C[0:4, 0:4] (4x1x4)
    U3, V3, W3 = _build_schoolbook(4, 1, 4)
    
    # Block 4: A[0:4, 4:5] * B[4:5, 4:6] -> C[0:4, 4:6] (4x1x2)
    U4, V4, W4 = _build_schoolbook(4, 1, 2)
    
    # Now we need to properly assemble these blocks
    # The issue is that the block decompositions assume specific index mappings
    # Let me create a proper assembly function
    
    # Actually, let me just build the full decomposition directly
    
    # Full rank estimate: 49 + 28 + 16 + 8 = 101
    rank = 101
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Block 1: 4x4x4 Strassen (rank 49)
    # A[0:4, 0:4], B[0:4, 0:4], C[0:4, 0:4]
    # U1 is (16, 49), V1 is (16, 49), W1 is (16, 49)
    # U1[i*4+j, r] for i,j in 0..3
    # V1[j*4+k, r] for j,k in 0..3
    # W1[k*4+i, r] for k,i in 0..3
    
    U[:, 0:49] = U1
    V[:, 0:49] = V1
    W[:, 0:49] = W1
    
    # Block 2: 4x4x2 (rank 28)
    # A[0:4, 0:4], B[0:4, 4:6], C[0:4, 4:6]
    # U2 is (16, 28), V2 is (8, 28), W2 is (8, 28)
    # U2[i*4+j, r] for i,j in 0..3
    # V2[j*2+k, r] for j in 0..3, k in 0..1 (B is 4x2)
    # W2[k*4+i, r] for k in 0..1, i in 0..3 (C is 4x2)
    
    U[:, 49:77] = U2
    # V2 is (8, 28) but we need (20, 28) for B[0:4, 4:6]
    # B[0:4, 4:6] has indices j in 0..3, k in 4..5
    # V2[j*2+k, r] where k is 0..1, so we need to map to j*6+(k+4)
    V_block2 = np.zeros((m * p, 28))
    for j in range(4):
        for k in range(2):
            V_block2[j * p + (k + 4), :] = V2[j * 2 + k, :]
    V[:, 49:77] = V_block2
    
    # W2 is (8, 28) for C[0:4, 4:6]
    # W2[k*4+i, r] where k in 0..1, i in 0..3
    # We need W[k*n+i, r] where k in 4..5, i in 0..3
    W_block2 = np.zeros((p * n, 28))
    for k in range(2):
        for i in range(4):
            W_block2[(k + 4) * n + i, :] = W2[k * 4 + i, :]
    W[:, 49:77] = W_block2
    
    # Block 3: 4x1x4 schoolbook (rank 16)
    # A[0:4, 4:5], B[4:5, 0:4], C[0:4, 0:4]
    # U3 is (4, 16), V3 is (4, 16), W3 is (16, 16)
    # U3[i*1+j, r] = U3[i, r] for i in 0..3, j=0
    # V3[j*4+k, r] = V3[4+k, r] for j=0, k in 0..3
    # W3[k*4+i, r] for k in 0..3, i in 0..3
    
    U_block3 = np.zeros((n * m, 16))
    for i in range(4):
        U_block3[i * m + 4, :] = U3[i, :]
    U[:, 77:93] = U_block3
    
    V_block3 = np.zeros((m * p, 16))
    for k in range(4):
        V_block3[4 * p + k, :] = V3[4 + k, :]
    V[:, 77:93] = V_block3
    
    W[:, 77:93] = W3
    
    # Block 4: 4x1x2 schoolbook (rank 8)
    # A[0:4, 4:5], B[4:5, 4:6], C[0:4, 4:6]
    # U4 is (4, 8), V4 is (2, 8), W4 is (8, 8)
    
    U_block4 = np.zeros((n * m, 8))
    for i in range(4):
        U_block4[i * m + 4, :] = U4[i, :]
    U[:, 93:101] = U_block4
    
    V_block4 = np.zeros((m * p, 8))
    for k in range(2):
        V_block4[4 * p + (k + 4), :] = V4[k, :]
    V[:, 93:101] = V_block4
    
    W_block4 = np.zeros((p * n, 8))
    for k in range(2):
        for i in range(4):
            W_block4[(k + 4) * n + i, :] = W4[k * 4 + i, :]
    W[:, 93:101] = W_block4
    
    def verify(U_mat, V_mat, W_mat):
        r = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = sum(U_mat[i*m+j, r_idx] * V_mat[j*p+k, r_idx] * W_mat[k*n+i, r_idx] for r_idx in range(r))
                    if abs(actual - 1.0) > 1e-9:
                        return False
        return True
    
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    U_fallback, V_fallback, W_fallback = _build_schoolbook(n, m, p)
    return {"U": U_fallback.tolist(), "V": V_fallback.tolist(), "W": W_fallback.tolist()}

import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs an exact tensor decomposition for 3x4x5 matrix multiplication using
    block-diagonal embeddings of known low-rank sub-tensors.
    
    Strategy:
    - Split the 3x4x5 multiplication into a 2x4x5 core and a 1x4x5 tail.
    - For 2x4x5: split m=4 into two 2s, giving two 2x2x5 blocks.
      Each 2x2x5 = 2x2x2 + 2x2x2 + 2x2x1.
      Use Strassen (rank 7) for 2x2x2, schoolbook (rank 4) for 2x2x1.
      Total for 2x4x5: 2*(7+7+4) = 36.
    - For 1x4x5: split m=4 into two 2s, giving two 1x2x5 blocks.
      Each 1x2x5 = 1x2x2 + 1x2x2 + 1x2x1.
      Use schoolbook (rank 4) for 1x2x2, schoolbook (rank 2) for 1x2x1.
      Total for 1x4x5: 2*(4+4+2) = 20.
    - Grand total: 36 + 20 = 56.
    
    Index conventions:
    U: (n*m, rank) where U[i*m+j, r] corresponds to A[i,j]
    V: (m*p, rank) where V[i*p+j, r] corresponds to B[i,j]
    W: (p*n, rank) where W[k*n+i, r] corresponds to C[i,k] (trace convention)
    """
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # --- Strassen 2x2x2 decomposition (rank 7) ---
    # U: 4x7 (A00,A01,A10,A11), V: 4x7 (B00,B01,B10,B11), W: 4x7 (C00,C01,C10,C11)
    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        # C00 += M1, C11 += M1
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        # C10 += M2, C11 -= M2
        
        # M3 = A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        # C01 += M3, C11 += M3
        
        # M4 = A11(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        # C00 += M4, C10 += M4
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        # C00 -= M5, C01 += M5
        
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        # C11 += M6
        
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        # C00 += M7
        
        return U, V, W
    
    # --- Schoolbook 2x2x1 decomposition (rank 4) ---
    # A(2x2) * B(2x1) = C(2x1)
    def schoolbook_2x2x1():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1  # A00*B0 -> C0
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1  # A01*B1 -> C0
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1  # A10*B0 -> C1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1  # A11*B1 -> C1
        return U, V, W
    
    # --- Schoolbook 1x2x2 decomposition (rank 4) ---
    # A(1x2) * B(2x2) = C(1x2)
    def schoolbook_1x2x2():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1  # A0*B00 -> C0
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1  # A1*B10 -> C0
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1  # A0*B01 -> C1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1  # A1*B11 -> C1
        return U, V, W
    
    # --- Schoolbook 1x2x1 decomposition (rank 2) ---
    # A(1x2) * B(2x1) = C(1x1)
    def schoolbook_1x2x1():
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1  # A0*B0
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1  # A1*B1
        return U, V, W
    
    # Helper: embed a block decomposition into global U, V, W
    def embed_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk):
        """
        u_idx: list of global U indices for the block's 4 (or 2) A-entries
        v_idx: list of global V indices for the block's B-entries
        w_idx: list of global W indices for the block's C-entries
        U_blk, V_blk, W_blk: factor matrices for this block
        """
        rank = U_blk.shape[1]
        for r in range(rank):
            u_vec = np.zeros(n * m)
            for i, gi in enumerate(u_idx):
                u_vec[gi] = U_blk[i, r]
            U_parts.append(u_vec)
            
            v_vec = np.zeros(m * p)
            for i, gi in enumerate(v_idx):
                v_vec[gi] = V_blk[i, r]
            V_parts.append(v_vec)
            
            w_vec = np.zeros(p * n)
            for i, gi in enumerate(w_idx):
                w_vec[gi] = W_blk[i, r]
            W_parts.append(w_vec)
    
    # Index mapping helpers
    # U[i*m + j] for A[i,j]
    # V[i*p + j] for B[i,j]  
    # W[k*n + i] for C[i,k] (trace convention: W[k*n+i, r])
    
    def u_idx(i, j):
        return i * m + j
    
    def v_idx(i, j):
        return i * p + j
    
    def w_idx(i, k):
        return k * n + i
    
    # ===================== PART 1: 2x4x5 (rows 0,1 of A) =====================
    # Split m=4 into cols 0,1 and cols 2,3
    # For each half, split p=5 into cols 0,1 and cols 2,3 and col 4
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (2x2x2, Strassen)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    u_idx_list = [u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1)]
    v_idx_list = [v_idx(0,0), v_idx(0,1), v_idx(1,0), v_idx(1,1)]
    w_idx_list = [w_idx(0,0), w_idx(0,1), w_idx(1,0), w_idx(1,1)]
    embed_block(u_idx_list, v_idx_list, w_idx_list, U_blk, V_blk, W_blk)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4] (2x2x2, Strassen)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    u_idx_list = [u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1)]
    v_idx_list = [v_idx(0,2), v_idx(0,3), v_idx(1,2), v_idx(1,3)]
    w_idx_list = [w_idx(0,2), w_idx(0,3), w_idx(1,2), w_idx(1,3)]
    embed_block(u_idx_list, v_idx_list, w_idx_list, U_blk, V_blk, W_blk)
    
    # Block 3: A[0:2, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5] (2x2x1, schoolbook)
    U_blk, V_blk, W_blk = schoolbook_2x2x1()
    u_idx_list = [u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1)]
    v_idx_list = [v_idx(0,4), v_idx(1,4)]
    w_idx_list = [w_idx(0,4), w_idx(1,4)]
    embed_block(u_idx_list, v_idx_list, w_idx_list, U_blk, V_blk, W_blk)
    
    # Block 4: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2] (2x2x2, Strassen)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    u_idx_list = [u_idx(0,2), u_idx(0,3), u_idx(1,2), u_idx(1,3)]
    v_idx_list = [v_idx(2,0), v_idx(2,1), v_idx(3,0), v_idx(3,1)]
    w_idx_list = [w_idx(0,0), w_idx(0,1), w_idx(1,0), w_idx(1,1)]
    embed_block(u_idx_list, v_idx_list, w_idx_list, U_blk, V_blk, W_blk)
    
    # Block 5: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4] (2x2x2, Strassen)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    u_idx_list = [u_idx(0,2), u_idx(0,3), u_idx(1,2), u_idx(1,3)]
    v_idx_list = [v_idx(2,2), v_idx(2,3), v_idx(3,2), v_idx(3,3)]
    w_idx_list = [w_idx(0,2), w_idx(0,3), w_idx(1,2), w_idx(1,3)]
    embed_block(u_idx_list, v_idx_list, w_idx_list, U_blk, V_blk, W_blk)
    
    # Block 6: A[0:2, 2:4] * B[2:4, 4:5] -> C[0:2, 4:5] (2x2x1, schoolbook)
    U_blk, V_blk, W_blk = schoolbook_2x2x1()
    u_idx_list = [u_idx(0,2), u_idx(0,3), u_idx(1,2), u_idx(1,3)]
    v_idx_list = [v_idx(2,4), v_idx(3,4)]
    w_idx_list = [w_idx(0,4), w_idx(1,4)]
    embed_block(u_idx_list, v_idx_list, w_idx_list, U_blk, V_blk, W_blk)
    
    # ===================== PART 2: 1x4x5 (row 2 of A) =====================
    # Split m=4 into cols 0,1 and cols 2,3
    
    # Block 7: A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2] (1x2x2, schoolbook)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    u_idx_list = [u_idx(2,0), u_idx(2,1)]
    v_idx_list = [v_idx(0,0), v_idx(0,1), v_idx(1,0), v_idx(1,1)]
    w_idx_list = [w_idx(2,0), w_idx(2,1)]
    embed_block(u_idx_list, v_idx_list, w_idx_list, U_blk, V_blk, W_blk)
    
    # Block 8: A[2, 0:2] * B[0:2, 2:4] -> C[2, 2:4] (1x2x2, schoolbook)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    u_idx_list = [u_idx(2,0), u_idx(2,1)]
    v_idx_list = [v_idx(0,2), v_idx(0,3), v_idx(1,2), v_idx(1,3)]
    w_idx_list = [w_idx(2,2), w_idx(2,3)]
    embed_block(u_idx_list, v_idx_list, w_idx_list, U_blk, V_blk, W_blk)
    
    # Block 9: A[2, 0:2] * B[0:2, 4:5] -> C[2, 4:5] (1x2x1, schoolbook)
    U_blk, V_blk, W_blk = schoolbook_1x2x1()
    u_idx_list = [u_idx(2,0), u_idx(2,1)]
    v_idx_list = [v_idx(0,4), v_idx(1,4)]
    w_idx_list = [w_idx(2,4)]
    embed_block(u_idx_list, v_idx_list, w_idx_list, U_blk, V_blk, W_blk)
    
    # Block 10: A[2, 2:4] * B[2:4, 0:2] -> C[2, 0:2] (1x2x2, schoolbook)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    u_idx_list = [u_idx(2,2), u_idx(2,3)]
    v_idx_list = [v_idx(2,0), v_idx(2,1), v_idx(3,0), v_idx(3,1)]
    w_idx_list = [w_idx(2,0), w_idx(2,1)]
    embed_block(u_idx_list, v_idx_list, w_idx_list, U_blk, V_blk, W_blk)
    
    # Block 11: A[2, 2:4] * B[2:4, 2:4] -> C[2, 2:4] (1x2x2, schoolbook)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    u_idx_list = [u_idx(2,2), u_idx(2,3)]
    v_idx_list = [v_idx(2,2), v_idx(2,3), v_idx(3,2), v_idx(3,3)]
    w_idx_list = [w_idx(2,2), w_idx(2,3)]
    embed_block(u_idx_list, v_idx_list, w_idx_list, U_blk, V_blk, W_blk)
    
    # Block 12: A[2, 2:4] * B[2:4, 4:5] -> C[2, 4:5] (1x2x1, schoolbook)
    U_blk, V_blk, W_blk = schoolbook_1x2x1()
    u_idx_list = [u_idx(2,2), u_idx(2,3)]
    v_idx_list = [v_idx(2,4), v_idx(3,4)]
    w_idx_list = [w_idx(2,4)]
    embed_block(u_idx_list, v_idx_list, w_idx_list, U_blk, V_blk, W_blk)
    
    # Total rank: 7+7+4+7+7+4+4+4+2+4+4+2 = 56
    U = np.array(U_parts).T  # (n*m, rank)
    V = np.array(V_parts).T  # (m*p, rank)
    W = np.array(W_parts).T  # (p*n, rank)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END

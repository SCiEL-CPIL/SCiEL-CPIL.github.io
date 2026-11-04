"""Tensor decomposition for 4x4 * 4x7 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    def make_schoolbook():
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

    def strassen_2x2x2():
        """Strassen's 7-product decomposition for 2x2 * 2x2 = 2x2.
        Returns U (4x7), V (4x7), W (4x7)."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # A: A11=0, A12=1, A21=2, A22=3
        # B: B11=0, B12=1, B21=2, B22=3
        # C: C11=0, C12=1, C21=2, C22=3
        
        # M1 = (A11+A22)*(B11+B22)
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[3,0]=1
        W[0,0]=1
        
        # M2 = (A21+A22)*B11
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        W[2,1]=1; W[3,1]=1
        
        # M3 = A11*(B12-B22)
        U[0,2]=1
        V[1,2]=1; V[3,2]=-1
        W[1,2]=1; W[3,2]=1
        
        # M4 = A22*(B21-B11)
        U[3,3]=1
        V[2,3]=1; V[0,3]=-1
        W[0,3]=1; W[2,3]=1
        
        # M5 = (A11+A12)*B22
        U[0,4]=1; U[1,4]=1
        V[3,4]=1
        W[0,4]=-1; W[1,4]=1
        
        # M6 = (A21-A11)*(B11+B12)
        U[2,5]=1; U[0,5]=-1
        V[0,5]=1; V[1,5]=1
        W[3,5]=1
        
        # M7 = (A12-A22)*(B21+B22)
        U[1,6]=1; U[3,6]=-1
        V[2,6]=1; V[3,6]=1
        W[0,6]=1
        
        return U, V, W

    def schoolbook_2x2x1():
        """Schoolbook 2x2 * 2x1 = 2x1. Rank 4.
        Returns U (4x4), V (2x4), W (2x4)."""
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        for i in range(2):
            for j in range(2):
                col = i*2 + j
                U[i*2+j, col] = 1
                V[j, col] = 1
                W[i, col] = 1
        return U, V, W

    def schoolbook_2x2x2():
        """Schoolbook 2x2 * 2x2 = 2x2. Rank 4.
        Returns U (4x4), V (4x4), W (4x4)."""
        U = np.zeros((4, 4))
        V = np.zeros((4, 4))
        W = np.zeros((4, 4))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    col = i*4 + j*2 + k
                    U[i*2+j, col] = 1
                    V[j*2+k, col] = 1
                    W[k*2+i, col] = 1
        return U, V, W

    def decompose_2x2x2p(p_val):
        """Decompose 2x2 * 2xp. 
        Split p into groups of 2 plus remainder 1 if odd.
        For each group of 2: use Strassen (rank 7).
        For remainder 1: use schoolbook 2x2x1 (rank 4).
        Returns U (4xR), V (2p x R), W (2p x R) where W has shape (2p, R).
        Actually W should have shape (p*2, R) for the 2x2 output.
        Wait: 2x2 * 2xp gives 2xp output. So W has shape (2p, R).
        U has shape (4, R), V has shape (2p, R), W has shape (2p, R).
        """
        # Number of groups of 2
        n_groups_2 = p_val // 2
        has_remainder = (p_val % 2 == 1)
        
        total_rank = n_groups_2 * 7 + (4 if has_remainder else 0)
        
        U = np.zeros((4, total_rank))
        V = np.zeros((2 * p_val, total_rank))
        W = np.zeros((2 * p_val, total_rank))
        
        col = 0
        # Process groups of 2 columns
        for g in range(n_groups_2):
            U_g, V_g, W_g = strassen_2x2x2()
            U[:, col:col+7] = U_g
            # V_g has shape (4, 7) for 2x2 input. We need to map to columns 2*g, 2*g+1
            V[2*g, col:col+7] = V_g[0, :]
            V[2*g+1, col:col+7] = V_g[1, :]
            V[2*g+2, col:col+7] = V_g[2, :]
            V[2*g+3, col:col+7] = V_g[3, :]
            # W_g has shape (4, 7) for 2x2 output. Map to rows 2*g, 2*g+1
            W[2*g, col:col+7] = W_g[0, :]
            W[2*g+1, col:col+7] = W_g[1, :]
            W[2*g+2, col:col+7] = W_g[2, :]
            W[2*g+3, col:col+7] = W_g[3, :]
            col += 7
        
        if has_remainder:
            U_r, V_r, W_r = schoolbook_2x2x1()
            U[:, col:col+4] = U_r
            # V_r has shape (2, 4) for 2x1 input. Column index 2*n_groups_2
            V[2*n_groups_2, col:col+4] = V_r[0, :]
            W[2*n_groups_2, col:col+4] = W_r[0, :]
            col += 4
        
        return U, V, W, total_rank

    def decompose_4x4x7():
        """Decompose 4x4 * 4x7 using block structure.
        Split A into four 2x2 blocks, B into four 2x7 blocks.
        C11 = A11*B1 + A12*B2  (2x7 output)
        C21 = A21*B1 + A22*B2  (2x7 output)
        
        Wait, B is 4x7. Split B into top 2 rows (B1: 2x7) and bottom 2 rows (B2: 2x7).
        A11, A12, A21, A22 are 2x2.
        
        C top 2 rows = A11 * B1 + A12 * B2  (each term is 2x7, sum is 2x7)
        C bottom 2 rows = A21 * B1 + A22 * B2  (each term is 2x7, sum is 2x7)
        
        So we need 4 decompositions of 2x2 * 2x7, each with rank R_2x2x7.
        Total rank = 4 * R_2x2x7.
        
        For 2x2 * 2x7: 7 = 2*3 + 1, so 3 groups of 2 (rank 7 each) + 1 remainder (rank 4) = 25.
        Total rank = 4 * 25 = 100.
        
        But wait, schoolbook is 4*4*7 = 112. So 100 is an improvement.
        """
        U_2x2_2x7, V_2x2_2x7, W_2x2_2x7, r_2x2x7 = decompose_2x2x2p(7)
        
        total_rank = 4 * r_2x2x7
        U = np.zeros((16, total_rank))
        V = np.zeros((28, total_rank))
        W = np.zeros((28, total_rank))
        
        # A blocks:
        # A11: rows 0,1 of A -> indices 0,1,4,5 in 4x4 row-major (i*4+j)
        # A12: rows 0,1 of A -> indices 2,3,6,7
        # A21: rows 2,3 of A -> indices 8,9,12,13
        # A22: rows 2,3 of A -> indices 10,11,14,15
        
        # B blocks (4x7, row-major j*7+k):
        # B1 (top 2 rows): indices 0..13 (j=0,1)
        # B2 (bottom 2 rows): indices 14..27 (j=2,3)
        
        # C output (4x7, row-major k*4+i):
        # C top 2 rows (i=0,1): indices 0..23 where for each k, i=0,1 -> k*4+0, k*4+1
        # C bottom 2 rows (i=2,3): indices where i=2,3 -> k*4+2, k*4+3
        
        # Block 1: A11 * B1 -> C top
        a11_rows = [0, 1, 4, 5]
        b1_rows = list(range(0, 14))  # 0..13
        c_top_rows = [k*4 + i for k in range(7) for i in range(2)]  # 0,1,4,5,8,9,12,13,16,17,20,21,24,25
        
        off = 0
        for idx_a, idx in enumerate(a11_rows):
            U[idx, off:off+r_2x2x7] = U_2x2_2x7[idx_a, :]
        for idx_v, idx in enumerate(b1_rows):
            V[idx, off:off+r_2x2x7] = V_2x2_2x7[idx_v, :]
        for idx_w, idx in enumerate(c_top_rows):
            W[idx, off:off+r_2x2x7] = W_2x2_2x7[idx_w, :]
        off += r_2x2x7
        
        # Block 2: A12 * B2 -> C top
        a12_rows = [2, 3, 6, 7]
        b2_rows = list(range(14, 28))  # 14..27
        off += 0  # already at correct offset
        
        for idx_a, idx in enumerate(a12_rows):
            U[idx, off:off+r_2x2x7] = U_2x2_2x7[idx_a, :]
        for idx_v, idx in enumerate(b2_rows):
            V[idx, off:off+r_2x2x7] = V_2x2_2x7[idx_v, :]
        for idx_w, idx in enumerate(c_top_rows):
            W[idx, off:off+r_2x2x7] = W_2x2_2x7[idx_w, :]
        off += r_2x2x7
        
        # Block 3: A21 * B1 -> C bottom
        a21_rows = [8, 9, 12, 13]
        c_bot_rows = [k*4 + i for k in range(7) for i in range(2, 4)]  # 2,3,6,7,10,11,14,15,18,19,22,23,26,27
        
        for idx_a, idx in enumerate(a21_rows):
            U[idx, off:off+r_2x2x7] = U_2x2_2x7[idx_a, :]
        for idx_v, idx in enumerate(b1_rows):
            V[idx, off:off+r_2x2x7] = V_2x2_2x7[idx_v, :]
        for idx_w, idx in enumerate(c_bot_rows):
            W[idx, off:off+r_2x2x7] = W_2x2_2x7[idx_w, :]
        off += r_2x2x7
        
        # Block 4: A22 * B2 -> C bottom
        a22_rows = [10, 11, 14, 15]
        
        for idx_a, idx in enumerate(a22_rows):
            U[idx, off:off+r_2x2x7] = U_2x2_2x7[idx_a, :]
        for idx_v, idx in enumerate(b2_rows):
            V[idx, off:off+r_2x2x7] = V_2x2_2x7[idx_v, :]
        for idx_w, idx in enumerate(c_bot_rows):
            W[idx, off:off+r_2x2x7] = W_2x2_2x7[idx_w, :]
        off += r_2x2x7
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, total_rank

    def verify(U, V, W):
        """Verify the decomposition. Returns max absolute error."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        # Build index arrays
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        u_idx = i_idx * m + j_idx
        v_idx = j_idx * p + k_idx
        w_idx = k_idx * n + i_idx
        
        u_vals = Ua[u_idx, :]
        v_vals = Va[v_idx, :]
        w_vals = Wa[w_idx, :]
        
        prod = u_vals * v_vals * w_vals
        result = prod.sum(axis=1)
        
        return float(np.max(np.abs(result - 1.0)))

    # Try the block decomposition
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    try:
        result, rank = decompose_4x4x7()
        err = verify(result["U"], result["V"], result["W"])
        if err < 1e-9:
            best_result = result
            best_rank = rank
    except Exception as e:
        pass
    
    if best_result is None:
        best_result = make_schoolbook()
    
    return best_result
# EVOLVE-BLOCK-END

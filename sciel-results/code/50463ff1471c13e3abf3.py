import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2 * 2x2 multiplication with rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen's 7 products
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # U corresponds to A coefficients
    # V corresponds to B coefficients
    # W corresponds to C coefficients
    
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # M1
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    W[0, 0] += 1; W[1, 0] += 1; W[2, 0] += 1; W[3, 0] += 1
    
    # M2
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    W[2, 1] += 1; W[3, 1] += 1
    
    # M3
    U[0, 2] += 1
    V[2, 2] += 1; V[3, 2] -= 1
    W[0, 2] += 1; W[1, 2] += 1
    
    # M4
    U[3, 3] += 1
    V[1, 3] += 1; V[0, 3] -= 1
    W[1, 3] += 1; W[3, 3] += 1
    
    # M5
    U[0, 4] += 1; U[1, 4] += 1
    V[3, 4] += 1
    W[0, 4] += 1; W[3, 4] += 1
    
    # M6
    U[2, 5] += 1; U[0, 5] -= 1
    V[0, 5] += 1; V[1, 5] += 1
    W[2, 5] += 1; W[3, 5] += 1
    
    # M7
    U[1, 6] += 1; U[3, 6] -= 1
    V[2, 7] += 1; V[3, 7] += 1 # Wait, index 6 is the 7th column.
    # Correcting indices for M7
    U[1, 6] += 1; U[3, 6] -= 1
    V[2, 6] += 1; V[3, 6] += 1
    W[0, 6] += 1; W[2, 6] += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition block by block.
    # Total rank is estimated to be 100.
    # We'll use a list to store the columns of U, V, W.
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_block(u_block, v_block, w_block, i_offset, j_offset, k_offset):
        """
        u_block: (n_sub * m_sub, rank_sub)
        v_block: (m_sub * p_sub, rank_sub)
        w_block: (p_sub * n_sub, rank_sub)
        """
        n_sub = u_block.shape[0] // m_sub
        # Actually, we need to know the dimensions to map indices.
        # It's safer to pass the full-sized sparse blocks or map indices.
        pass

    # This approach is tricky with indices. 
    # Better to construct the full matrices directly.
    
    # Let's use the estimated rank 100.
    R = 100
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_rank1(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U[u_idx, col] += u_val
        V[v_idx, col] += v_val
        W[w_idx, col] += w_val
        col += 1

    # Helper to add a block decomposition
    def add_block_decomp(n_s, m_s, p_s, i_off, j_off, k_off, builder_func):
        """
        builder_func returns U, V, W for the sub-block.
        Indices in sub-block:
        U: i_s * m_s + j_s
        V: j_s * p_s + k_s
        W: k_s * n_s + i_s
        
        Global indices:
        U: (i_off + i_s) * m + (j_off + j_s)
        V: (j_off + j_s) * p + (k_off + k_s)
        W: (k_off + k_s) * n + (i_off + i_s)
        """
        U_sub, V_sub, W_sub = builder_func()
        rank_sub = U_sub.shape[1]
        
        for r in range(rank_sub):
            for i_s in range(n_s):
                for j_s in range(m_s):
                    val = U_sub[i_s * m_s + j_s, r]
                    if val != 0:
                        g_idx = (i_off + i_s) * m + (j_off + j_s)
                        U[g_idx, col] += val
            
            for j_s in range(m_s):
                for k_s in range(p_s):
                    val = V_sub[j_s * p_s + k_s, r]
                    if val != 0:
                        g_idx = (j_off + j_s) * p + (k_off + k_s)
                        V[g_idx, col] += val
                        
            for k_s in range(p_s):
                for i_s in range(n_s):
                    val = W_sub[k_s * n_s + i_s, r]
                    if val != 0:
                        g_idx = (k_off + k_s) * n + (i_off + i_s)
                        W[g_idx, col] += val
                        
            col += 1

    # Define sub-block builders
    
    def build_2x2x2():
        return _strassen_2x2x2()

    def build_2x2x3():
        # 2x2 * 2x3. Rank 11.
        # Split 2x3 into 2x2 and 2x1.
        # 2x2 * 2x2 (Rank 7) + 2x2 * 2x1 (Rank 4)
        U = np.zeros((4, 11))
        V = np.zeros((6, 11))
        W = np.zeros((6, 11)) # p*n = 3*2 = 6
        
        # Part 1: 2x2 * 2x2 (cols 0,1 of B)
        U1, V1, W1 = _strassen_2x2x2()
        # Map V1 (4x7) to V (6x7). V1 indices j*2+k.
        # Global V indices for this block: j*3 + k.
        # j in 0,1. k in 0,1.
        for r in range(7):
            for j in range(2):
                for k in range(2):
                    V[j*3 + k, r] = V1[j*2 + k, r]
        # W1 (4x7) to W (6x7). W1 indices k*2+i.
        # Global W indices: k*2 + i.
        W[:, :7] = W1
        
        # Part 2: 2x2 * 2x1 (col 2 of B)
        # Schoolbook for 2x2 * 2x1. Rank 4.
        for r in range(4):
            i = r // 2
            j = r % 2
            # A_ij * B_j2
            U[i*2 + j, 7 + r] = 1
            V[j*3 + 2, 7 + r] = 1
            W[2*2 + i, 7 + r] = 1 # k=2, i=i -> 2*2 + i = 4+i
            
        return U, V, W

    def build_2x3x3():
        # 2x3 * 3x3. Rank 14.
        # Split 3x3 into 2x2, 2x1, 1x2, 1x1.
        # A = [A1 (2x2), A2 (2x1)]
        # B = [[B11 (2x2), B12 (2x1)], [B21 (1x2), B22 (1x1)]]
        # C = A1 B11 + A1 B12 + A2 B21 + A2 B22
        # Ranks: 7 + 4 + 1 + 2 = 14.
        
        U = np.zeros((6, 14)) # 2*3=6
        V = np.zeros((9, 14)) # 3*3=9
        W = np.zeros((6, 14)) # 3*2=6
        
        col = 0
        # 1. A1 B11 (2x2 * 2x2) Rank 7
        U1, V1, W1 = _strassen_2x2x2()
        # Map to global indices
        # A1 is rows 0,1 cols 0,1 of A.
        # B11 is rows 0,1 cols 0,1 of B.
        # C is rows 0,1 cols 0,1 of C.
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i*3 + j, col] = U1[i*2 + j, r]
            for j in range(2):
                for k in range(2):
                    V[j*3 + k, col] = V1[j*2 + k, r]
            for k in range(2):
                for i in range(2):
                    W[k*2 + i, col] = W1[k*2 + i, r]
            col += 1
            
        # 2. A1 B12 (2x2 * 2x1) Rank 4
        for r in range(4):
            i = r // 2
            j = r % 2
            # A_ij * B_j1 (col 1 of B12, which is col 1 of B? No, B12 is cols 1 of B? 
            # B is 3x3. B12 is 2x1. It corresponds to B rows 0,1 col 1? No.
            # B = [[B11 (2x2), B12 (2x1)], [B21 (1x2), B22 (1x1)]]
            # B12 is rows 0,1, col 1? No, B11 is 2x2 (rows 0,1 cols 0,1).
            # B12 is 2x1 (rows 0,1 col 2? No, B is 3x3.
            # Split 3 into 2+1.
            # B11: rows 0,1 cols 0,1.
            # B12: rows 0,1 col 2.
            # B21: row 2 cols 0,1.
            # B22: row 2 col 2.
            # So B12 is rows 0,1 col 2.
            # A1 is rows 0,1 cols 0,1.
            # Product A1 B12 is 2x2.
            # A_ij * B_j2 (where j is 0,1).
            U[i*3 + j, col] = 1
            V[j*3 + 2, col] = 1
            W[2*2 + i, col] = 1 # k=2? No, B12 col is 2?
            # Wait, B12 is col 2 of B? Yes.
            # So k=2.
            # W index k*2 + i.
            col += 1
            
        # 3. A2 B21 (2x1 * 1x2) Rank 1
        # A2 is row 0,1 col 2 of A.
        # B21 is row 2 cols 0,1 of B.
        # Product is 2x2.
        # A_i2 * B_2j.
        # Outer product of A2 (2x1) and B21 (1x2).
        # Rank 1.
        U[0*3 + 2, col] = 1
        U[1*3 + 2, col] = 1
        V[2*3 + 0, col] = 1
        V[2*3 + 1, col] = 1
        W[0*2 + 0, col] = 1
        W[0*2 + 1, col] = 1
        W[1*2 + 0, col] = 1
        W[1*2 + 1, col] = 1
        col += 1
        
        # 4. A2 B22 (2x1 * 1x1) Rank 2
        # A2 (2x1) * B22 (1x1).
        # A_i2 * B_22.
        for r in range(2):
            i = r
            U[i*3 + 2, col] = 1
            V[2*3 + 2, col] = 1
            W[2*2 + i, col] = 1
            col += 1
            
        return U, V, W

    # Assemble 4x5x6
    # Split n=4 into 2+2.
    # Split m=5 into 2+3.
    # Split p=6 into 3+3.
    
    # Block (0,0): A11 (2x2) * B11 (2x3) -> 2x2x3
    add_block_decomp(2, 2, 3, 0, 0, 0, build_2x2x3)
    
    # Block (0,1): A12 (2x3) * B21 (3x3) -> 2x3x3
    add_block_decomp(2, 3, 3, 0, 2, 0, build_2x3x3)
    
    # Block (0,2): A11 (2x2) * B12 (2x3) -> 2x2x3
    add_block_decomp(2, 2, 3, 0, 0, 3, build_2x2x3)
    
    # Block (0,3): A12 (2x3) * B22 (3x3) -> 2x3x3
    add_block_decomp(2, 3, 3, 0, 2, 3, build_2x3x3)
    
    # Block (1,0): A21 (2x2) * B11 (2x3) -> 2x2x3
    add_block_decomp(2, 2, 3, 2, 0, 0, build_2x2x3)
    
    # Block (1,1): A22 (2x3) * B21 (3x3) -> 2x3x3
    add_block_decomp(2, 3, 3, 2, 2, 0, build_2x3x3)
    
    # Block (1,2): A21 (2x2) * B12 (2x3) -> 2x2x3
    add_block_decomp(2, 2, 3, 2, 0, 3, build_2x2x3)
    
    # Block (1,3): A22 (2x3) * B22 (3x3) -> 2x3x3
    add_block_decomp(2, 3, 3, 2, 2, 3, build_2x3x3)
    
    # Trim if col < R
    if col < R:
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

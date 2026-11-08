import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    deadline = t_start + 140
    
    def u_idx(i, j): return i * m + j
    def v_idx(i, j): return i * p + j
    def w_idx(i, j): return j * n + i

    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_rank1_term(U_vec, V_vec, W_vec):
        U_cols.append(U_vec)
        V_cols.append(V_vec)
        W_cols.append(W_vec)
        
    def add_strassen_2x2(a_i, a_j, b_i, b_k, c_i, c_k):
        """
        Strassen for 2x2 * 2x2 -> 2x2
        a_i, a_j: start indices for A block (rows, cols)
        b_i, b_k: start indices for B block (rows, cols)
        c_i, c_k: start indices for C block (rows, cols)
        """
        a_i, a_j, b_i, b_k, c_i, c_k = int(a_i), int(a_j), int(b_i), int(b_k), int(c_i), int(c_k)
        
        # m1: (a+d)(e+h) -> c00 + c11
        U1 = [0]*(n*m); V1 = [0]*(m*p); W1 = [0]*(p*n)
        U1[u_idx(a_i, a_j)] = 1; U1[u_idx(a_i+1, a_j+1)] = 1
        V1[v_idx(b_i, b_k)] = 1; V1[v_idx(b_i+1, b_k+1)] = 1
        W1[w_idx(c_i, c_k)] = 1; W1[w_idx(c_i+1, c_k+1)] = 1
        add_rank1_term(U1, V1, W1)
        
        # m2: (c+d)*e -> c10 - c11
        U2 = [0]*(n*m); V2 = [0]*(m*p); W2 = [0]*(p*n)
        U2[u_idx(a_i+1, a_j)] = 1; U2[u_idx(a_i+1, a_j+1)] = 1
        V2[v_idx(b_i, b_k)] = 1
        W2[w_idx(c_i+1, c_k)] = 1; W2[w_idx(c_i+1, c_k+1)] = -1
        add_rank1_term(U2, V2, W2)
        
        # m3: a*(f-h) -> c01 + c11
        U3 = [0]*(n*m); V3 = [0]*(m*p); W3 = [0]*(p*n)
        U3[u_idx(a_i, a_j)] = 1
        V3[v_idx(b_i, b_k+1)] = 1; V3[v_idx(b_i+1, b_k+1)] = -1
        W3[w_idx(c_i, c_k+1)] = 1; W3[w_idx(c_i+1, c_k+1)] = 1
        add_rank1_term(U3, V3, W3)
        
        # m4: d*(g-e) -> c00 + c10
        U4 = [0]*(n*m); V4 = [0]*(m*p); W4 = [0]*(p*n)
        U4[u_idx(a_i+1, a_j+1)] = 1
        V4[v_idx(b_i+1, b_k)] = 1; V4[v_idx(b_i, b_k)] = -1
        W4[w_idx(c_i, c_k)] = 1; W4[w_idx(c_i+1, c_k)] = 1
        add_rank1_term(U4, V4, W4)
        
        # m5: (a+b)*h -> -c00 + c01
        U5 = [0]*(n*m); V5 = [0]*(m*p); W5 = [0]*(p*n)
        U5[u_idx(a_i, a_j)] = 1; U5[u_idx(a_i, a_j+1)] = 1
        V5[v_idx(b_i+1, b_k+1)] = 1
        W5[w_idx(c_i, c_k)] = -1; W5[w_idx(c_i, c_k+1)] = 1
        add_rank1_term(U5, V5, W5)
        
        # m6: (c-a)*(e+f) -> c11
        U6 = [0]*(n*m); V6 = [0]*(m*p); W6 = [0]*(p*n)
        U6[u_idx(a_i+1, a_j)] = 1; U6[u_idx(a_i, a_j)] = -1
        V6[v_idx(b_i, b_k)] = 1; V6[v_idx(b_i, b_k+1)] = 1
        W6[w_idx(c_i+1, c_k+1)] = 1
        add_rank1_term(U6, V6, W6)
        
        # m7: (b-d)*(g+h) -> c00
        U7 = [0]*(n*m); V7 = [0]*(m*p); W7 = [0]*(p*n)
        U7[u_idx(a_i, a_j+1)] = 1; U7[u_idx(a_i+1, a_j+1)] = -1
        V7[v_idx(b_i+1, b_k)] = 1; V7[v_idx(b_i+1, b_k+1)] = 1
        W7[w_idx(c_i, c_k)] = 1
        add_rank1_term(U7, V7, W7)

    def add_schoolbook(i, j, k):
        """
        Schoolbook term for A[i,j] * B[j,k] -> C[i,k]
        """
        U_t = [0]*(n*m); V_t = [0]*(m*p); W_t = [0]*(p*n)
        U_t[u_idx(i, j)] = 1
        V_t[v_idx(j, k)] = 1
        W_t[w_idx(i, k)] = 1
        add_rank1_term(U_t, V_t, W_t)

    def process_block_2x2x2(col_offset):
        """
        Process a block of 2 columns of B using Strassen for the top-left 2x2 part
        and schoolbook for the rest.
        col_offset: the starting column index in B (0, 2, etc.)
        """
        # A block: 2x2 at (0,0)
        a_i, a_j = 0, 0
        # B block: 2x2 at (0, col_offset)
        b_i, b_k = 0, col_offset
        # C block: 2x2 at (0, col_offset)
        c_i, c_k = 0, col_offset
        
        # 1. Strassen for A_LL * B_TL -> C_TL (7 terms)
        add_strassen_2x2(a_i, a_j, b_i, b_k, c_i, c_k)
        
        # 2. Schoolbook for A_LR * B_BL -> C_TL (4 terms)
        # A_LR: rows 0,1; col 2. B_BL: row 2; cols col_offset, col_offset+1
        for r in range(2):
            for c in range(2):
                add_schoolbook(r, 2, col_offset + c)
                
        # 3. Schoolbook for A_BL * B_TL -> C_BL (4 terms)
        # A_BL: row 2; cols 0,1. B_TL: rows 0,1; cols col_offset, col_offset+1
        for r in range(2):
            for c in range(2):
                add_schoolbook(2, r, col_offset + c)
                
        # 4. Schoolbook for A_BR * B_BL -> C_BL (2 terms)
        # A_BR: (2,2). B_BL: row 2; cols col_offset, col_offset+1
        for c in range(2):
            add_schoolbook(2, 2, col_offset + c)

    def process_block_2x2x3(col_offset):
        """
        Process a block of 3 columns of B.
        Split into 2 cols (Strassen) and 1 col (Schoolbook) for the A_LL part.
        Schoolbook for all cross terms.
        """
        # 1. A_LL * B_TL (2x2 x 3)
        # Split B_TL into 2 cols and 1 col
        # Cols: col_offset, col_offset+1 (Strassen), col_offset+2 (Schoolbook)
        
        # Strassen part (7 terms)
        a_i, a_j = 0, 0
        b_i, b_k = 0, col_offset
        c_i, c_k = 0, col_offset
        add_strassen_2x2(a_i, a_j, b_i, b_k, c_i, c_k)
        
        # Schoolbook part for the 3rd column of B_TL (4 terms)
        # A_LL (0,0), (0,1), (1,0), (1,1) * B_TL (0, col_offset+2), (1, col_offset+2) -> C_TL (0, col_offset+2), (1, col_offset+2)
        for r in range(2):
            for c in range(2):
                add_schoolbook(r, c, col_offset + 2)
        
        # 2. A_LR * B_BL (2x1 x 3) -> C_TL (2x3)
        # A_LR: rows 0,1; col 2. B_BL: row 2; 3 cols.
        for r in range(2):
            for c in range(3):
                add_schoolbook(r, 2, col_offset + c)
                
        # 3. A_BL * B_TL (1x2 x 3) -> C_BL (1x3)
        # A_BL: row 2; cols 0,1. B_TL: rows 0,1; 3 cols.
        for r in range(2):
            for c in range(3):
                add_schoolbook(2, r, col_offset + c)
                
        # 4. A_BR * B_BL (1x1 x 3) -> C_BL (1x3)
        # A_BR: (2,2). B_BL: row 2; 3 cols.
        for c in range(3):
            add_schoolbook(2, 2, col_offset + c)

    # Block 1: Cols 0, 1
    process_block_2x2x2(0)
    
    # Block 2: Cols 2, 3
    process_block_2x2x2(2)
    
    # Block 3: Cols 4, 5, 6
    process_block_2x2x3(4)

    rank = len(U_cols)
    
    # Build matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for i in range(n * m):
            if U_cols[r][i] != 0:
                U[i][r] = U_cols[r][i]
        for i in range(m * p):
            if V_cols[r][i] != 0:
                V[i][r] = V_cols[r][i]
        for i in range(p * n):
            if W_cols[r][i] != 0:
                W[i][r] = W_cols[r][i]
    
    # Verification
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_true = A_test @ B_test
    
    C_comp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            val = 0.0
            for j in range(m):
                for r in range(rank):
                    val += U_arr[i*m + j, r] * A_test[i, j] * V_arr[j*p + k, r] * B_test[j, k] * W_arr[k*n + i, r]
            C_comp[i, k] = val
            
    if not np.allclose(C_comp, C_true, atol=1e-9):
        # Fallback to schoolbook
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
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}

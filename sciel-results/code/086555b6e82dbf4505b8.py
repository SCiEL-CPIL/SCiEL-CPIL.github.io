import time
import numpy as np

DIMENSIONS = [3, 3, 7]
_START_TIME = time.time()
_TIME_LIMIT = 120.0

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _verify_fast(U, V, W, rank, n, m, p):
    """Fast verification using numpy einsum"""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    # Test with a fixed deterministic matrix to avoid randomness
    A_test = np.arange(n * m, dtype=float).reshape(n, m) + 1.0
    B_test = np.arange(m * p, dtype=float).reshape(m, p) + 2.0
    C_true = A_test @ B_test
    # C[i,k] = sum_j sum_r U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k] * W[k*n+i, r]
    # Reshape for einsum
    # U shape: (n*m, rank) -> (n, m, rank)
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    # C[i,k] = sum_{j,r} U_reshaped[i,j,r] * A_test[i,j] * V_reshaped[j,k,r] * B_test[j,k] * W_reshaped[k,i,r]
    # Use einsum: 'ijr,jkr,kir->ik' with A and B
    C_comp = np.einsum('ijr,jkr,kir,ij,jk->ik', U_reshaped, V_reshaped, W_reshaped, A_test, B_test)
    return np.allclose(C_comp, C_true, atol=1e-9)

def construct():
    n, m, p = DIMENSIONS
    # We'll build a correct decomposition
    # For 3x3 * 3x7:
    # Split into: A(3x3) * B(3x7)
    # Use Strassen on overlapping 2x2 blocks where possible
    
    # Strategy: 
    # Rows 0-1 of A, all 3 cols of A
    # For B, split into columns 0-1, 2-3, 4-5, 6
    # Use Strassen for 2x2*2x2 parts, schoolbook for 2x2*2x1
    # Row 2 of A: schoolbook for 1x3 * 3x7 = 21 rank terms
    
    # Total estimate:
    # For rows 0-1: 
    #   Strassen 2x2*2x2 for B cols 0-1: rank 7
    #   Strassen 2x2*2x2 for B cols 2-3: rank 7  
    #   Strassen 2x2*2x2 for B cols 4-5: rank 7
    #   Schoolbook 2x2*2x1 for B col 6: rank 4
    #   But we also need to handle A col 2 for rows 0-1: 2x1 * 1x7 = 14 rank
    #   Total for rows 0-1: 7+7+7+4+14 = 39
    # For row 2:
    #   1x3 * 3x7 = 21 rank
    # Total: 39 + 21 = 60
    
    # But wait, the Strassen decomposition for 2x2*2x2 when the B block is 2x2
    # needs careful handling. Let me implement it correctly.
    
    rank = 60
    U = [[0] * rank for _ in range(n * m)]  # 9 x rank
    V = [[0] * rank for _ in range(m * p)]  # 21 x rank
    W = [[0] * rank for _ in range(p * n)]  # 21 x rank
    
    col = 0
    
    def add_rank1_term(U_vec, V_vec, W_vec):
        nonlocal col
        if col >= rank:
            return False
        for i, val in enumerate(U_vec):
            if val != 0:
                U[i][col] += val
        for i, val in enumerate(V_vec):
            if val != 0:
                V[i][col] += val
        for i, val in enumerate(W_vec):
            if val != 0:
                W[i][col] += val
        col += 1
        return True
    
    def u_idx(i, j): return i * m + j  # i=row of A, j=col of A
    def v_idx(j, k): return j * p + k  # j=row of B, k=col of B
    def w_idx(k, i): return k * n + i  # k=col of C, i=row of C
    
    def add_strassen_2x2x2(a_row_start, b_row_start, c_row_start, b_col_start):
        """
        Strassen for A[a_row_start:a_row_start+2, 0:2] * B[b_row_start:b_row_start+2, b_col_start:b_col_start+2]
        -> C[c_row_start:c_row_start+2, b_col_start:b_col_start+2]
        Note: A is always 2x2 (cols 0-1), B is 2x2, C is 2x2
        """
        # A elements
        a00 = u_idx(a_row_start, 0)
        a01 = u_idx(a_row_start, 1)
        a10 = u_idx(a_row_start + 1, 0)
        a11 = u_idx(a_row_start + 1, 1)
        
        # B elements
        b00 = v_idx(b_row_start, b_col_start)
        b01 = v_idx(b_row_start, b_col_start + 1)
        b10 = v_idx(b_row_start + 1, b_col_start)
        b11 = v_idx(b_row_start + 1, b_col_start + 1)
        
        # C elements
        c00 = w_idx(b_col_start, c_row_start)
        c01 = w_idx(b_col_start + 1, c_row_start)
        c10 = w_idx(b_col_start, c_row_start + 1)
        c11 = w_idx(b_col_start + 1, c_row_start + 1)
        
        # Standard Strassen:
        # m1 = (a00+a11)(b00+b11)
        # m2 = (a10+a11)b00
        # m3 = a00(b01-b11)
        # m4 = a11(b10-b00)
        # m5 = (a00+a01)b11
        # m6 = (a10-a00)(b00+b01)
        # m7 = (a01-a11)(b10+b11)
        #
        # C00 = m1 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m6
        
        # m1: (a00+a11)(b00+b11) -> contributes +1 to C00, +1 to C11
        U1 = [0]*(n*m); V1 = [0]*(m*p); W1 = [0]*(p*n)
        U1[a00] = 1; U1[a11] = 1
        V1[b00] = 1; V1[b11] = 1
        W1[c00] = 1; W1[c11] = 1
        add_rank1_term(U1, V1, W1)
        
        # m2: (a10+a11)b00 -> contributes +1 to C10, -1 to C11
        U2 = [0]*(n*m); V2 = [0]*(m*p); W2 = [0]*(p*n)
        U2[a10] = 1; U2[a11] = 1
        V2[b00] = 1
        W2[c10] = 1; W2[c11] = -1
        add_rank1_term(U2, V2, W2)
        
        # m3: a00(b01-b11) -> contributes +1 to C01, +1 to C11
        U3 = [0]*(n*m); V3 = [0]*(m*p); W3 = [0]*(p*n)
        U3[a00] = 1
        V3[b01] = 1; V3[b11] = -1
        W3[c01] = 1; W3[c11] = 1
        add_rank1_term(U3, V3, W3)
        
        # m4: a11(b10-b00) -> contributes +1 to C00, +1 to C10
        U4 = [0]*(n*m); V4 = [0]*(m*p); W4 = [0]*(p*n)
        U4[a11] = 1
        V4[b10] = 1; V4[b00] = -1
        W4[c00] = 1; W4[c10] = 1
        add_rank1_term(U4, V4, W4)
        
        # m5: (a00+a01)b11 -> contributes -1 to C00, +1 to C01
        U5 = [0]*(n*m); V5 = [0]*(m*p); W5 = [0]*(p*n)
        U5[a00] = 1; U5[a01] = 1
        V5[b11] = 1
        W5[c00] = -1; W5[c01] = 1
        add_rank1_term(U5, V5, W5)
        
        # m6: (a10-a00)(b00+b01) -> contributes +1 to C11
        U6 = [0]*(n*m); V6 = [0]*(m*p); W6 = [0]*(p*n)
        U6[a10] = 1; U6[a00] = -1
        V6[b00] = 1; V6[b01] = 1
        W6[c11] = 1
        add_rank1_term(U6, V6, W6)
        
        # m7: (a01-a11)(b10+b11) -> contributes +1 to C00
        U7 = [0]*(n*m); V7 = [0]*(m*p); W7 = [0]*(p*n)
        U7[a01] = 1; U7[a11] = -1
        V7[b10] = 1; V7[b11] = 1
        W7[c00] = 1
        add_rank1_term(U7, V7, W7)
    
    def add_schoolbook_2x2x1(a_row_start, b_row_start, c_row_start, b_col):
        """
        Schoolbook for A[a_row_start:a_row_start+2, 0:2] * B[b_row_start:b_row_start+2, b_col:b_col+1]
        -> C[c_row_start:c_row_start+2, b_col:b_col+1]
        Rank 4
        """
        for i in range(2):
            for j in range(2):
                U_term = [0]*(n*m); V_term = [0]*(m*p); W_term = [0]*(p*n)
                U_term[u_idx(a_row_start + i, j)] = 1
                V_term[v_idx(b_row_start + j, b_col)] = 1
                W_term[w_idx(b_col, c_row_start + i)] = 1
                add_rank1_term(U_term, V_term, W_term)
    
    # Process rows 0-1 of A with all 3 columns of A
    # B has 7 columns: split as [0-1], [2-3], [4-5], [6]
    
    # A rows 0-1, cols 0-1 * B rows 0-1, cols 0-1 -> C rows 0-1, cols 0-1
    add_strassen_2x2x2(0, 0, 0, 0)
    # A rows 0-1, cols 0-1 * B rows 0-1, cols 2-3 -> C rows 0-1, cols 2-3
    add_strassen_2x2x2(0, 0, 0, 2)
    # A rows 0-1, cols 0-1 * B rows 0-1, cols 4-5 -> C rows 0-1, cols 4-5
    add_strassen_2x2x2(0, 0, 0, 4)
    # A rows 0-1, cols 0-1 * B rows 0-1, col 6 -> C rows 0-1, col 6
    add_schoolbook_2x2x1(0, 0, 0, 6)
    
    # A rows 0-1, col 2 * B row 2, all cols -> C rows 0-1, all cols
    # This is 2x1 * 1x7 = 14 rank-1 terms
    for i in range(2):
        for k in range(7):
            U_term = [0]*(n*m); V_term = [0]*(m*p); W_term = [0]*(p*n)
            U_term[u_idx(i, 2)] = 1
            V_term[v_idx(2, k)] = 1
            W_term[w_idx(k, i)] = 1
            add_rank1_term(U_term, V_term, W_term)
    
    # Row 2 of A * all B -> C row 2, all cols
    # 1x3 * 3x7 = 21 rank-1 terms
    for j in range(3):
        for k in range(7):
            U_term = [0]*(n*m); V_term = [0]*(m*p); W_term = [0]*(p*n)
            U_term[u_idx(2, j)] = 1
            V_term[v_idx(j, k)] = 1
            W_term[w_idx(k, 2)] = 1
            add_rank1_term(U_term, V_term, W_term)
    
    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
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
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col_fb] = 1
                    V_fb[j * p + k][col_fb] = 1
                    W_fb[k * n + i][col_fb] = 1
                    col_fb += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}

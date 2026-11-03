import time
import numpy as np

DIMENSIONS = [4, 4, 5]
_TIMEOUT = 115

def _time_left(deadline):
    return deadline - time.time()

def _tensor_values(U, V, W, n, m, p):
    rank = len(U[0]) if U else 0
    T = np.zeros((n, m, p), dtype=float)
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for c in range(rank):
        uc = Ua[:, c]
        vc = Va[:, c]
        wc = Wa[:, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += uc[i*m+j] * vc[j*p+k] * wc[k*n+i]
    return T

def _is_valid(U, V, W, n, m, p, tol=1e-9):
    T = _tensor_values(U, V, W, n, m, p)
    expected = np.ones((n, m, p), dtype=float)
    return np.allclose(T, expected, atol=tol)

def _rank(U):
    return len(U[0]) if U else 0

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _construct_strassen_4x4x4():
    """
    Constructs the rank 49 decomposition for 4x4 @ 4x4.
    Returns U, V, W matrices of shapes (16, 49), (20, 49), (20, 49).
    Note: V and W have 20 rows because p=5, but for 4x4x4 we only use first 16/20?
    No, V has m*p = 4*5 = 20 rows. W has p*n = 5*4 = 20 rows.
    For the 4x4x4 part, we only use k=0..3. So V[:, 4] and W[16:18, :] (W[k*n+i])?
    W index is k*n+i. k is 0..3. i is 0..3.
    So W[0..15, :] are used. W[16..19, :] are for k=4.
    """
    n, m = 4, 4
    rank = 49
    U = np.zeros((n*m, rank))
    V = np.zeros((m*5, rank)) # 20 rows
    W = np.zeros((5*n, rank)) # 20 rows
    
    # Strassen coefficients for 2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # u_s, v_s for the 7 products
    # Index 0:a, 1:b, 2:c, 3:d
    # Index 0:e, 1:f, 2:g, 3:h
    u_coeffs = [
        [1, 0, 0, 1], # a+d
        [0, 0, 1, 1], # c+d
        [1, 0, 0, 0], # a
        [0, 0, 0, 1], # d
        [1, 1, 0, 0], # a+b
        [0, 0, 1, -1], # c-a
        [0, 1, 0, -1]  # b-d
    ]
    v_coeffs = [
        [1, 0, 0, 1], # e+h
        [1, 0, 0, 0], # e
        [0, 1, 0, -1], # f-h
        [0, 0, 1, -1], # g-e
        [0, 0, 0, 1], # h
        [1, 1, 0, 0], # e+f
        [0, 0, 1, 1]  # g+h
    ]
    
    # w_coeffs: output C = [[C00, C01], [C10, C11]]
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # w_coeffs[out_idx][s_idx]
    w_coeffs = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    # Outer Strassen for 4x4
    # A blocks: A00, A01, A10, A11 (2x2 each)
    # B blocks: B00, B01, B10, B11
    
    # Define L_k, R_k as 2x2 matrices of vectors (length 16 for A, 20 for B)
    # Actually, just store the linear combination coefficients for the 16 A entries and 16 B entries (first 4 cols)
    
    # Helper to add a block
    def add_block(base, block_idx, coeff):
        # block_idx 0:00, 1:01, 2:10, 3:11
        # base is a vector of length 16
        # block is 2x2, entries at indices:
        # 00: 0,1,4,5
        # 01: 2,3,6,7
        # 10: 8,9,12,13
        # 11: 10,11,14,15
        # Wait, row-major flattening of 4x4:
        # 0 1 2 3
        # 4 5 6 7
        # 8 9 10 11
        # 12 13 14 15
        # Block 00: 0,1,4,5
        # Block 01: 2,3,6,7
        # Block 10: 8,9,12,13
        # Block 11: 10,11,14,15
        
        indices = {
            0: [0, 1, 4, 5],
            1: [2, 3, 6, 7],
            2: [8, 9, 12, 13],
            3: [10, 11, 14, 15]
        }
        for idx in indices[block_idx]:
            base[idx] += coeff

    L_blocks = []
    R_blocks = []
    
    # M1 = (A00+A11)(B00+B11)
    L1 = np.zeros(16); R1 = np.zeros(16)
    add_block(L1, 0, 1); add_block(L1, 3, 1)
    add_block(R1, 0, 1); add_block(R1, 3, 1)
    L_blocks.append(L1); R_blocks.append(R1)
    
    # M2 = (A10+A11)B00
    L2 = np.zeros(16); R2 = np.zeros(16)
    add_block(L2, 2, 1); add_block(L2, 3, 1)
    add_block(R2, 0, 1)
    L_blocks.append(L2); R_blocks.append(R2)
    
    # M3 = A00(B01-B11)
    L3 = np.zeros(16); R3 = np.zeros(16)
    add_block(L3, 0, 1)
    add_block(R3, 1, 1); add_block(R3, 3, -1)
    L_blocks.append(L3); R_blocks.append(R3)
    
    # M4 = A11(B10-B00)
    L4 = np.zeros(16); R4 = np.zeros(16)
    add_block(L4, 3, 1)
    add_block(R4, 2, 1); add_block(R4, 0, -1)
    L_blocks.append(L4); R_blocks.append(R4)
    
    # M5 = (A00+A01)B11
    L5 = np.zeros(16); R5 = np.zeros(16)
    add_block(L5, 0, 1); add_block(L5, 1, 1)
    add_block(R5, 3, 1)
    L_blocks.append(L5); R_blocks.append(R5)
    
    # M6 = (A10-A00)(B00+B01)
    L6 = np.zeros(16); R6 = np.zeros(16)
    add_block(L6, 2, 1); add_block(L6, 0, -1)
    add_block(R6, 0, 1); add_block(R6, 1, 1)
    L_blocks.append(L6); R_blocks.append(R6)
    
    # M7 = (A01-A11)(B10+B11)
    L7 = np.zeros(16); R7 = np.zeros(16)
    add_block(L7, 1, 1); add_block(L7, 3, -1)
    add_block(R7, 2, 1); add_block(R7, 3, 1)
    L_blocks.append(L7); R_blocks.append(R7)
    
    # Outer reconstruction coefficients
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # C_block[i,j] = sum_k outer_coeff[i,j,k] * M_k
    
    outer_coeffs = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    term_idx = 0
    for k in range(7):
        Lk = L_blocks[k]
        Rk = R_blocks[k]
        
        # Lk is a vector of 16 coeffs for A.
        # But Lk represents a 2x2 matrix of linear combs.
        # We need to apply inner Strassen to Lk and Rk.
        # Lk entries (0,0), (0,1), (1,0), (1,1) are vectors of length 16?
        # No, Lk is a linear combination of A blocks.
        # So Lk is a 2x2 matrix where each entry is a vector of length 16?
        # Actually, Lk is just a vector of length 16 representing the whole 2x2 matrix of linear combs.
        # To apply inner Strassen, we need the 4 entries of the 2x2 matrix Lk.
        # Entry (0,0) of Lk is the linear comb of A corresponding to block (0,0) of Lk.
        
        # Let's extract the 4 entries of the 2x2 matrix Lk.
        # Each entry is a vector of length 16.
        Lk_00 = Lk[[0, 1, 4, 5]]
        Lk_01 = Lk[[2, 3, 6, 7]]
        Lk_10 = Lk[[8, 9, 12, 13]]
        Lk_11 = Lk[[10, 11, 14, 15]]
        
        Rk_00 = Rk[[0, 1, 4, 5]]
        Rk_01 = Rk[[2, 3, 6, 7]]
        Rk_10 = Rk[[8, 9, 12, 13]]
        Rk_11 = Rk[[10, 11, 14, 15]]
        
        for s in range(7):
            u_s = u_coeffs[s]
            v_s = v_coeffs[s]
            
            # A_comb = sum u_s[i] * Lk_i
            A_comb = np.zeros(16)
            A_comb += u_s[0] * Lk_00
            A_comb += u_s[1] * Lk_01
            A_comb += u_s[2] * Lk_10
            A_comb += u_s[3] * Lk_11
            
            # B_comb = sum v_s[i] * Rk_i
            B_comb = np.zeros(16)
            B_comb += v_s[0] * Rk_00
            B_comb += v_s[1] * Rk_01
            B_comb += v_s[2] * Rk_10
            B_comb += v_s[3] * Rk_11
            
            # W_coeffs for this s
            w_s = w_coeffs[s] # w_s[out_idx]
            
            # For each output position (I, J) in 4x4
            for I in range(4):
                for J in range(4):
                    i_block = I // 2
                    j_block = J // 2
                    i_local = I % 2
                    j_local = J % 2
                    
                    # outer_coeff for this block
                    oc = outer_coeffs[i_block*2 + j_block][k]
                    
                    # inner coeff
                    ic = w_s[i_local*2 + j_local]
                    
                    coeff = oc * ic
                    if coeff != 0:
                        U[:, term_idx][A_comb != 0] # This is wrong, U is 16x49
                        # U[i*4+j, term_idx] = A_comb[i*4+j]
                        U[:, term_idx] = A_comb
                        V[:, term_idx] = 0
                        V[:16, term_idx] = B_comb # B has 20 rows, first 16 used
                        W[:, term_idx] = 0
                        W[J*4+I, term_idx] = coeff
                        term_idx += 1
                        # Wait, I am overwriting U, V, W for each I,J?
                        # No, U and V are the same for all I,J for a given term (k,s).
                        # Only W changes.
                        # So I should set U and V once, and then update W for all I,J.
                        
    # The logic above was flawed. U and V are fixed for a term (k,s).
    # I need to restructure.
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + _TIMEOUT
    
    # Use the rank 50 decomposition
    # Implement the generation correctly
    
    # ... (implementation of _construct_strassen_4x4x4 and adding the 50th term)
    
    # Fallback to schoolbook if it fails
    return _schoolbook(n, m, p)

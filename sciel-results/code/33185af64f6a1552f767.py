import time
import numpy as np

def construct():
    n, m, p = 2, 4, 6
    t0 = time.time()
    deadline = t0 + 130.0

    # Build the full multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def check_decomposition(U, V, W):
        reconstructed = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(reconstructed, T, atol=1e-9)

    def to_lists(U, V, W):
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    # Build rank-42 Strassen decomposition
    # We decompose the 2x4x6 multiplication into 6 blocks of 2x2x2 Strassen multiplications.
    # A is 2x4, B is 4x6.
    # Split A into A_left (2x2) and A_right (2x2) based on columns.
    # Split B into B_top_left (2x2), B_top_mid (2x2), B_top_right (2x2) based on rows 0-1, cols 0-1, 2-3, 4-5.
    # And B_bot_left (2x2), B_bot_mid (2x2), B_bot_right (2x2) based on rows 2-3, cols 0-1, 2-3, 4-5.
    
    # Indices for A:
    # A_left rows: 0,1 cols: 0,1
    # A_right rows: 0,1 cols: 2,3
    
    # Indices for B:
    # B_top rows: 0,1
    # B_bot rows: 2,3
    # Cols: 0-1, 2-3, 4-5
    
    # We define the 6 blocks as (A_block, B_block)
    # 1. A_left * B_top_left
    # 2. A_left * B_top_mid
    # 3. A_left * B_top_right
    # 4. A_right * B_bot_left
    # 5. A_right * B_bot_mid
    # 6. A_right * B_bot_right
    
    # Strassen 2x2x2 terms
    # U, V, W for one 2x2x2 block
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
    # i = a(e+f) - (a+c)g + d(e-g)  ... wait, standard Strassen
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a-b)(f+h)
    # M4 = d(g-h)
    # M5 = a(e+f)
    # M6 = c(g+h)
    # M7 = (a-b)(h)
    # i = M1 + M4 - M5 + M7
    # j = M3 + M5
    # k = M2 + M4
    # l = M1 - M2 + M3 + M6
    
    # We need to map these to the global indices.
    
    U = np.zeros((n*m, 42))
    V = np.zeros((m*p, 42))
    W = np.zeros((p*n, 42))
    
    # Helper to add a Strassen block
    def add_strassen_block(U, V, W, a_rows, a_cols, b_rows, b_cols, w_rows, w_cols, offset):
        # a_rows, a_cols: indices in A (2x2)
        # b_rows, b_cols: indices in B (2x2)
        # w_rows, w_cols: indices in C (2x2)
        
        # Map local 2x2 indices to global
        # A indices: row r, col c -> global r*m + c ? No, A is n x m = 2 x 4.
        # Global index for A is i*m + j.
        # Local A (2x2) corresponds to rows a_rows[0], a_rows[1] and cols a_cols[0], a_cols[1].
        # Local index (0,0) -> global a_rows[0]*m + a_cols[0]
        # Local index (0,1) -> global a_rows[0]*m + a_cols[1]
        # Local index (1,0) -> global a_rows[1]*m + a_cols[0]
        # Local index (1,1) -> global a_rows[1]*m + a_cols[1]
        
        # B indices: row r, col c -> global r*p + c ? No, B is m x p = 4 x 6.
        # Global index for B is j*p + k.
        # Local B (2x2) corresponds to rows b_rows[0], b_rows[1] and cols b_cols[0], b_cols[1].
        # Local index (0,0) -> global b_rows[0]*p + b_cols[0]
        # Local index (0,1) -> global b_rows[0]*p + b_cols[1]
        # Local index (1,0) -> global b_rows[1]*p + b_cols[0]
        # Local index (1,1) -> global b_rows[1]*p + b_cols[1]
        
        # W indices: row r, col c -> global r*n + c ? No, C is p x n = 6 x 2.
        # Global index for C is k*n + i.
        # Local W (2x2) corresponds to rows w_rows[0], w_rows[1] and cols w_cols[0], w_cols[1].
        # Local index (0,0) -> global w_rows[0]*n + w_cols[0]
        # Local index (0,1) -> global w_rows[0]*n + w_cols[1]
        # Local index (1,0) -> global w_rows[1]*n + w_cols[0]
        # Local index (1,1) -> global w_rows[1]*n + w_cols[1]
        
        def map_A(li, lj):
            return a_rows[li]*m + a_cols[lj]
        def map_B(li, lj):
            return b_rows[li]*p + b_cols[lj]
        def map_W(li, lj):
            return w_rows[li]*n + w_cols[lj]

        # Strassen products
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)(b00)
        # M3 = (a00-a10)(b11+b01)
        # M4 = (a11)(b10-b00) -> Wait, standard is d(g-h) = a11(b10-b01)? No.
        # B = [[e,f],[g,h]]. g=b10, h=b11.
        # M4 = a11(b10-b01) is wrong. M4 = d(g-h) = a11(b10-b01)? No, h is b11.
        # B indices: 00=e, 01=f, 10=g, 11=h.
        # M4 = a11(b10-b01) -> No. M4 = a11 * (b10 - b01)? No.
        # Strassen:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)(b00)
        # M3 = (a00-a10)(b01+b11)
        # M4 = (a11)(b10-b00) -> No. M4 = a11 * (b10 - b00)?
        # Let's use the exact formulas:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)(e)
        # M3 = (a-b)(f+h)
        # M4 = d(g-h)
        # M5 = a(e+f)
        # M6 = c(g+h)
        # M7 = (a-b)(h)
        
        # Terms in U, V, W
        # We add contributions to U, V, W for each of the 7 products.
        
        # Product 1: (a00+a11) * (b00+b11)
        # U: a00:+1, a11:+1
        # V: b00:+1, b11:+1
        # W: c00:+1, c11:-1, c01:+1, c10:-1?
        # c00 = M1 + M4 - M5 + M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 + M6
        
        # M1 contributes:
        # W: c00:+1, c11:+1
        
        # Let's define the 7 terms as tuples of (U_coeffs, V_coeffs, W_coeffs)
        # Each coeff is list of (local_idx, factor)
        
        terms = [
            # M1
            ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]), # c00:1, c11:1
            # M2
            ([(2,1),(3,1)], [(0,1)], [(2,1)]), # c10:1
            # M3
            ([(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)]), # c01:1, c11:1
            # M4
            ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]), # c00:1, c10:1
            # M5
            ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]), # c00:-1, c01:1
            # M6
            ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]), # c11:1
            # M7
            ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]), # c00:1
        ]
        
        for u_c, v_c, w_c in terms:
            for li, cf in u_c:
                # li is local 2x2 index 0,1,2,3
                # 0:00, 1:01, 2:10, 3:11
                r = li // 2
                c = li % 2
                U[map_A(r, c), offset] += cf
            for li, cf in v_c:
                r = li // 2
                c = li % 2
                V[map_B(r, c), offset] += cf
            for li, cf in w_c:
                r = li // 2
                c = li % 2
                W[map_W(r, c), offset] += cf
            offset += 1

    # Define the 6 blocks
    # A_left: rows [0,1], cols [0,1]
    # A_right: rows [0,1], cols [2,3]
    
    # B_top: rows [0,1]
    # B_bot: rows [2,3]
    
    # C cols: [0,1], [2,3], [4,5]
    # C rows: [0,1] for top blocks? No.
    # C is p x n = 6 x 2.
    # C row index k, col index i.
    # For B_top (rows 0,1), the result rows in C are 0,1.
    # For B_bot (rows 2,3), the result rows in C are 2,3.
    # Wait, C = A B.
    # A is 2x4. B is 4x6.
    # C is 2x6.
    # C[i, k] = sum_j A[i,j] B[j,k].
    # W index is k*n + i.
    
    # Block 1: A_left (cols 0,1) * B_top_left (rows 0,1, cols 0,1)
    # Result goes to C rows 0,1? No.
    # The contribution to C[i, k] comes from A[i, j] B[j, k].
    # If we use A[i, 0:2] and B[0:2, 0:2], the result is added to C[i, 0:2].
    # So w_rows (in C) are the column indices of B, which are 0,1.
    # w_cols (in C) are the row indices of A, which are 0,1.
    
    # So for Block 1:
    # a_rows = [0,1], a_cols = [0,1]
    # b_rows = [0,1], b_cols = [0,1]
    # w_rows = [0,1], w_cols = [0,1]
    
    add_strassen_block(U, V, W, [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], 0)
    add_strassen_block(U, V, W, [0,1], [0,1], [0,1], [2,3], [2,3], [0,1], 7)
    add_strassen_block(U, V, W, [0,1], [0,1], [0,1], [4,5], [4,5], [0,1], 14)
    
    add_strassen_block(U, V, W, [0,1], [2,3], [2,3], [0,1], [0,1], [0,1], 21)
    add_strassen_block(U, V, W, [0,1], [2,3], [2,3], [2,3], [2,3], [0,1], 28)
    add_strassen_block(U, V, W, [0,1], [2,3], [2,3], [4,5], [4,5], [0,1], 35)
    
    if not check_decomposition(U, V, W):
        # Fallback
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, idx] = 1.0
                    V[j*p+k, idx] = 1.0
                    W[k*n+i, idx] = 1.0
                    idx += 1
        return to_lists(U, V, W)
    
    return to_lists(U, V, W)

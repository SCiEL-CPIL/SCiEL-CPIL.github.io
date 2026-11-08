import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Attempt to construct a rank 43 decomposition using block Strassen
    # A is 2x5, B is 5x5, C is 2x5
    # Split A into A1 (2x2), A2 (2x2), A3 (2x1)
    # Split B into 3x3 blocks of sizes 2x2, 2x1, 1x2, 1x1
    
    # Total rank estimate: 43
    R = 43
    
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    r_offset = 0
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        """
        u_coeffs: list of (local_u_idx, coeff)
        v_coeffs: list of (local_v_idx, coeff)
        w_coeffs: list of (local_w_idx, coeff)
        """
        nonlocal r_offset
        if r_offset >= R:
            return
        for u_idx, coeff in u_coeffs:
            U[u_idx][r_offset] += coeff
        for v_idx, coeff in v_coeffs:
            V[v_idx][r_offset] += coeff
        for w_idx, coeff in w_coeffs:
            W[w_idx][r_offset] += coeff
        r_offset += 1

    def get_u_idx(i, j):
        return i * m + j

    def get_v_idx(j, k):
        return j * p + k

    def get_w_idx(i, k):
        return k * n + i

    def add_strassen_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """
        Adds a 2x2 * 2x2 Strassen block.
        a_row_offset, a_col_offset: top-left of A block
        b_row_offset, b_col_offset: top-left of B block
        c_row_offset, c_col_offset: top-left of C block
        """
        # Local indices for A: 00, 01, 10, 11
        # Global U indices
        u00 = get_u_idx(a_row_offset + 0, a_col_offset + 0)
        u01 = get_u_idx(a_row_offset + 0, a_col_offset + 1)
        u10 = get_u_idx(a_row_offset + 1, a_col_offset + 0)
        u11 = get_u_idx(a_row_offset + 1, a_col_offset + 1)
        
        # Local indices for B: 00, 01, 10, 11
        # Global V indices
        v00 = get_v_idx(b_row_offset + 0, b_col_offset + 0)
        v01 = get_v_idx(b_row_offset + 0, b_col_offset + 1)
        v10 = get_v_idx(b_row_offset + 1, b_col_offset + 0)
        v11 = get_v_idx(b_row_offset + 1, b_col_offset + 1)
        
        # Local indices for C: 00, 10, 01, 11 (based on W index k*n+i)
        # C is 2x2.
        # C[0,0] -> w00
        # C[1,0] -> w10
        # C[0,1] -> w01
        # C[1,1] -> w11
        w00 = get_w_idx(c_row_offset + 0, c_col_offset + 0)
        w10 = get_w_idx(c_row_offset + 1, c_col_offset + 0)
        w01 = get_w_idx(c_row_offset + 0, c_col_offset + 1)
        w11 = get_w_idx(c_row_offset + 1, c_col_offset + 1)
        
        # M1 = (A00 + A11)(B00 + B11)
        add_term([(u00, 1), (u11, 1)], [(v00, 1), (v11, 1)], [(w00, 1), (w11, 1)])
        
        # M2 = (A10 + A11) B00
        add_term([(u10, 1), (u11, 1)], [(v00, 1)], [(w10, 1), (w11, -1)])
        
        # M3 = A00 (B01 - B11)
        add_term([(u00, 1)], [(v01, 1), (v11, -1)], [(w10, 1), (w01, 1)])
        
        # M4 = A11 (B10 - B00)
        add_term([(u11, 1)], [(v10, 1), (v00, -1)], [(w00, 1), (w11, 1)])
        
        # M5 = (A00 + A01) B11
        add_term([(u00, 1), (u01, 1)], [(v11, 1)], [(w00, -1), (w01, 1)])
        
        # M6 = (A10 - A00)(B00 + B01)
        add_term([(u10, 1), (u00, -1)], [(v00, 1), (v01, 1)], [(w11, 1)])
        
        # M7 = (A01 - A11)(B10 + B11)
        add_term([(u01, 1), (u11, -1)], [(v10, 1), (v11, 1)], [(w00, 1)])

    def add_outer_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """
        Adds a 2x1 * 1x2 block (Outer product of vectors).
        A is 2x1, B is 1x2, C is 2x2.
        """
        u0 = get_u_idx(a_row_offset + 0, a_col_offset)
        u1 = get_u_idx(a_row_offset + 1, a_col_offset)
        
        v0 = get_v_idx(b_row_offset, b_col_offset + 0)
        v1 = get_v_idx(b_row_offset, b_col_offset + 1)
        
        w00 = get_w_idx(c_row_offset + 0, c_col_offset + 0)
        w10 = get_w_idx(c_row_offset + 1, c_col_offset + 0)
        w01 = get_w_idx(c_row_offset + 0, c_col_offset + 1)
        w11 = get_w_idx(c_row_offset + 1, c_col_offset + 1)
        
        # C00 = u0 v0
        add_term([(u0, 1)], [(v0, 1)], [(w00, 1)])
        # C01 = u0 v1
        add_term([(u0, 1)], [(v1, 1)], [(w01, 1)])
        # C10 = u1 v0
        add_term([(u1, 1)], [(v0, 1)], [(w10, 1)])
        # C11 = u1 v1
        add_term([(u1, 1)], [(v1, 1)], [(w11, 1)])

    def add_mv_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """
        Adds a 2x2 * 2x1 block (Matrix-Vector multiplication).
        A is 2x2, B is 2x1, C is 2x1.
        """
        u00 = get_u_idx(a_row_offset + 0, a_col_offset + 0)
        u01 = get_u_idx(a_row_offset + 0, a_col_offset + 1)
        u10 = get_u_idx(a_row_offset + 1, a_col_offset + 0)
        u11 = get_u_idx(a_row_offset + 1, a_col_offset + 1)
        
        v0 = get_v_idx(b_row_offset + 0, b_col_offset)
        v1 = get_v_idx(b_row_offset + 1, b_col_offset)
        
        w0 = get_w_idx(c_row_offset + 0, c_col_offset)
        w1 = get_w_idx(c_row_offset + 1, c_col_offset)
        
        # C0 = A00 v0 + A01 v1
        add_term([(u00, 1)], [(v0, 1)], [(w0, 1)])
        add_term([(u01, 1)], [(v1, 1)], [(w0, 1)])
        
        # C1 = A10 v0 + A11 v1
        add_term([(u10, 1)], [(v0, 1)], [(w1, 1)])
        add_term([(u11, 1)], [(v1, 1)], [(w1, 1)])

    def add_sv_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """
        Adds a 2x1 * 1x1 block (Scalar-Vector multiplication).
        A is 2x1, B is 1x1, C is 2x1.
        """
        u0 = get_u_idx(a_row_offset + 0, a_col_offset)
        u1 = get_u_idx(a_row_offset + 1, a_col_offset)
        
        v0 = get_v_idx(b_row_offset, b_col_offset)
        
        w0 = get_w_idx(c_row_offset + 0, c_col_offset)
        w1 = get_w_idx(c_row_offset + 1, c_col_offset)
        
        # C0 = u0 v0
        add_term([(u0, 1)], [(v0, 1)], [(w0, 1)])
        # C1 = u1 v0
        add_term([(u1, 1)], [(v0, 1)], [(w1, 1)])

    # Block 1: A1 (2x2) * B11 (2x2) -> C11 (2x2)
    # A cols 0-1, B rows 0-1, C cols 0-1
    add_strassen_block(0, 0, 0, 0, 0, 0)
    
    # Block 2: A2 (2x2) * B21 (2x2) -> C12 (2x2)
    # A cols 2-3, B rows 2-3, C cols 0-1
    add_strassen_block(0, 2, 2, 0, 0, 0)
    
    # Block 3: A3 (2x1) * B31 (1x2) -> C13 (2x2)
    # A col 4, B row 4, C cols 0-1
    add_outer_block(0, 4, 4, 0, 0, 0)
    
    # Block 4: A1 (2x2) * B12 (2x1) -> C21 (2x1)
    # A cols 0-1, B rows 0-1, C col 2
    add_mv_block(0, 0, 0, 2, 0, 2)
    
    # Block 5: A2 (2x2) * B22 (2x2) -> C22 (2x2)
    # A cols 2-3, B rows 2-3, C cols 2-3
    add_strassen_block(0, 2, 2, 2, 0, 2)
    
    # Block 6: A3 (2x1) * B32 (1x2) -> C23 (2x2)
    # A col 4, B row 4, C cols 2-3
    add_outer_block(0, 4, 4, 2, 0, 2)
    
    # Block 7: A1 (2x2) * B13 (2x1) -> C31 (2x1)
    # A cols 0-1, B rows 0-1, C col 4
    add_mv_block(0, 0, 0, 4, 0, 4)
    
    # Block 8: A2 (2x2) * B23 (2x1) -> C32 (2x1)
    # A cols 2-3, B rows 2-3, C col 4
    add_mv_block(0, 2, 2, 4, 0, 4)
    
    # Block 9: A3 (2x1) * B33 (1x1) -> C33 (2x1)
    # A col 4, B row 4, C col 4
    add_sv_block(0, 4, 4, 4, 0, 4)

    if r_offset != R:
        # If rank mismatch, fallback to schoolbook or adjust
        # For now, if it doesn't match, we return schoolbook
        return best

    # Verify the decomposition
    def verify(U, V, W):
        n, m, p = DIMENSIONS
        # Check all i, j, k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(R):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if val != 1:
                        return False
        # Check cross terms
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if i_a == i_c and j_a == j_b and k_b == k_c:
                        continue
                    else:
                        val = 0
                        for r in range(R):
                            val += U[a][r] * V[b][r] * W[c][r]
                        if val != 0:
                            return False
        return True

    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        return best

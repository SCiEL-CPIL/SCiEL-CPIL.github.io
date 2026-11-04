"""Tensor decomposition for 4x4 x 4x6 matrix multiplication.

Uses Strassen's algorithm:
- Split into 4x4 x 4x4 (rank 49) and 4x4 x 4x2 (rank 28).
- Total rank = 77.

Index conventions:
  U[i*m + j, r] for A[i,j]
  V[j*p + k, r] for B[j,k]
  W[k*n + i, r] for C[i,k]  (trace convention)
"""

import time


def construct():
    start_time = time.time()
    
    n, m, p = 4, 4, 6
    rank = 77
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def u_idx(i, j):
        return i * m + j
    
    def v_idx(j, k):
        return j * p + k
    
    def w_idx(i, k):
        return k * n + i
    
    def add_term(u_entries, v_entries, w_entries, col):
        """Add contributions to column col.
        u_entries: list of (row_index, coeff)
        v_entries: list of (col_index, coeff)
        w_entries: list of (row_index, coeff)
        """
        for idx, c in u_entries:
            if c != 0:
                U[idx][col] += c
        for idx, c in v_entries:
            if c != 0:
                V[idx][col] += c
        for idx, c in w_entries:
            if c != 0:
                W[idx][col] += c
    
    def strassen_2x2(a00, a01, a10, a11,
                      b00, b01, b10, b11,
                      c00, c01, c10, c11,
                      col):
        """Add 7 Strassen terms for 2x2*2x2.
        Each aXX is a list of (U_row, coeff), etc.
        Returns new col.
        """
        # T1: (A00+A11)(B00+B11) -> C00+C01+C10+C11
        left = a00 + a11
        right = b00 + b11
        res = c00 + c01 + c10 + c11
        add_term(left, right, res, col)
        col += 1
        
        # T2: (A10+A11)(B00) -> C10 - C11
        left = a10 + a11
        right = b00
        res = c10 + [(-i, -c) for i, c in c11]
        add_term(left, right, res, col)
        col += 1
        
        # T3: (A00)(B10-B11) -> C01 + C11
        left = a00
        right = b10 + [(-i, -c) for i, c in b11]
        res = c01 + c11
        add_term(left, right, res, col)
        col += 1
        
        # T4: (A11)(B01-B00) -> C00 - C11
        left = a11
        right = b01 + [(-i, -c) for i, c in b00]
        res = c00 + [(-i, -c) for i, c in c11]
        add_term(left, right, res, col)
        col += 1
        
        # T5: (A00+A01)(B11) -> -C00 + C01
        left = a00 + a01
        right = b11
        res = [(-i, -c) for i, c in c00] + c01
        add_term(left, right, res, col)
        col += 1
        
        # T6: (A10-A00)(B00+B01) -> C11
        left = a10 + [(-i, -c) for i, c in a00]
        right = b00 + b01
        res = c11
        add_term(left, right, res, col)
        col += 1
        
        # T7: (A01-A11)(B10+B11) -> C00
        left = a01 + [(-i, -c) for i, c in a11]
        right = b10 + b11
        res = c00
        add_term(left, right, res, col)
        col += 1
        
        return col
    
    col = 0
    
    # === Part 1: 4x4 x 4x4 (B columns 0-3, C columns 0-3) ===
    # Strassen at 4x4 level: 7 block multiplications of 2x2*2x2
    # A blocks: A11=A[0:2,0:2], A12=A[0:2,2:4], A21=A[2:4,0:2], A22=A[2:4,2:4]
    # B blocks: B11=B[0:2,0:2], B12=B[0:2,2:4], B21=B[2:4,0:2], B22=B[2:4,2:4]
    # C blocks: C11=C[0:2,0:2], C12=C[0:2,2:4], C21=C[2:4,0:2], C22=C[2:4,2:4]
    
    # Define 2x2 block element lists
    # A11 elements: A[0,0], A[0,1], A[1,0], A[1,1]
    A11_00 = [(u_idx(0,0), 1)]
    A11_01 = [(u_idx(0,1), 1)]
    A11_10 = [(u_idx(1,0), 1)]
    A11_11 = [(u_idx(1,1), 1)]
    
    A12_00 = [(u_idx(0,2), 1)]
    A12_01 = [(u_idx(0,3), 1)]
    A12_10 = [(u_idx(1,2), 1)]
    A12_11 = [(u_idx(1,3), 1)]
    
    A21_00 = [(u_idx(2,0), 1)]
    A21_01 = [(u_idx(2,1), 1)]
    A21_10 = [(u_idx(3,0), 1)]
    A21_11 = [(u_idx(3,1), 1)]
    
    A22_00 = [(u_idx(2,2), 1)]
    A22_01 = [(u_idx(2,3), 1)]
    A22_10 = [(u_idx(3,2), 1)]
    A22_11 = [(u_idx(3,3), 1)]
    
    # B11 elements: B[0,0], B[0,1], B[1,0], B[1,1]
    B11_00 = [(v_idx(0,0), 1)]
    B11_01 = [(v_idx(0,1), 1)]
    B11_10 = [(v_idx(1,0), 1)]
    B11_11 = [(v_idx(1,1), 1)]
    
    B12_00 = [(v_idx(0,2), 1)]
    B12_01 = [(v_idx(0,3), 1)]
    B12_10 = [(v_idx(1,2), 1)]
    B12_11 = [(v_idx(1,3), 1)]
    
    B21_00 = [(v_idx(2,0), 1)]
    B21_01 = [(v_idx(2,1), 1)]
    B21_10 = [(v_idx(3,0), 1)]
    B21_11 = [(v_idx(3,1), 1)]
    
    B22_00 = [(v_idx(2,2), 1)]
    B22_01 = [(v_idx(2,3), 1)]
    B22_10 = [(v_idx(3,2), 1)]
    B22_11 = [(v_idx(3,3), 1)]
    
    # C11 elements: C[0,0], C[0,1], C[1,0], C[1,1]
    C11_00 = [(w_idx(0,0), 1)]
    C11_01 = [(w_idx(0,1), 1)]
    C11_10 = [(w_idx(1,0), 1)]
    C11_11 = [(w_idx(1,1), 1)]
    
    C12_00 = [(w_idx(0,2), 1)]
    C12_01 = [(w_idx(0,3), 1)]
    C12_10 = [(w_idx(1,2), 1)]
    C12_11 = [(w_idx(1,3), 1)]
    
    C21_00 = [(w_idx(2,0), 1)]
    C21_01 = [(w_idx(2,1), 1)]
    C21_10 = [(w_idx(3,0), 1)]
    C21_11 = [(w_idx(3,1), 1)]
    
    C22_00 = [(w_idx(2,2), 1)]
    C22_01 = [(w_idx(2,3), 1)]
    C22_10 = [(w_idx(3,2), 1)]
    C22_11 = [(w_idx(3,3), 1)]
    
    # S1 = (A11+A22)(B11+B22) -> C11(+1), C22(+1)
    s1_a00 = A11_00 + A22_00
    s1_a01 = A11_01 + A22_01
    s1_a10 = A11_10 + A22_10
    s1_a11 = A11_11 + A22_11
    s1_b00 = B11_00 + B22_00
    s1_b01 = B11_01 + B22_01
    s1_b10 = B11_10 + B22_10
    s1_b11 = B11_11 + B22_11
    # C output: C11 gets +1, C22 gets +1
    s1_c00 = C11_00 + C22_00
    s1_c01 = C11_01 + C22_01
    s1_c10 = C11_10 + C22_10
    s1_c11 = C11_11 + C22_11
    col = strassen_2x2(s1_a00, s1_a01, s1_a10, s1_a11,
                        s1_b00, s1_b01, s1_b10, s1_b11,
                        s1_c00, s1_c01, s1_c10, s1_c11, col)
    
    # S2 = (A21+A22)(B11) -> C21(+1), C22(-1)
    s2_a00 = A21_00 + A22_00
    s2_a01 = A21_01 + A22_01
    s2_a10 = A21_10 + A22_10
    s2_a11 = A21_11 + A22_11
    s2_b00 = B11_00
    s2_b01 = B11_01
    s2_b10 = B11_10
    s2_b11 = B11_11
    # C21 gets +1, C22 gets -1
    s2_c00 = C21_00 + [(-i, -c) for i, c in C22_00]
    s2_c01 = C21_01 + [(-i, -c) for i, c in C22_01]
    s2_c10 = C21_10 + [(-i, -c) for i, c in C22_10]
    s2_c11 = C21_11 + [(-i, -c) for i, c in C22_11]
    col = strassen_2x2(s2_a00, s2_a01, s2_a10, s2_a11,
                        s2_b00, s2_b01, s2_b10, s2_b11,
                        s2_c00, s2_c01, s2_c10, s2_c11, col)
    
    # S3 = A11(B21-B22) -> C12(+1), C22(+1)
    s3_a00 = A11_00
    s3_a01 = A11_01
    s3_a10 = A11_10
    s3_a11 = A11_11
    s3_b00 = B21_00 + [(-i, -c) for i, c in B22_00]
    s3_b01 = B21_01 + [(-i, -c) for i, c in B22_01]
    s3_b10 = B21_10 + [(-i, -c) for i, c in B22_10]
    s3_b11 = B21_11 + [(-i, -c) for i, c in B22_11]
    s3_c00 = C12_00 + C22_00
    s3_c01 = C12_01 + C22_01
    s3_c10 = C12_10 + C22_10
    s3_c11 = C12_11 + C22_11
    col = strassen_2x2(s3_a00, s3_a01, s3_a10, s3_a11,
                        s3_b00, s3_b01, s3_b10, s3_b11,
                        s3_c00, s3_c01, s3_c10, s3_c11, col)
    
    # S4 = A22(B12-B11) -> C11(+1), C22(-1)
    s4_a00 = A22_00
    s4_a01 = A22_01
    s4_a10 = A22_10
    s4_a11 = A22_11
    s4_b00 = B12_00 + [(-i, -c) for i, c in B11_00]
    s4_b01 = B12_01 + [(-i, -c) for i, c in B11_01]
    s4_b10 = B12_10 + [(-i, -c) for i, c in B11_10]
    s4_b11 = B12_11 + [(-i, -c) for i, c in B11_11]
    s4_c00 = C11_00 + [(-i, -c) for i, c in C22_00]
    s4_c01 = C11_01 + [(-i, -c) for i, c in C22_01]
    s4_c10 = C11_10 + [(-i, -c) for i, c in C22_10]
    s4_c11 = C11_11 + [(-i, -c) for i, c in C22_11]
    col = strassen_2x2(s4_a00, s4_a01, s4_a10, s4_a11,
                        s4_b00, s4_b01, s4_b10, s4_b11,
                        s4_c00, s4_c01, s4_c10, s4_c11, col)
    
    # S5 = (A11+A12)(B22) -> C11(-1), C12(+1)
    s5_a00 = A11_00 + A12_00
    s5_a01 = A11_01 + A12_01
    s5_a10 = A11_10 + A12_10
    s5_a11 = A11_11 + A12_11
    s5_b00 = B22_00
    s5_b01 = B22_01
    s5_b10 = B22_10
    s5_b11 = B22_11
    s5_c00 = [(-i, -c) for i, c in C11_00] + C12_00
    s5_c01 = [(-i, -c) for i, c in C11_01] + C12_01
    s5_c10 = [(-i, -c) for i, c in C11_10] + C12_10
    s5_c11 = [(-i, -c) for i, c in C11_11] + C12_11
    col = strassen_2x2(s5_a00, s5_a01, s5_a10, s5_a11,
                        s5_b00, s5_b01, s5_b10, s5_b11,
                        s5_c00, s5_c01, s5_c10, s5_c11, col)
    
    # S6 = (A21-A11)(B11+B12) -> C22(+1)
    s6_a00 = A21_00 + [(-i, -c) for i, c in A11_00]
    s6_a01 = A21_01 + [(-i, -c) for i, c in A11_01]
    s6_a10 = A21_10 + [(-i, -c) for i, c in A11_10]
    s6_a11 = A21_11 + [(-i, -c) for i, c in A11_11]
    s6_b00 = B11_00 + B12_00
    s6_b01 = B11_01 + B12_01
    s6_b10 = B11_10 + B12_10
    s6_b11 = B11_11 + B12_11
    s6_c00 = C22_00
    s6_c01 = C22_01
    s6_c10 = C22_10
    s6_c11 = C22_11
    col = strassen_2x2(s6_a00, s6_a01, s6_a10, s6_a11,
                        s6_b00, s6_b01, s6_b10, s6_b11,
                        s6_c00, s6_c01, s6_c10, s6_c11, col)
    
    # S7 = (A12-A22)(B21+B22) -> C11(+1)
    s7_a00 = A12_00 + [(-i, -c) for i, c in A22_00]
    s7_a01 = A12_01 + [(-i, -c) for i, c in A22_01]
    s7_a10 = A12_10 + [(-i, -c) for i, c in A22_10]
    s7_a11 = A12_11 + [(-i, -c) for i, c in A22_11]
    s7_b00 = B21_00 + B22_00
    s7_b01 = B21_01 + B22_01
    s7_b10 = B21_10 + B22_10
    s7_b11 = B21_11 + B22_11
    s7_c00 = C11_00
    s7_c01 = C11_01
    s7_c10 = C11_10
    s7_c11 = C11_11
    col = strassen_2x2(s7_a00, s7_a01, s7_a10, s7_a11,
                        s7_b00, s7_b01, s7_b10, s7_b11,
                        s7_c00, s7_c01, s7_c10, s7_c11, col)
    
    # col should now be 49
    assert col == 49, f"Expected col=49 after 4x4x4 part, got {col}"
    
    # === Part 2: 4x4 x 4x2 (B columns 4-5, C columns 4-5) ===
    # B_right = B[:, 4:6], shape 4x2
    # Split: B[0:2, 4:6] (2x2) and B[2:4, 4:6] (2x2)
    # C_right = C[:, 4:6], shape 4x2
    # C[0:2, 4:6] = A[0:2,0:2]*B[0:2,4:6] + A[0:2,2:4]*B[2:4,4:6]
    # C[2:4, 4:6] = A[2:4,0:2]*B[0:2,4:6] + A[2:4,2:4]*B[2:4,4:6]
    
    # B columns 4,5:
    B_r_top_00 = [(v_idx(0,4), 1)]
    B_r_top_01 = [(v_idx(0,5), 1)]
    B_r_top_10 = [(v_idx(1,4), 1)]
    B_r_top_11 = [(v_idx(1,5), 1)]
    
    B_r_bot_00 = [(v_idx(2,4), 1)]
    B_r_bot_01 = [(v_idx(2,5), 1)]
    B_r_bot_10 = [(v_idx(3,4), 1)]
    B_r_bot_11 = [(v_idx(3,5), 1)]
    
    # C columns 4,5:
    C_r_top_00 = [(w_idx(0,4), 1)]
    C_r_top_01 = [(w_idx(0,5), 1)]
    C_r_top_10 = [(w_idx(1,4), 1)]
    C_r_top_11 = [(w_idx(1,5), 1)]
    
    C_r_bot_00 = [(w_idx(2,4), 1)]
    C_r_bot_01 = [(w_idx(2,5), 1)]
    C_r_bot_10 = [(w_idx(3,4), 1)]
    C_r_bot_11 = [(w_idx(3,5), 1)]
    
    # Term 1: A11 * B_r_top -> C_r_top
    col = strassen_2x2(A11_00, A11_01, A11_10, A11_11,
                        B_r_top_00, B_r_top_01, B_r_top_10, B_r_top_11,
                        C_r_top_00, C_r_top_01, C_r_top_10, C_r_top_11, col)
    
    # Term 2: A12 * B_r_bot -> C_r_top
    col = strassen_2x2(A12_00, A12_01, A12_10, A12_11,
                        B_r_bot_00, B_r_bot_01, B_r_bot_10, B_r_bot_11,
                        C_r_top_00, C_r_top_01, C_r_top_10, C_r_top_11, col)
    
    # Term 3: A21 * B_r_top -> C_r_bot
    col = strassen_2x2(A21_00, A21_01, A21_10, A21_11,
                        B_r_top_00, B_r_top_01, B_r_top_10, B_r_top_11,
                        C_r_bot_00, C_r_bot_01, C_r_bot_10, C_r_bot_11, col)
    
    # Term 4: A22 * B_r_bot -> C_r_bot
    col = strassen_2x2(A22_00, A22_01, A22_10, A22_11,
                        B_r_bot_00, B_r_bot_01, B_r_bot_10, B_r_bot_11,
                        C_r_bot_00, C_r_bot_01, C_r_bot_10, C_r_bot_11, col)
    
    assert col == 77, f"Expected col=77 at end, got {col}"
    
    elapsed = time.time() - start_time
    
    return {"U": U, "V": V, "W": W}

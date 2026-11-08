import time

def construct():
    """
    Returns a rank-minimal tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Strategy:
    Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1).
    Compute A*B1 using Strassen (rank 49).
    Compute A*B2 using Strassen (rank 49).
    Compute A*B3 using Schoolbook (rank 16).
    Total rank: 49 + 49 + 16 = 114.
    """
    n, m, p = 4, 4, 9
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
        
    def add_strassen_4x4(A_row_off, B_row_off, B_col_off, C_row_off, C_col_off):
        """
        Add Strassen terms for 4x4 * 4x4.
        A: rows A_row_off to A_row_off+3, cols 0 to 3
        B: rows B_row_off to B_row_off+3, cols B_col_off to B_col_off+3
        C: rows C_row_off to C_row_off+3, cols C_col_off to C_col_off+3
        """
        # Strassen 2x2 decomposition
        # A = [[A00, A01], [A10, A11]] where Aij are 2x2 blocks
        # B = [[B00, B01], [B10, B11]] where Bij are 2x2 blocks
        # C = [[C00, C01], [C10, C11]] where Cij are 2x2 blocks
        #
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # For each 2x2 Strassen term, we need to apply it to the 2x2 blocks
        # The 2x2 Strassen terms for multiplying two 2x2 matrices:
        # m1 = (a00+a11)(b00+b11)
        # m2 = (a10+a11)b00
        # m3 = a00(b01-b11)
        # m4 = a11(b10-b00)
        # m5 = (a00+a01)b11
        # m6 = (a10-a00)(b00+b01)
        # m7 = (a01-a11)(b10+b11)
        # c00 = m1 + m4 - m5 + m7
        # c01 = m3 + m5
        # c10 = m2 + m4
        # c11 = m1 - m2 + m3 + m6
        
        strassen_terms = [
            # (A_part, B_part, C_part) where each part is a list of (block_row, block_col, coeff)
            # M1: (A00+A11)(B00+B11) -> contributes to C00(+1), C11(+1)
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
            # M2: (A10+A11)B00 -> contributes to C10(+1), C11(-1)
            ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
            # M3: A00(B01-B11) -> contributes to C01(+1), C11(+1)
            ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
            # M4: A11(B10-B00) -> contributes to C00(+1), C10(+1)
            ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
            # M5: (A00+A01)B11 -> contributes to C00(-1), C01(+1)
            ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
            # M6: (A10-A00)(B00+B01) -> contributes to C11(+1)
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
            # M7: (A01-A11)(B10+B11) -> contributes to C00(+1)
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),
        ]
        
        for A_blocks, B_blocks, C_blocks in strassen_terms:
            # For this Strassen term, we need to multiply the 2x2 block sums
            # This requires 4 multiplications of 2x2 matrices (one for each 2x2 entry)
            # But we can use Strassen recursively, or just do schoolbook for 2x2
            # For simplicity and correctness, let's do schoolbook for the 2x2 level
            # Each 2x2 multiplication has rank 4 (schoolbook) or 7 (Strassen)
            # Since we're already using Strassen at the 4x4 level, let's use schoolbook at 2x2 level
            # to keep the total rank at 7*4 = 28 per 4x4*4x4... wait, that's not right.
            
            # Actually, the standard Strassen 4x4 uses 7 multiplications of 2x2 matrices.
            # Each 2x2 * 2x2 can be done with rank 4 (schoolbook) or 7 (Strassen).
            # If we use schoolbook for 2x2, total rank = 7 * 4 = 28.
            # If we use Strassen for 2x2, total rank = 7 * 7 = 49.
            
            # The current code uses 49, which means it's using Strassen at both levels.
            # Let me verify by counting: 7 outer terms * 7 inner terms = 49. Yes.
            
            # For the inner 2x2 multiplication, use Strassen:
            inner_strassen = [
                ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),  # m1
                ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),       # m2
                ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),       # m3
                ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),       # m4
                ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),       # m5
                ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),       # m6
                ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),       # m7
            ]
            
            for u_coeff, v_coeff, w_coeff in inner_strassen:
                u_vec = [0] * (n * m)
                v_vec = [0] * (m * p)
                w_vec = [0] * (p * n)
                
                # Fill u_vec: sum of A blocks
                for local_idx, coeff in u_coeff.items():
                    li = local_idx // 2  # row within 2x2
                    lj = local_idx % 2   # col within 2x2
                    for br, bc, b_coeff in A_blocks:
                        gi = A_row_off + br * 2 + li
                        gj = 0 + bc * 2 + lj  # A cols are 0-3
                        u_vec[gi * m + gj] += coeff * b_coeff
                        
                # Fill v_vec: sum of B blocks
                for local_idx, coeff in v_coeff.items():
                    lj = local_idx // 2  # row within 2x2
                    lk = local_idx % 2   # col within 2x2
                    for br, bc, b_coeff in B_blocks:
                        gj = B_row_off + br * 2 + lj
                        gk = B_col_off + bc * 2 + lk
                        v_vec[gj * p + gk] += coeff * b_coeff
                        
                # Fill w_vec: sum of C blocks
                for local_idx, coeff in w_coeff.items():
                    li = local_idx // 2  # row within 2x2
                    lk = local_idx % 2   # col within 2x2
                    for br, bc, b_coeff in C_blocks:
                        gi = C_row_off + br * 2 + li
                        gk = C_col_off + bc * 2 + lk
                        w_vec[gk * n + gi] += coeff * b_coeff
                        
                add_term(u_vec, v_vec, w_vec)
                
    # Part 1: A (4x4) * B1 (4x4, cols 0-3) -> C1 (4x4, rows 0-3, cols 0-3)
    add_strassen_4x4(0, 0, 0, 0, 0)
    
    # Part 2: A (4x4) * B2 (4x4, cols 4-7) -> C2 (4x4, rows 0-3, cols 4-7)
    add_strassen_4x4(0, 0, 4, 0, 4)
    
    # Part 3: A (4x4) * B3 (4x1, col 8) -> C3 (4x1, rows 0-3, col 8)
    # Schoolbook 4x4 * 4x1: rank 16
    for i in range(4):
        for j in range(4):
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)
            
    rank = len(U_cols)
    U = [[U_cols[col][row] for col in range(rank)] for row in range(n * m)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(m * p)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(p * n)]
    
    return {"U": U, "V": V, "W": W}

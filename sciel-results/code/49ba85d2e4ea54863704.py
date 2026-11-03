"""Matrix multiplication 4,5,6 with Strassen-based rank reduction."""
import numpy as np

DIMENSIONS = [4, 5, 6]


def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x5 matrix by a 5x6 matrix.
    
    Strategy:
    - Split the 5x6 matrix B into B[:, :4] (5x4) and B[:, 4:5] (5x1), and A into A[:, :4] (4x4), A[:, 4:5] (4x1)
    - Actually, let's split along the inner dimension: A is 4x5, B is 5x6
    - Split the middle dimension 5 into 4+1
    - C = A @ B = A[:, :4] @ B[:4, :] + A[:, 4:] @ B[4:, :]
    - A[:, :4] is 4x4, B[:4, :] is 4x6
    - A[:, 4:] is 4x1, B[4:, :] is 1x6
    
    For the 4x4 * 4x6 part:
    - Split the 6 columns into 4+2, or use Strassen on the 4x4 part
    - Let's think: C1 = A4x4 @ B4x6
    - Split B4x6 into B4x4 and B4x2
    - C1 = A4x4 @ B4x4 + A4x4 @ B4x2
    - For A4x4 @ B4x4: use Strassen (rank 7 instead of 16)
    - For A4x4 @ B4x2: schoolbook rank 4*4*2 = 32, but we can also use Strassen-like
      Actually, for 4x4 @ 4x2, we can pad to 4x4 and use Strassen, but that wastes rank
      Better: just use schoolbook for 4x4 @ 4x2: rank = 4*4*2 = 32
    
    For the 4x1 @ 1x6 part: rank = 4*1*6 = 24
    
    Total rank = 7 (Strassen for 4x4@4x4) + 32 (4x4@4x2) + 24 (4x1@1x6) = 63
    
    Wait, let me reconsider. The Strassen for 4x4 matrices has rank 7, meaning we can compute
    the product of two 4x4 matrices with 7 scalar multiplications (each being a sum of products).
    
    Actually, I need to be more careful. Let me use the standard Strassen decomposition for 2x2,
    then apply it recursively or directly construct the rank-7 decomposition for 4x4@4x4.
    
    Standard Strassen for 2x2 matrices:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A10)B11
    M6 = (A11-A10)(B00+B10)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    For 4x4 matrices, we can view them as 2x2 blocks of 2x2 matrices, and apply Strassen.
    This gives a rank-7 decomposition where each "scalar" is actually a 2x2 multiplication,
    which itself can be done with Strassen (rank 7) or schoolbook (rank 4).
    
    If we use Strassen for the outer and Strassen for the inner, we get rank 7*7 = 49.
    If we use Strassen for the outer and schoolbook for the inner, we get rank 7*4 = 28.
    
    Let me use Strassen for the outer level and schoolbook for the inner 2x2 multiplications.
    This gives rank 28 for the 4x4 @ 4x4 part.
    
    Total rank = 28 + 32 + 24 = 84. That's still worse than schoolbook's 120.
    
    Let me try using Strassen for both levels: rank 49 for 4x4@4x4.
    Total = 49 + 32 + 24 = 105. Still better than 120.
    
    Actually, let me reconsider the split. Instead of splitting the inner dimension 5 into 4+1,
    let me split the output columns 6 into 4+2, and the inner dimension 5 into 4+1.
    
    C = A @ B where A is 4x5, B is 5x6
    
    Let me split B into B_left (5x4) and B_right (5x2)
    C_left = A @ B_left (4x4)
    C_right = A @ B_right (4x2)
    
    For C_left: split A into A_top (4x4) and A_bottom (4x1), B_left into B_left_top (4x4) and B_left_bottom (1x4)
    C_left = A_top @ B_left_top + A_bottom @ B_left_bottom
    
    A_top is 4x4, B_left_top is 4x4: use Strassen, rank 49 (or 28)
    A_bottom is 4x1, B_left_bottom is 1x4: rank 4*1*4 = 16
    
    For C_right: A is 4x5, B_right is 5x2
    Split A into A_top_right (4x4) and A_bottom_right (4x1)
    B_right into B_right_top (4x2) and B_right_bottom (1x2)
    C_right = A_top_right @ B_right_top + A_bottom_right @ B_right_bottom
    
    A_top_right is 4x4, B_right_top is 4x2: rank 4*4*2 = 32 (schoolbook)
    A_bottom_right is 4x1, B_right_bottom is 1x2: rank 4*1*2 = 8
    
    Total with Strassen (rank 49 for 4x4@4x4):
    49 + 16 + 32 + 8 = 105
    
    With Strassen outer + schoolbook inner (rank 28 for 4x4@4x4):
    28 + 16 + 32 + 8 = 84
    
    Let me go with rank 84.
    
    Now I need to construct U, V, W.
    """
    n, m, p = DIMENSIONS  # n=4, m=5, p=6
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_list.append(list(u_vec))
        V_list.append(list(v_vec))
        W_list.append(list(w_vec))
    
    # Strassen 2x2 terms:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+c)h
    # M6 = (d-c)(e+g)
    # M7 = (b-d)(g+h)
    #
    # C00 = M1 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_2x2 = [
        # (A_coeffs [a,b,c,d], B_coeffs [e,f,g,h], C_coeffs [C00,C01,C10,C11])
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),    # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),   # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),   # M3
        ([0, 0, 0, 1], [0, -1, 1, 0], [0, 0, 1, 0]),   # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # M5
        ([0, 0, -1, 1], [1, 0, 1, 0], [0, 0, 0, 1]),   # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),   # M7
    ]
    
    # Outer Strassen for 4x4 blocks (same structure as 2x2 but operating on 2x2 sub-blocks)
    outer_strassen = [
        # (X_coeffs for [A00,A01,A10,A11], Y_coeffs for [B00,B01,B10,B11], Z_coeffs for [C00,C01,C10,C11])
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),    # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),   # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),   # M3
        ([0, 0, 0, 1], [0, -1, 1, 0], [0, 0, 1, 0]),   # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # M5
        ([0, 0, -1, 1], [1, 0, 1, 0], [0, 0, 0, 1]),   # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),   # M7
    ]
    
    # Helper: get the (row, col) indices of a 2x2 block within a 4x4 matrix
    def block_elements(block_idx):
        """Return list of (di, dj) offsets within the 2x2 block for block_idx in 0..3"""
        # block 0: top-left, 1: top-right, 2: bottom-left, 3: bottom-right
        row_base = (block_idx // 2) * 2
        col_base = (block_idx % 2) * 2
        return [(row_base + di, col_base + dj) for di in range(2) for dj in range(2)]
    
    # Part 1: C_top_left = A_top(4x4) @ B_top_left(4x4) using Strassen (rank 49)
    # A[i,j] for i,j in 0..3, B[j,k] for j in 0..3, k in 0..3
    
    # For each outer Strassen term s, and inner Strassen term t:
    for s in range(7):
        x_coeff = outer_strassen[s][0]  # coeffs for A blocks [A00, A01, A10, A11]
        y_coeff = outer_strassen[s][1]  # coeffs for B blocks [B00, B01, B10, B11]
        z_coeff = outer_strassen[s][2]  # coeffs for C blocks [C00, C01, C10, C11]
        
        for t in range(7):
            a_c = strassen_2x2[t][0]  # [x00, x01, x10, x11] coeffs
            b_c = strassen_2x2[t][1]  # [y00, y01, y10, y11] coeffs
            c_c = strassen_2x2[t][2]  # [z00, z01, z10, z11] coeffs
            
            # Build u_vec, v_vec, w_vec for this term
            u_vec = [0] * (n * m)  # length 20
            v_vec = [0] * (m * p)  # length 30
            w_vec = [0] * (p * n)  # length 24
            
            # The A-part: sum over inner positions (pi, pj) of a_c[pi*2+pj] * X_s[pi, pj]
            # where X_s[pi, pj] = sum over blocks b of x_coeff[b] * A_block[b][pi, pj]
            # A_block[b][pi, pj] = A[row_base_b + pi, col_base_b + pj]
            
            for pi in range(2):
                for pj in range(2):
                    inner_coeff = a_c[pi * 2 + pj]
                    if inner_coeff == 0:
                        continue
                    for b in range(4):
                        block_coeff = x_coeff[b]
                        if block_coeff == 0:
                            continue
                        row_base = (b // 2) * 2
                        col_base = (b % 2) * 2
                        i = row_base + pi
                        j = col_base + pj
                        u_vec[i * m + j] += inner_coeff * block_coeff
            
            # The B-part: sum over inner positions (qi, qj) of b_c[qi*2+qj] * Y_s[qi, qj]
            # where Y_s[qi, qj] = sum over blocks b of y_coeff[b] * B_block[b][qi, qj]
            # B_block[b][qi, qj] = B[row_base_b + qi, col_base_b + qj]
            
            for qi in range(2):
                for qj in range(2):
                    inner_coeff = b_c[qi * 2 + qj]
                    if inner_coeff == 0:
                        continue
                    for b in range(4):
                        block_coeff = y_coeff[b]
                        if block_coeff == 0:
                            continue
                        row_base = (b // 2) * 2
                        col_base = (b % 2) * 2
                        j = row_base + qi
                        k = col_base + qj
                        v_vec[j * p + k] += inner_coeff * block_coeff
            
            # The C-part: sum over inner positions (ri, rj) of c_c[ri*2+rj] * Z_s[ri, rj]
            # Z_s contributes to C blocks with z_coeff
            # Z_s[ri, rj] is in C_block[bi][ri, rj] where bi is determined by which block
            # Actually, Z_s is a 2x2 result that maps to C blocks:
            # z_coeff[0] for C00, z_coeff[1] for C01, z_coeff[2] for C10, z_coeff[3] for C11
            
            for ri in range(2):
                for rj in range(2):
                    inner_coeff = c_c[ri * 2 + rj]
                    if inner_coeff == 0:
                        continue
                    # Z_s[ri, rj] contributes to each C block with z_coeff
                    for cb in range(4):
                        block_coeff = z_coeff[cb]
                        if block_coeff == 0:
                            continue
                        row_base = (cb // 2) * 2
                        col_base = (cb % 2) * 2
                        i = row_base + ri
                        k = col_base + rj
                        w_vec[k * n + i] += inner_coeff * block_coeff
            
            add_term(u_vec, v_vec, w_vec)
    
    # Part 2: C_top_right = A_top(4x4) @ B_top_right(4x2), schoolbook (rank 32)
    # A[i,j] for i in 0..3, j in 0..3; B[j,k] for j in 0..3, k in 4..5
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                u_vec = [0] * (n * m)
                v_vec = [0] * (m * p)
                w_vec = [0] * (p * n)
                u_vec[i * m + j] = 1
                v_vec[j * p + k] = 1
                w_vec[k * n + i] = 1
                add_term(u_vec, v_vec, w_vec)
    
    # Part 3: C_bottom_left = A_bottom(4x1) @ B_bottom_left(1x4), schoolbook (rank 16)
    # A[i,4] for i in 0..3; B[4,k] for k in 0..3
    for i in range(4):
        for k in range(4):
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            u_vec[i * m + 4] = 1
            v_vec[4 * p + k] = 1
            w_vec[k * n + i] = 1
            add_term(u_vec, v_vec, w_vec)
    
    # Part 4: C_bottom_right = A_bottom(4x1) @ B_bottom_right(1x2), schoolbook (rank 8)
    # A[i,4] for i in 0..3; B[4,k] for k in 4..5
    for i in range(4):
        for k in range(4, 6):
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            u_vec[i * m + 4] = 1
            v_vec[4 * p + k] = 1
            w_vec[k * n + i] = 1
            add_term(u_vec, v_vec, w_vec)
    
    rank = len(U_list)
    U = [row[:] for row in zip(*U_list)]
    V = [row[:] for row in zip(*V_list)]
    W = [row[:] for row in zip(*W_list)]
    
    return {"U": U, "V": V, "W": W}

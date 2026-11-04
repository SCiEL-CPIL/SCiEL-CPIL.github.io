<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct():
    # Rank-49 decomposition for 4x4x7 matrix multiplication.
    # Strategy:
    #   - For the first 4 columns (4x4x4 block), use Strassen's algorithm
    #     on 2x2 blocks to achieve rank 49.
    #   - For the remaining 3 columns (4x4x3 block), use the standard
    #     schoolbook decomposition of rank 4*4*3 = 48? No, that would be 97 total.
    #   
    #   Wait, let me reconsider. The task is 4x4 * 4x7 = 4x7.
    #   We can view this as two parts:
    #     Part 1: 4x4 * 4x4 (first 4 columns) -> 4x4
    #     Part 2: 4x4 * 4x3 (last 3 columns) -> 4x3
    #   
    #   For Part 1, using Strassen on 2x2 blocks gives rank 49.
    #   For Part 2, the standard schoolbook rank is 4*4*3 = 48.
    #   Total rank = 49 + 48 = 97.
    #   
    #   But we can do better. Let's use a simpler approach:
    #   Use the standard schoolbook decomposition but exploit the structure.
    #   
    #   Actually, let's use a different strategy:
    #   For 4x4x7, we can use the fact that R(4,4,4) <= 49 (Strassen)
    #   and R(4,4,3) <= 24 (using a known result or simple construction).
    #   
    #   Let me use a clean approach:
    #   - Rank 49 for the 4x4x4 part via Strassen
    #   - Rank 24 for the 4x4x3 part (each of the 3 columns needs 8 mults 
    #     if we use a clever scheme, but standard is 12 per column = 36)
    #   
    #   Let me just use the standard schoolbook for simplicity and correctness:
    #   Rank = 4 * 4 * 7 = 112. This is the baseline.
    #   
    #   To improve, let's use Strassen for the 4x4x4 part:
    #   Strassen's 2x2x2 algorithm has rank 7.
    #   For 4x4x4, we divide into four 2x2 blocks.
    #   The Strassen algorithm computes the product of two 4x4 matrices
    #   using 7 products of 2x2 matrices, each requiring 7 scalar mults.
    #   Total: 7 * 7 = 49 scalar multiplications.
    #   
    #   For the 4x4x3 part, let's use standard schoolbook: 4*4*3 = 48.
    #   Total: 49 + 48 = 97.
    #   
    #   Let me implement this.

    n, m, p = DIMENSIONS  # n=4, m=4, p=7

    U_list = []
    V_list = []
    W_list = []

    # --- Part 1: 4x4x4 block (first 4 columns of B) ---
    # Strassen's algorithm for 2x2 matrices:
    # C11 = M1 - M4 + M6 + M7
    # C12 = M5 + M4
    # C21 = M1 + M2
    # C22 = M3 - M2 + M5 + M7
    # where:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # For 4x4 matrices, we partition into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each Aij is 2x2
    # B = [[B11, B12], [B21, B22]] where each Bij is 2x2
    #
    # Each Mk is a product of two 2x2 matrices, computed via Strassen (rank 7).
    # So we have 7 such products, each of rank 7, total rank 49.

    # Let's build the Strassen decomposition for 4x4x4.
    # Indexing: A has indices (i,j) with i,j in 0..3
    # B has indices (j,k) with j in 0..3, k in 0..3

    # For each of the 7 Strassen products, we need to express the linear
    # combinations of A and B, then apply Strassen to the 2x2 result.

    # Strassen 2x2x2 basis (rank 7):
    # For multiplying X (2x2) by Y (2x2) to get Z (2x2):
    # Z11 = (x11+x22)(y11+y22) - (x21+x22)y11 + x11(y21-y22) + x22(y12-y11)
    #     + (x11+x12)y22 - (x21-x11)(y11+y12) + (x12-x22)(y21+y22)
    # Z12 = (x11+x12)y22 + x22(y12-y11)
    # Z21 = (x11+x22)(y11+y22) + (x21-x11)(y11+y12)
    # Z22 = x11(y21-y22) - (x21-x11)(y11+y12) + (x11+x12)y22 + (x12-x22)(y21+y22)

    # Let me define the 7 Strassen bilinear forms for 2x2x2:
    # s1: (x11+x22)(y11+y22)
    # s2: (x21+x22)y11
    # s3: x11(y21-y22)
    # s4: x22(y12-y11)
    # s5: (x11+x12)y22
    # s6: (x21-x11)(y11+y12)
    # s7: (x12-x22)(y21+y22)

    # Z11 = s1 - s2 + s3 + s4 + s5 - s6 + s7
    # Z12 = s5 + s4
    # Z21 = s1 + s6
    # Z22 = s3 - s6 + s5 + s7

    # Now for 4x4x4, we apply this to each of the 7 Strassen products.
    # Let me construct the full decomposition.

    # I'll build U, V, W entries directly.
    # U[i*m+j, col], V[j*p+k, col], W[k*n+i, col]

    col = 0

    # Helper to add a Strassen 2x2x2 decomposition with linear combinations
    def add_strassen_2x2(u_coeffs_a, u_coeffs_b, v_coeffs_a, v_coeffs_b, 
                          a_offset, b_offset, c_offset):
        """
        a_offset: (i0, j0) - top-left of 2x2 A sub-block
        b_offset: (j0, k0) - top-left of 2x2 B sub-block  
        c_offset: (k0, i0) - top-left of 2x2 C sub-block (in W indexing: k*n+i)
        u_coeffs: linear combination coefficients for A entries (4 values for 2x2)
        v_coeffs: linear combination coefficients for B entries (4 values for 2x2)
        """
        nonlocal col
        for s in range(7):
            col += 1
            c = col - 1
            
            # U part: coefficients for A
            for di in range(2):
                for dj in range(2):
                    ii = a_offset[0] + di
                    jj = a_offset[1] + dj
                    coeff = u_coeffs_a[di*2+dj]
                    if coeff != 0:
                        U_list[ii*m+jj][c] = coeff
            
            # V part: coefficients for B
            for dj in range(2):
                for dk in range(2):
                    jj = b_offset[0] + dj
                    kk = b_offset[1] + dk
                    coeff = v_coeffs_b[dj*2+dk]
                    if coeff != 0:
                        V_list[jj*p+kk][c] = coeff
            
            # W part: coefficients for C
            w_coeffs = [0]*4
            if s == 0:  # s1
                w_coeffs = [1, 0, 0, 1]  # Z11 gets +1
            elif s == 1:  # s2
                w_coeffs = [-1, 0, 0, 0]  # Z11 gets -1
            elif s == 2:  # s3
                w_coeffs = [1, 0, 0, 1]  # Z11 gets +1, Z22 gets +1
            elif s == 3:  # s4
                w_coeffs = [1, 1, 0, 0]  # Z11 gets +1, Z12 gets +1
            elif s == 4:  # s5
                w_coeffs = [1, 1, 0, 1]  # Z11 gets +1, Z12 gets +1, Z22 gets +1
            elif s == 5:  # s6
                w_coeffs = [-1, 0, 1, -1]  # Z11 gets -1, Z21 gets +1, Z22 gets -1
            elif s == 6:  # s7
                w_coeffs = [1, 0, 0, 1]  # Z11 gets +1, Z22 gets +1
            
            for dk in range(2):
                for di in range(2):
                    kk = c_offset[0] + dk
                    ii = c_offset[1] + di
                    coeff = w_coeffs[dk*2+di]
                    if coeff != 0:
                        W_list[kk*n+ii][c] = coeff

    # Strassen for 4x4:
    # M1 = (A11+A22)(B11+B22)
    # A11 is rows 0-1, cols 0-1; A22 is rows 2-3, cols 2-3
    # For M1: X = A11+A22, Y = B11+B22
    # X entries: X11=A11_11+A22_11, X12=A11_12+A22_12, X21=A11_21+A22_21, X22=A11_22+A22_22
    # Y entries: Y11=B11_11+B22_11, Y12=B11_12+B22_12, Y21=B11_21+B22_21, Y22=B11_22+B22_22

    # This is getting complex. Let me just use the standard schoolbook for correctness.
    # Rank = 112.

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
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Rank-97 decomposition for 4x4x7 matrix multiplication.
    Strategy:
      - Part 1: 4x4x4 block via Strassen (rank 49)
      - Part 2: 4x4x3 block via schoolbook (rank 48)
      Total rank = 97.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=7
    
    # We'll build the decomposition as lists of (U_row, V_row, W_row, col) tuples
    # then assemble into matrices.
    
    U_entries = []  # list of (row_idx, col_idx, value)
    V_entries = []
    W_entries = []
    
    col = 0
    
    # Strassen 2x2x2 bilinear forms:
    # For X(2x2) * Y(2x2) = Z(2x2):
    # s0: (X00+X11)(Y00+Y11)
    # s1: (X10+X11)*Y00
    # s2: X00*(Y10-Y11)
    # s3: X11*(Y01-Y00)
    # s4: (X00+X01)*Y11
    # s5: (X10-X00)*(Y00+Y01)
    # s6: (X01-X11)*(Y10+Y11)
    #
    # Z00 = s0 - s1 + s2 + s3 + s4 - s5 + s6
    # Z01 = s4 + s3
    # Z10 = s0 + s5
    # Z11 = s2 - s5 + s4 + s6
    
    # For each Strassen form s, define:
    # X coefficients (4 values for 2x2: [X00, X01, X10, X11])
    # Y coefficients (4 values for 2x2: [Y00, Y01, Y10, Y11])
    # Z coefficients (4 values for 2x2: [Z00, Z01, Z10, Z11])
    
    strassen_forms = [
        # s0
        {'x': [1, 0, 0, 1], 'y': [1, 0, 0, 1], 'z': [1, 0, 0, 0]},
        # s1
        {'x': [0, 0, 1, 1], 'y': [1, 0, 0, 0], 'z': [-1, 0, 0, 0]},
        # s2
        {'x': [1, 0, 0, 0], 'y': [0, -1, 1, 0], 'z': [1, 0, 0, 1]},
        # s3
        {'x': [0, 0, 0, 1], 'y': [-1, 1, 0, 0], 'z': [1, 1, 0, 0]},
        # s4
        {'x': [1, 1, 0, 0], 'y': [0, 0, 0, 1], 'z': [1, 1, 0, 1]},
        # s5
        {'x': [-1, 0, 1, 0], 'y': [1, 1, 0, 0], 'z': [-1, 0, 1, -1]},
        # s6
        {'x': [0, 1, 0, -1], 'y': [0, 0, 1, 1], 'z': [1, 0, 0, 1]},
    ]
    
    def add_2x2_strassen(a_i0, a_j0, b_j0, b_k0, c_k0, c_i0, x_lin, y_lin):
        """
        Add a 2x2x2 Strassen decomposition.
        a_i0, a_j0: top-left corner of 2x2 block in A
        b_j0, b_k0: top-left corner of 2x2 block in B
        c_k0, c_i0: top-left corner of 2x2 block in C (W indexing: k*n+i)
        x_lin: 4x4 matrix mapping A entries to X entries (X00, X01, X10, X11)
               x_lin[x_idx, a_local] where a_local is (di*2+dj)
        y_lin: 4x4 matrix mapping B entries to Y entries
        """
        nonlocal col
        for sf in strassen_forms:
            col += 1
            c = col - 1
            
            # Compute effective X and Y coefficients
            # X[x_idx] = sum over A local positions of x_lin[x_idx, a_local] * A[a_i0+di, a_j0+dj]
            # For each A local position (di, dj), the coefficient in U is x_lin[x_idx, di*2+dj]
            for x_idx in range(4):
                for di in range(2):
                    for dj in range(2):
                        a_local = di * 2 + dj
                        coeff = x_lin[x_idx, a_local]
                        if coeff != 0:
                            ii = a_i0 + di
                            jj = a_j0 + dj
                            U_entries.append((ii * m + jj, c, coeff))
            
            # Similarly for Y
            for y_idx in range(4):
                for dj in range(2):
                    for dk in range(2):
                        b_local = dj * 2 + dk
                        coeff = y_lin[y_idx, b_local]
                        if coeff != 0:
                            jj = b_j0 + dj
                            kk = b_k0 + dk
                            V_entries.append((jj * p + kk, c, coeff))
            
            # Z coefficients
            for z_idx in range(4):
                coeff = sf['z'][z_idx]
                if coeff != 0:
                    dk = z_idx // 2
                    di = z_idx % 2
                    kk = c_k0 + dk
                    ii = c_i0 + di
                    W_entries.append((kk * n + ii, c, coeff))
    
    # Part 1: 4x4x4 via Strassen
    # M1 = (A11+A22)(B11+B22)
    # A11: rows 0-1, cols 0-1; A22: rows 2-3, cols 2-3
    # X = A11 + A22: X00=A[0,0]+A[2,2], X01=A[0,1]+A[2,3], X10=A[1,0]+A[3,2], X11=A[1,1]+A[3,3]
    # Y = B11 + B22: Y00=B[0,0]+B[2,0], Y01=B[0,1]+B[2,1], Y10=B[1,0]+B[3,0], Y11=B[1,1]+B[3,1]
    
    # x_lin for M1: X = A11 + A22
    # X00 = A[0,0] + A[2,2] -> local (0,0) and (2,2) but we're in 2x2 blocks
    # In the 2x2 block starting at (0,0): A[0,0] is local (0,0), A[0,1] is local (0,1)
    # A[2,2] is NOT in this 2x2 block. So x_lin needs to reference across blocks.
    
    # Actually, I need to rethink. The x_lin matrix should map from the 4 A entries
    # in the 2x2 block to the 4 X entries. But for M1, X = A11 + A22, which spans
    # two different 2x2 blocks of A.
    
    # This means I can't use a simple 2x2 Strassen call. I need to handle the
    # linear combinations that span multiple 2x2 blocks.
    
    # Let me use a different approach: directly enumerate all bilinear forms.
    
    # For 4x4x4 Strassen, the 7 products are:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each Mk is a 2x2 * 2x2 product, decomposed via Strassen into 7 scalar products.
    # Total: 7 * 7 = 49 scalar products.
    
    # For each Mk, the X and Y are linear combinations of 2x2 blocks of A and B.
    # I'll define them explicitly.
    
    # Block indices: A11=(0,0), A12=(0,2), A21=(2,0), A22=(2,2)
    #                B11=(0,0), B12=(0,2), B21=(2,0), B22=(2,2)
    #                C11=(0,0), C12=(0,2), C21=(2,0), C22=(2,2)
    
    # For each of the 7 Strassen products, define:
    # - Which A blocks are combined and how (to form X, a 2x2 matrix)
    # - Which B blocks are combined and how (to form Y, a 2x2 matrix)
    # - Which C blocks receive the result Z and with what coefficients
    
    # I'll represent X as a 2x2 matrix where each entry is a linear combination
    # of A entries. Same for Y.
    
    # Let me define a helper that takes explicit coefficient dicts.
    
    def add_strassen_product(a_coeffs, b_coeffs, c_coeffs):
        """
        a_coeffs: dict mapping (x_idx) -> list of (a_row, a_col, coeff)
        b_coeffs: dict mapping (y_idx) -> list of (b_row, b_col, coeff)
        c_coeffs: dict mapping (z_idx) -> list of (c_row, c_col, coeff)
        where x_idx, y_idx, z_idx are in 0..3 for the 2x2 sub-matrices
        """
        nonlocal col
        for sf in strassen_forms:
            col += 1
            c = col - 1
            
            for x_idx in range(4):
                for (ar, ac, coeff) in a_coeffs.get(x_idx, []):
                    if coeff != 0:
                        U_entries.append((ar * m + ac, c, coeff))
            
            for y_idx in range(4):
                for (br, bc, coeff) in b_coeffs.get(y_idx, []):
                    if coeff != 0:
                        V_entries.append((br * p + bc, c, coeff))
            
            for z_idx in range(4):
                z_coeff = sf['z'][z_idx]
                if z_coeff != 0:
                    for (cr, cc, w_coeff) in c_coeffs.get(z_idx, []):
                        total = z_coeff * w_coeff
                        if total != 0:
                            W_entries.append((cr * n + cc, c, total))
    
    # Define the 7 Strassen products for 4x4x4
    
    # M1 = (A11+A22)(B11+B22)
    # X00 = A[0,0]+A[2,2], X01 = A[0,1]+A[2,3]
    # X10 = A[1,0]+A[3,2], X11 = A[1,1]+A[3,3]
    # Y00 = B[0,0]+B[2,0], Y01 = B[0,1]+B[2,1]
    # Y10 = B[1,0]+B[3,0], Y11 = B[1,1]+B[3,1]
    # Z contributes to C11 (2x2 block at (0,0))
    a_m1 = {
        0: [(0,0,1), (2,2,1)], 1: [(0,1,1), (2,3,1)],
        2: [(1,0,1), (3,2,1)], 3: [(1,1,1), (3,3,1)]
    }
    b_m1 = {
        0: [(0,0,1), (2,0,1)], 1: [(0,1,1), (2,1,1)],
        2: [(1,0,1), (3,0,1)], 3: [(1,1,1), (3,1,1)]
    }
    c_m1 = {0: [(0,0,1)], 1: [(0,2,1)], 2: [(2,0,1)], 3: [(2,2,1)]}
    # C11 is at rows 0-1, cols 0-1 in C. W indexing: k*n+i
    # C[0,0] -> W[0*4+0], C[0,1] -> W[0*4+1], C[1,0] -> W[1*4+0], C[1,1] -> W[1*4+1]
    # So z_idx 0 (Z00) -> C[0,0] -> W[0], z_idx 1 (Z01) -> C[0,1] -> W[1]
    # z_idx 2 (Z10) -> C[1,0] -> W[4], z_idx 3 (Z11) -> C[1,1] -> W[5]
    c_m1 = {0: [(0,0,1)], 1: [(0,1,1)], 2: [(1,0,1)], 3: [(1,1,1)]}
    add_strassen_product(a_m1, b_m1, c_m1)
    
    # M2 = (A21+A22)B11
    # X = A21+A22: X00=A[2,0]+A[2,2], X01=A[2,1]+A[2,3], X10=A[3,0]+A[3,2], X11=A[3,1]+A[3,3]
    # Y = B11: Y00=B[0,0], Y01=B[0,1], Y10=B[1,0], Y11=B[1,1]
    # Z contributes to C21 (2x2 block at rows 2-3, cols 0-1)
    a_m2 = {
        0: [(2,0,1), (2,2,1)], 1: [(2,1,1), (2,3,1)],
        2: [(3,0,1), (3,2,1)], 3: [(3,1,1), (3,3,1)]
    }
    b_m2 = {
        0: [(0,0,1)], 1: [(0,1,1)],
        2: [(1,0,1)], 3: [(1,1,1)]
    }
    # C21: C[2,0]->W[0*4+2]=W[2], C[2,1]->W[0*4+3]=W[3], C[3,0]->W[1*4+2]=W[6], C[3,1]->W[1*4+3]=W[7]
    c_m2 = {0: [(0,2,1)], 1: [(0,3,1)], 2: [(1,2,1)], 3: [(1,3,1)]}
    add_strassen_product(a_m2, b_m2, c_m2)
    
    # M3 = A11(B21-B22)
    # X = A11: X00=A[0,0], X01=A[0,1], X10=A[1,0], X11=A[1,1]
    # Y = B21-B22: Y00=B[2,0]-B[0,0], Y01=B[2,1]-B[0,1], Y10=B[3,0]-B[1,0], Y11=B[3,1]-B[1,1]
    # Z contributes to C22 (2x2 block at rows 2-3, cols 2-3)
    a_m3 = {
        0: [(0,0,1)], 1: [(0,1,1)],
        2: [(1,0,1)], 3: [(1,1,1)]
    }
    b_m3 = {
        0: [(2,0,1), (0,0,-1)], 1: [(2,1,1), (0,1,-1)],
        2: [(3,0,1), (1,0,-1)], 3: [(3,1,1), (1,1,-1)]
    }
    # C22: C[2,2]->W[2*4+2]=W[10], C[2,3]->W[2*4+3]=W[11], C[3,2]->W[3*4+2]=W[14], C[3,3]->W[3*4+3]=W[15]
    c_m3 = {0: [(2,2,1)], 1: [(2,3,1)], 2: [(3,2,1)], 3: [(3,3,1)]}
    add_strassen_product(a_m3, b_m3, c_m3)
    
    # M4 = A22(B12-B11)
    # X = A22: X00=A[2,2], X01=A[2,3], X10=A[3,2], X11=A[3,3]
    # Y = B12-B11: Y00=B[0,2]-B[0,0], Y01=B[0,3]-B[0,1], Y10=B[1,2]-B[1,0], Y11=B[1,3]-B[1,1]
    # Z contributes to C11 with -1, and C12 with +1
    a_m4 = {
        0: [(2,2,1)], 1: [(2,3,1)],
        2: [(3,2,1)], 3: [(3,3,1)]
    }
    b_m4 = {
        0: [(0,2,1), (0,0,-1)], 1: [(0,3,1), (0,1,-1)],
        2: [(1,2,1), (1,0,-1)], 3: [(1,3,1), (1,1,-1)]
    }
    # C11: -1 * Z, C12: +1 * Z
    c_m4 = {0: [(0,0,-1), (0,2,1)], 1: [(0,1,-1), (0,3,1)], 2: [(1,0,-1), (1,2,1)], 3: [(1,1,-1), (1,3,1)]}
    add_strassen_product(a_m4, b_m4, c_m4)
    
    # M5 = (A11+A12)B22
    # X = A11+A12: X00=A[0,0]+A[0,2], X01=A[0,1]+A[0,3], X10=A[1,0]+A[1,2], X11=A[1,1]+A[1,3]
    # Y = B22: Y00=B[2,2], Y01=B[2,3], Y10=B[3,2], Y11=B[3,3]
    # Z contributes to C12 with +1, C22 with +1
    a_m5 = {
        0: [(0,0,1), (0,2,1)], 1: [(0,1,1), (0,3,1)],
        2: [(1,0,1), (1,2,1)], 3: [(1,1,1), (1,3,1)]
    }
    b_m5 = {
        0: [(2,2,1)], 1: [(2,3,1)],
        2: [(3,2,1)], 3: [(3,3,1)]
    }
    c_m5 = {0: [(0,2,1), (2,2,1)], 1: [(0,3,1), (2,3,1)], 2: [(1,2,1), (3,2,1)], 3: [(1,3,1), (3,3,1)]}
    add_strassen_product(a_m5, b_m5, c_m5)
    
    # M6 = (A21-A11)(B11+B12)
    # X = A21-A11: X00=A[2,0]-A[0,0], X01=A[2,1]-A[0,1], X10=A[3,0]-A[1,0], X11=A[3,1]-A[1,1]
    # Y = B11+B12: Y00=B[0,0]+B[0,2], Y01=B[0,1]+B[0,3], Y10=B[1,0]+B[1,2], Y11=B[1,1]+B[1,3]
    # Z contributes to C11 with -1, C21 with +1, C22 with -1
    a_m6 = {
        0: [(2,0,1), (0,0,-1)], 1: [(2,1,1), (0,1,-1)],
        2: [(3,0,1), (1,0,-1)], 3: [(3,1,1), (1,1,-1)]
    }
    b_m6 = {
        0: [(0,0,1), (0,2,1)], 1: [(0,1,1), (0,3,1)],
        2: [(1,0,1), (1,2,1)], 3: [(1,1,1), (1,3,1)]
    }
    c_m6 = {0: [(0,0,-1), (2,0,1), (2,2,-1)], 1: [(0,1,0), (2,1,1), (2,3,0)], 2: [(1,0,0), (3,0,1), (3,2,0)], 3: [(1,1,0), (3,1,1), (3,3,0)]}
    # Wait, let me recheck. Z10 and Z11 go to C21. Z00,Z01 go to C11.
    # c_m6 z_idx 0 (Z00): C11 with -1, C22 with -1? No...
    # C11 = M1 - M4 + M6 + M7, so M6 contributes +1 to C11
    # C21 = M1 + M2, so M6 does NOT contribute to C21
    # C22 = M3 - M2 + M5 + M7, so M6 does NOT contribute to C22
    # Wait, I made an error. Let me recheck Strassen:
    # C11 = M1 - M4 + M6 + M7
    # C12 = M5 + M4
    # C21 = M1 + M2
    # C22 = M3 - M2 + M5 + M7
    # So M6 only contributes to C11 with +1.
    c_m6 = {0: [(0,0,1)], 1: [(0,1,1)], 2: [(1,0,1)], 3: [(1,1,1)]}
    add_strassen_product(a_m6, b_m6, c_m6)
    
    # M7 = (A12-A22)(B21+B22)
    # X = A12-A22: X00=A[0,2]-A[2,2], X01=A[0,3]-A[2,3], X10=A[1,2]-A[3,2], X11=A[1,3]-A[3,3]
    # Y = B21+B22: Y00=B[2,0]+B[2,2], Y01=B[2,1]+B[2,3], Y10=B[3,0]+B[3,2], Y11=B[3,1]+B[3,3]
    # M7 contributes to C11 with +1 and C22 with +1
    a_m7 = {
        0: [(0,2,1), (2,2,-1)], 1: [(0,3,1), (2,3,-1)],
        2: [(1,2,1), (3,2,-1)], 3: [(1,3,1), (3,3,-1)]
    }
    b_m7 = {
        0: [(2,0,1), (2,2,1)], 1: [(2,1,1), (2,3,1)],
        2: [(3,0,1), (3,2,1)], 3: [(3,1,1), (3,3,1)]
    }
    c_m7 = {0: [(0,0,1), (2,2,1)], 1: [(0,1,1), (2,3,1)], 2: [(1,0,1), (3,2,1)], 3: [(1,1,1), (3,3,1)]}
    add_strassen_product(a_m7, b_m7, c_m7)
    
    # Part 2: 4x4x3 via schoolbook (rank 48)
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                col += 1
                c = col - 1
                U_entries.append((i * m + j, c, 1))
                V_entries.append((j * p + k, c, 1))
                W_entries.append((k * n + i, c, 1))
    
    # Assemble matrices
    rank = col
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for (row, c, val) in U_entries:
        U[row][c] += val
    for (row, c, val) in V_entries:
        V[row][c] += val
    for (row, c, val) in W_entries:
        W[row][c] += val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I need to verify my Strassen decomposition is correct. Let me trace through the C matrix reconstruction:

C11 = M1 - M4 + M6 + M7
C12 = M5 + M4  
C21 = M1 + M2
C22 = M3 - M2 + M5 + M7

Let me verify each C block gets the right contributions:

C11 (rows 0-1, cols 0-1):
- M1: +1 ✓ (c_m1 has +1 for C11)
- M4: -1 ✓ (c_m4 has -1 for C11)
- M6: +1 ✓ (c_m6 has +1 for C11)
- M7: +1 ✓ (c_m7 has +1 for C11)

C12 (rows 0-1, cols 2-3):
- M5: +1 ✓ (c_m5 has +1 for C12)
- M4: +1 ✓ (c_m4 has +1 for C12)

C21 (rows 2-3, cols 0-1):
- M1: +1 ✓ (c_m1 has +1 for C21)
- M2: +1 ✓ (c_m2 has +1 for C21)

C22 (rows 2-3, cols 2-3):
- M3: +1 ✓ (c_m3 has +1 for C22)
- M2: -1 ✓ (c_m2 should have -1 for C22, but I only put +1 for C21)

I made an error! M2 should contribute to C21 with +1 AND to C22 with -1. Let me fix c_m2.

Also checking:
- M5: +1 for C22 ✓
- M7: +1 for C22 ✓

Let me fix the c_m2 and also double-check all others.


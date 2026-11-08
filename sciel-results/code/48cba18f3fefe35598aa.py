"""
Rank-49 decomposition for 4x4 matrix multiplication via Kronecker square of Strassen.
"""
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-49 exact tensor decomposition for 4x4 matrix multiplication.
    
    The decomposition is derived from the Kronecker square of the 2x2 Strassen algorithm.
    Each of the 7 Strassen bilinear terms for 2x2 is replaced by a 4x4 block structure,
    yielding 7*7 = 49 bilinear terms total.
    
    U has shape (16, 49), V has shape (16, 49), W has shape (16, 49).
    Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] where n=m=p=4.
    """
    n = m = p = 4
    rank = 49
    
    # Initialize with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 bilinear terms.
    # For 2x2 matrices A (rows a0,a1) and B (cols b0,b1):
    # M0 = (a0+a1)*(b0+b1)  -> indices in A: row 0,1; in B: col 0,1
    # M1 = (a0+a1)*b1       -> A: row 0,1; B: col 1
    # M2 = a0*(b0-b1)       -> A: row 0; B: col 0,1
    # M3 = a1*b0            -> A: row 1; B: col 0
    # M4 = (a0-a1)*(b0+b1)  -> A: row 0,1; B: col 0,1
    # M5 = (a0+a1)*b1       -> wait, let me use standard Strassen
    #
    # Standard Strassen for 2x2:
    # C = A*B where A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C00 = M1 - M4 + M6 + M7
    # C01 = M2 + M3
    # C10 = M5 + M4
    # C11 = M1 - M2 + M3 + M5
    #
    # M1 = (a00 + a11) * (b00 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = (a00 + a01) * b11
    # M4 = a00 * (b01 - b11)
    # M5 = a11 * (b10 - b00)
    # M6 = (a00 + a01) * b10
    # M7 = (a10 + a11) * b01
    #
    # For the Kronecker square, we treat the 4x4 matrices as 2x2 blocks of 2x2.
    # A = [[A00, A01], [A10, A11]] where each Ai0 is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bj0 is 2x2
    #
    # C = A*B = [[C00, C01], [C10, C11]] where each Ci0 is 2x2
    #
    # C00 = M1 - M4 + M6 + M7
    # C01 = M2 + M3
    # C10 = M5 + M4
    # C11 = M1 - M2 + M3 + M5
    #
    # where Mj = (linear combo of A blocks) * (linear combo of B blocks)
    # and each Mj is a 2x2 matrix computed via Strassen on 2x2 blocks.
    #
    # So each Mj = Xj * Yj where Xj is a 2x2 matrix (linear combo of A blocks)
    # and Yj is a 2x2 matrix (linear combo of B blocks), and Xj*Yj is computed
    # via the 7 Strassen bilinear forms.
    #
    # Total: 7 (outer Strassen) * 7 (inner Strassen) = 49 bilinear terms.
    
    # Define the 7 Strassen linear forms for a 2x2 matrix.
    # For matrix X = [[x00, x01], [x10, x11]], the 7 linear forms are:
    # L0(X) = x00 + x11
    # L1(X) = x10 + x11
    # L2(X) = x00 + x01
    # L3(X) = x00
    # L4(X) = x11
    # L5(X) = x00 + x01
    # L6(X) = x10 + x11
    #
    # Wait, let me be more careful. In Strassen, we have:
    # M1 = L0(A) * L0(B) where L0(M) = m00 + m11
    # M2 = L1(A) * L3(B) where L1(M) = m10 + m11, L3(M) = m00
    # M3 = L2(A) * L4(B) where L2(M) = m00 + m01, L4(M) = m11
    # M4 = L3(A) * (L5(B)) where L5(B) = b01 - b11
    # M5 = L4(A) * (L6(B)) where L6(B) = b10 - b00
    # M6 = L2(A) * L3(B)  [same as L2 for A, L3 for B but different meaning]
    # M7 = L1(A) * L7(B) where L7(B) = b01
    #
    # Let me redefine more cleanly. The 7 Strassen bilinear forms are:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = (a00+a01)*b11
    # M4 = a00*(b01-b11)
    # M5 = a11*(b10-b00)
    # M6 = (a00+a01)*b10
    # M7 = (a10+a11)*b01
    #
    # For the Kronecker square, A is 4x4 viewed as 2x2 blocks of 2x2:
    # A = [[A00, A01], [A10, A11]]
    # where A00 = [[a00,a01],[a02,a03]], A01 = [[a04,a05],[a06,a07]],
    #       A10 = [[a08,a09],[a10,a11]], A11 = [[a12,a13],[a14,a15]]
    #
    # Similarly for B.
    #
    # The outer Strassen gives us 7 terms, each of which is a product of a 
    # linear combination of A-blocks and a linear combination of B-blocks.
    # Each such product is a 2x2 * 2x2 = 2x2 matrix multiplication, which 
    # we compute via the inner 7 Strassen terms.
    #
    # So we need to track, for each outer term t (0..6) and inner term s (0..6),
    # what linear combination of A's entries and B's entries goes into U and V.
    
    # Let's define the outer Strassen linear forms on the 2x2 block matrix.
    # For a 2x2 block matrix M = [[M00, M01], [M10, M11]] where each Mi0 is 2x2:
    # 
    # Outer form 0: M00 + M11 (block-wise sum)
    # Outer form 1: M10 + M11
    # Outer form 2: M00 + M01
    # Outer form 3: M00
    # Outer form 4: M11
    # Outer form 5: M00 + M01  (same as form 2 for A, but for B it's different)
    # Outer form 6: M10 + M11  (same as form 1)
    
    # Actually, let me just directly enumerate all 49 terms.
    # 
    # Outer Strassen terms (7):
    # O0: (A00+A11) * (B00+B11)
    # O1: (A10+A11) * B00
    # O2: (A00+A01) * B11
    # O3: A00 * (B01-B11)
    # O4: A11 * (B10-B00)
    # O5: (A00+A01) * B10
    # O6: (A10+A11) * B01
    #
    # For each outer term, we have X * Y where X is a 2x2 matrix (linear combo of A blocks)
    # and Y is a 2x2 matrix (linear combo of B blocks).
    #
    # The inner Strassen for X*Y (2x2*2x2) gives 7 bilinear terms:
    # I0: (x00+x11)*(y00+y11)
    # I1: (x10+x11)*y00
    # I2: (x00+x01)*y11
    # I3: x00*(y01-y11)
    # I4: x11*(y10-y00)
    # I5: (x00+x01)*y10
    # I6: (x10+x11)*y01
    #
    # And the result 2x2 matrix C = X*Y is:
    # C00 = I0 - I3 + I5 + I6
    # C01 = I1 + I2
    # C10 = I4 + I3
    # C11 = I0 - I1 + I2 + I4
    #
    # Now, for the full 4x4 result:
    # C00 (2x2) = O0_C00 - O3_C00 + O5_C00 + O6_C00
    # C01 (2x2) = O1_C00 + O2_C00
    # C10 (2x2) = O4_C00 + O3_C00
    # C11 (2x2) = O0_C00 - O1_C00 + O2_C00 + O4_C00
    #
    # where Oi_C00 means the (0,0) entry of the 2x2 result of outer term i.
    
    # Let me define the linear forms more explicitly.
    # 
    # For the outer level, the 2x2 block matrices have entries that are 2x2 submatrices.
    # A00 = [[a[0], a[1]], [a[2], a[3]]]  (using row-major: a[i*m+j] where i=row, j=col)
    # A01 = [[a[4], a[5]], [a[6], a[7]]]
    # A10 = [[a[8], a[9]], [a[10], a[11]]]
    # A11 = [[a[12], a[13]], [a[14], a[15]]]
    #
    # Similarly for B:
    # B00 = [[b[0], b[1]], [b[2], b[3]]]
    # B01 = [[b[4], b[5]], [b[6], b[7]]]
    # B10 = [[b[8], b[9]], [b[10], b[11]]]
    # B11 = [[b[12], b[13]], [b[14], b[15]]]
    
    # Define outer linear forms. Each returns a 2x2 matrix as a dict of (row,col) -> coeff.
    # We represent a 2x2 matrix as a list of 4 entries: [m00, m01, m10, m11]
    
    def outer_form_A(idx):
        """Return linear form on A's 2x2 blocks, as a 2x2 matrix of coefficients."""
        # Returns a 2x2 matrix where each entry is a linear combination of a[0..15]
        # Represented as dict: {(r,c): {a_idx: coeff}}
        result = {}
        for r in range(2):
            for c in range(2):
                result[(r,c)] = {}
        
        if idx == 0:  # A00 + A11
            for r in range(2):
                for c in range(2):
                    # A00[r,c] is at index r*2+c, A11[r,c] is at index 12+r*2+c
                    result[(r,c)][r*2+c] = 1
                    result[(r,c)][12+r*2+c] = 1
        elif idx == 1:  # A10 + A11
            for r in range(2):
                for c in range(2):
                    # A10[r,c] is at index 8+r*2+c, A11[r,c] is at index 12+r*2+c
                    result[(r,c)][8+r*2+c] = 1
                    result[(r,c)][12+r*2+c] = 1
        elif idx == 2:  # A00 + A01
            for r in range(2):
                for c in range(2):
                    # A00[r,c] is at index r*2+c, A01[r,c] is at index 4+r*2+c
                    result[(r,c)][r*2+c] = 1
                    result[(r,c)][4+r*2+c] = 1
        elif idx == 3:  # A00
            for r in range(2):
                for c in range(2):
                    result[(r,c)][r*2+c] = 1
        elif idx == 4:  # A11
            for r in range(2):
                for c in range(2):
                    result[(r,c)][12+r*2+c] = 1
        elif idx == 5:  # A00 + A01 (same as idx 2)
            for r in range(2):
                for c in range(2):
                    result[(r,c)][r*2+c] = 1
                    result[(r,c)][4+r*2+c] = 1
        elif idx == 6:  # A10 + A11 (same as idx 1)
            for r in range(2):
                for c in range(2):
                    result[(r,c)][8+r*2+c] = 1
                    result[(r,c)][12+r*2+c] = 1
        return result
    
    def outer_form_B(idx):
        """Return linear form on B's 2x2 blocks, as a 2x2 matrix of coefficients."""
        result = {}
        for r in range(2):
            for c in range(2):
                result[(r,c)] = {}
        
        if idx == 0:  # B00 + B11
            for r in range(2):
                for c in range(2):
                    result[(r,c)][r*2+c] = 1
                    result[(r,c)][12+r*2+c] = 1
        elif idx == 1:  # B00
            for r in range(2):
                for c in range(2):
                    result[(r,c)][r*2+c] = 1
        elif idx == 2:  # B11
            for r in range(2):
                for c in range(2):
                    result[(r,c)][12+r*2+c] = 1
        elif idx == 3:  # B01 - B11
            for r in range(2):
                for c in range(2):
                    result[(r,c)][4+r*2+c] = 1
                    result[(r,c)][12+r*2+c] = -1
        elif idx == 4:  # B10 - B00
            for r in range(2):
                for c in range(2):
                    result[(r,c)][8+r*2+c] = 1
                    result[(r,c)][r*2+c] = -1
        elif idx == 5:  # B10
            for r in range(2):
                for c in range(2):
                    result[(r,c)][8+r*2+c] = 1
        elif idx == 6:  # B01
            for r in range(2):
                for c in range(2):
                    result[(r,c)][4+r*2+c] = 1
        return result
    
    # Inner Strassen linear forms for 2x2 matrices.
    # For X = [[x00, x01], [x10, x11]]:
    # I0: (x00+x11) * (y00+y11)
    # I1: (x10+x11) * y00
    # I2: (x00+x01) * y11
    # I3: x00 * (y01-y11)
    # I4: x11 * (y10-y00)
    # I5: (x00+x01) * y10
    # I6: (x10+x11) * y01
    
    # For each inner term s, we need the linear form on X (2x2) and on Y (2x2).
    # X is itself a linear combination of A's entries, so we need to compose.
    
    def inner_form_X(s):
        """Return the linear form on X's 2x2 entries for inner Strassen term s.
        Returns a 2x2 matrix of coefficients: {(r,c): coeff}"""
        result = {}
        for r in range(2):
            for c in range(2):
                result[(r,c)] = 0
        
        if s == 0:  # x00 + x11
            result[(0,0)] = 1
            result[(1,1)] = 1
        elif s == 1:  # x10 + x11
            result[(1,0)] = 1
            result[(1,1)] = 1
        elif s == 2:  # x00 + x01
            result[(0,0)] = 1
            result[(0,1)] = 1
        elif s == 3:  # x00
            result[(0,0)] = 1
        elif s == 4:  # x11
            result[(1,1)] = 1
        elif s == 5:  # x00 + x01
            result[(0,0)] = 1
            result[(0,1)] = 1
        elif s == 6:  # x10 + x11
            result[(1,0)] = 1
            result[(1,1)] = 1
        return result
    
    def inner_form_Y(s):
        """Return the linear form on Y's 2x2 entries for inner Strassen term s.
        Returns a 2x2 matrix of coefficients: {(r,c): coeff}"""
        result = {}
        for r in range(2):
            for c in range(2):
                result[(r,c)] = 0
        
        if s == 0:  # y00 + y11
            result[(0,0)] = 1
            result[(1,1)] = 1
        elif s == 1:  # y00
            result[(0,0)] = 1
        elif s == 2:  # y11
            result[(1,1)] = 1
        elif s == 3:  # y01 - y11
            result[(0,1)] = 1
            result[(1,1)] = -1
        elif s == 4:  # y10 - y00
            result[(1,0)] = 1
            result[(0,0)] = -1
        elif s == 5:  # y10
            result[(1,0)] = 1
        elif s == 6:  # y01
            result[(0,1)] = 1
        return result
    
    # Now, for each outer term t (0..6) and inner term s (0..6), we have a bilinear term.
    # The U factor for this term is the composition of outer_form_A(t) and inner_form_X(s).
    # The V factor is the composition of outer_form_B(t) and inner_form_Y(s).
    
    # U[r, col] where col = t*7 + s
    # r is the row index in the 4x4 A matrix (0..15)
    
    for t in range(7):
        X_outer = outer_form_A(t)  # 2x2 matrix of dicts {a_idx: coeff}
        for s in range(7):
            X_inner = inner_form_X(s)  # 2x2 matrix of coeffs
            col = t * 7 + s
            # Compose: for each (r,c) in X_inner, add X_inner[(r,c)] * X_outer[(r,c)]
            for r in range(2):
                for c in range(2):
                    coeff = X_inner[(r,c)]
                    if coeff == 0:
                        continue
                    for a_idx, a_coeff in X_outer[(r,c)].items():
                        if a_coeff == 0:
                            continue
                        U[a_idx][col] += coeff * a_coeff
    
    # V[r, col] where r is the row index in the 4x4 B matrix (0..15)
    for t in range(7):
        Y_outer = outer_form_B(t)  # 2x2 matrix of dicts {b_idx: coeff}
        for s in range(7):
            Y_inner = inner_form_Y(s)  # 2x2 matrix of coeffs
            col = t * 7 + s
            for r in range(2):
                for c in range(2):
                    coeff = Y_inner[(r,c)]
                    if coeff == 0:
                        continue
                    for b_idx, b_coeff in Y_outer[(r,c)].items():
                        if b_coeff == 0:
                            continue
                        V[b_idx][col] += coeff * b_coeff
    
    # Now compute W.
    # The output C is a 4x4 matrix, indexed as C[k*n+i] = C[k*4+i] where k=row, i=col.
    # Wait, the task says W[k*n+i, r] where n=4, so W[k*4+i, r].
    # k is the row index of C (0..3), i is the column index of C (0..3).
    
    # C = A*B
    # C as 2x2 blocks: C = [[C00, C01], [C10, C11]]
    # C00 = O0_C00 - O3_C00 + O5_C00 + O6_C00
    # C01 = O1_C00 + O2_C00
    # C10 = O4_C00 + O3_C00
    # C11 = O0_C00 - O1_C00 + O2_C00 + O4_C00
    #
    # where Oi_C00 is the 2x2 result of outer term i, which is X_i * Y_i computed via inner Strassen.
    #
    # For each outer term t, the 2x2 result C_t = X_t * Y_t is:
    # C_t[0,0] = I0 - I3 + I5 + I6
    # C_t[0,1] = I1 + I2
    # C_t[1,0] = I4 + I3
    # C_t[1,1] = I0 - I1 + I2 + I4
    #
    # where Ij is the j-th inner Strassen bilinear term.
    
    # So for each output position (block_row, block_col) and inner position (ir, ic),
    # we need to sum over outer terms t with appropriate signs.
    
    # Output C[k*4+i] where k = block_row*2 + ir, i = block_col*2 + ic
    
    # Coefficients for each (block_row, block_col) position:
    # C00 (block 0,0): O0:+1, O3:-1, O5:+1, O6:+1
    # C01 (block 0,1): O1:+1, O2:+1
    # C10 (block 1,0): O4:+1, O3:+1
    # C11 (block 1,1): O0:+1, O1:-1, O2:+1, O4:+1
    
    block_coeffs = {
        (0,0): {0: 1, 3: -1, 5: 1, 6: 1},
        (0,1): {1: 1, 2: 1},
        (1,0): {4: 1, 3: 1},
        (1,1): {0: 1, 1: -1, 2: 1, 4: 1},
    }
    
    # Inner Strassen result coefficients:
    # C_t[0,0] = I0 - I3 + I5 + I6
    # C_t[0,1] = I1 + I2
    # C_t[1,0] = I4 + I3
    # C_t[1,1] = I0 - I1 + I2 + I4
    
    inner_result_coeffs = {
        (0,0): {0: 1, 3: -1, 5: 1, 6: 1},
        (0,1): {1: 1, 2: 1},
        (1,0): {4: 1, 3: 1},
        (1,1): {0: 1, 1: -1, 2: 1, 4: 1},
    }
    
    for block_row in range(2):
        for block_col in range(2):
            for ir in range(2):
                for ic in range(2):
                    # Output index: k = block_row*2 + ir, i = block_col*2 + ic
                    out_idx = (block_row * 2 + ir) * 4 + (block_col * 2 + ic)
                    for t in block_coeffs[(block_row, block_col)]:
                        t_coeff = block_coeffs[(block_row, block_col)][t]
                        for s in inner_result_coeffs[(ir, ic)]:
                            s_coeff = inner_result_coeffs[(ir, ic)][s]
                            col = t * 7 + s
                            W[out_idx][col] += t_coeff * s_coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

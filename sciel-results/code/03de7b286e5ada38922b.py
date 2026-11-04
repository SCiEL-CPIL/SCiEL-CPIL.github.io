"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 4
    # Strassen rank for 4x4 is 49
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices:
    # A: 4x4. Index i*4+j
    # B: 4x4. Index j*4+k
    # C: 4x4. Index k*4+i (W index)
    
    # Helper to add to U/V/W
    def add_to_U(i, j, val, col):
        U[i*m + j][col] += val
    def add_to_V(j, k, val, col):
        V[j*p + k][col] += val
    def add_to_W(k, i, val, col):
        W[k*n + i][col] += val
        
    # Strassen 2x2 block definitions
    # Blocks of A: A00, A01, A10, A11
    # A00: (0,0),(0,1),(1,0),(1,1)
    # A01: (0,2),(0,3),(1,2),(1,3)
    # A10: (2,0),(2,1),(3,0),(3,1)
    # A11: (2,2),(2,3),(3,2),(3,3)
    
    # We represent a 2x2 matrix of linear forms as a 4x4 matrix of coefficients?
    # No, just a list of (coeff, i, j) tuples.
    
    # Define outer Strassen combinations
    # Returns list of 7 tuples: (X_coeffs, Y_coeffs)
    # X_coeffs is a 2x2 matrix where each entry is a list of (val, i, j)
    # Y_coeffs is a 2x2 matrix where each entry is a list of (val, j, k)
    
    def get_A_block(bi, bj):
        # Returns list of (val, i, j) for the 2x2 block
        return [(1, bi*2+i, bj*2+j) for i in range(2) for j in range(2)]
        
    def get_B_block(bi, bj):
        # Returns list of (val, j, k) for the 2x2 block
        return [(1, bi*2+j, bj*2+k) for j in range(2) for k in range(2)]

    # We need to combine linear forms.
    def combine_forms(forms1, forms2, sign2=1):
        # forms1, forms2 are lists of (val, idx1, idx2)
        # Returns combined list
        res = {}
        for val, i, j in forms1:
            key = (i, j)
            res[key] = res.get(key, 0) + val
        for val, i, j in forms2:
            key = (i, j)
            res[key] = res.get(key, 0) + sign2 * val
        return [(v, i, j) for (i, j), v in res.items() if v != 0]

    # Define the 7 outer Strassen steps
    # Each step defines X (2x2 of A-forms) and Y (2x2 of B-forms)
    
    A00 = get_A_block(0,0); A01 = get_A_block(0,1)
    A10 = get_A_block(1,0); A11 = get_A_block(1,1)
    B00 = get_B_block(0,0); B01 = get_B_block(0,1)
    B10 = get_B_block(1,0); B11 = get_B_block(1,1)
    
    # M0 = (A00+A11)(B00+B11)
    X0 = [[combine_forms(A00, A11), [0]], [ [0], [0] ]] # Placeholder, need full 2x2
    # Actually, X0 is a 2x2 matrix of forms.
    # X0[0][0] = A00+A11, X0[0][1] = 0, etc.
    # Wait, Strassen multiplies two 2x2 matrices.
    # X0 = [[A00+A11, 0], [0, 0]]? No.
    # The Strassen algorithm computes 7 products of 2x2 matrices.
    # M0 = (A00+A11) * (B00+B11). Here (A00+A11) is a 2x2 matrix.
    # So X0 is the 2x2 matrix A00+A11.
    # X0[0][0] = A00[0][0] + A11[0][0] ...
    
    def add_2x2_forms(M1, M2):
        # M1, M2 are 2x2 lists of forms
        res = [[None for _ in range(2)] for _ in range(2)]
        for i in range(2):
            for j in range(2):
                res[i][j] = combine_forms(M1[i][j], M2[i][j])
        return res

    def zero_2x2():
        return [[[] for _ in range(2)] for _ in range(2)]

    # Construct X_k and Y_k for k=0..6
    # X0 = A00+A11, Y0 = B00+B11
    X0 = add_2x2_forms([[A00[i][j] for j in range(2)] for i in range(2)], 
                       [[A11[i][j] for j in range(2)] for i in range(2)])
    # Wait, A00 is a flat list. I need 2x2 structure.
    
    # Let's restructure A blocks as 2x2 matrices of forms.
    def block_to_2x2(block):
        # block is list of 4 items: (val, i, j) for i,j in 0..1
        # Map to 2x2
        res = [[[] for _ in range(2)] for _ in range(2)]
        for val, i, j in block:
            res[i][j].append((val, i, j))
        return res

    A00m = block_to_2x2(A00); A01m = block_to_2x2(A01)
    A10m = block_to_2x2(A10); A11m = block_to_2x2(A11)
    B00m = block_to_2x2(B00); B01m = block_to_2x2(B01)
    B10m = block_to_2x2(B10); B11m = block_to_2x2(B11)

    # Define outer Strassen X, Y
    X0 = add_2x2_forms(A00m, A11m); Y0 = add_2x2_forms(B00m, B11m)
    X1 = add_2x2_forms(A10m, A11m); Y1 = B00m
    X2 = A00m; Y2 = add_2x2_forms(B01m, B11m, -1) # B01 - B11
    X3 = A11m; Y3 = add_2x2_forms(B10m, B00m, -1) # B10 - B00
    X4 = add_2x2_forms(A00m, A01m); Y4 = B11m
    X5 = add_2x2_forms(A10m, A00m, -1); Y5 = add_2x2_forms(B00m, B01m) # A10-A00, B00+B01
    X6 = add_2x2_forms(A01m, A11m, -1); Y6 = add_2x2_forms(B10m, B11m) # A01-A11, B10+B11

    OuterStrassen = [(X0, Y0), (X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4), (X5, Y5), (X6, Y6)]

    # Inner Strassen for each 2x2 matrix product
    # For each k, we have X_k, Y_k.
    # We compute 7 scalar multiplications.
    
    col = 0
    for k in range(7):
        X, Y = OuterStrassen[k]
        # Inner Strassen combinations
        # s0 = (x00+x11)(y00+y11)
        # s1 = (x10+x11)y00
        # s2 = x00(y01-y11)
        # s3 = x11(y10-y00)
        # s4 = (x00+x01)y11
        # s5 = (x10-x00)(y00+y01)
        # s6 = (x01-x11)(y10+y11)
        
        # We need to extract the linear forms for the scalar multiplications.
        # x00 is X[0][0], etc.
        
        # Define inner Strassen pairs (L_A, L_B)
        # L_A is a list of (val, i, j)
        # L_B is a list of (val, j, k)
        
        def extract_form(M, i, j):
            return M[i][j]

        # s0
        LA = combine_forms(extract_form(X,0,0), extract_form(X,1,1))
        LB = combine_forms(extract_form(Y,0,0), extract_form(Y,1,1))
        # s1
        LA1 = combine_forms(extract_form(X,1,0), extract_form(X,1,1))
        LB1 = extract_form(Y,0,0)
        # s2
        LA2 = extract_form(X,0,0)
        LB2 = combine_forms(extract_form(Y,0,1), extract_form(Y,1,1), -1)
        # s3
        LA3 = extract_form(X,1,1)
        LB3 = combine_forms(extract_form(Y,1,0), extract_form(Y,0,0), -1)
        # s4
        LA4 = combine_forms(extract_form(X,0,0), extract_form(X,0,1))
        LB4 = extract_form(Y,1,1)
        # s5
        LA5 = combine_forms(extract_form(X,1,0), extract_form(X,0,0), -1)
        LB5 = combine_forms(extract_form(Y,0,0), extract_form(Y,0,1))
        # s6
        LA6 = combine_forms(extract_form(X,0,1), extract_form(X,1,1), -1)
        LB6 = combine_forms(extract_form(Y,1,0), extract_form(Y,1,1))

        InnerStrassen = [
            (LA, LB),
            (LA1, LB1),
            (LA2, LB2),
            (LA3, LB3),
            (LA4, LB4),
            (LA5, LB5),
            (LA6, LB6)
        ]

        # For each inner multiplication, update U and V
        # And determine contribution to W
        
        # Contribution of m_{k,l} to M_k (2x2 block)
        # M_k00 = s0 + s3 - s4 + s6
        # M_k01 = s2 + s4
        # M_k10 = s1 + s3
        # M_k11 = s0 - s1 + s2 + s5
        
        # Coefficients of s_l in M_k elements
        # M_k00: s0(1), s3(1), s4(-1), s6(1)
        # M_k01: s2(1), s4(1)
        # M_k10: s1(1), s3(1)
        # M_k11: s0(1), s1(-1), s2(1), s5(1)
        
        M_coeffs = [
            {0: 1, 3: 1, 4: -1, 6: 1}, # M00
            {2: 1, 4: 1},               # M01
            {1: 1, 3: 1},               # M10
            {0: 1, 1: -1, 2: 1, 5: 1}   # M11
        ]

        # Contribution of M_k to C (4x4)
        # C00 = M00 + M3 - M4 + M6
        # C01 = M2 + M4
        # C10 = M1 + M3
        # C11 = M0 - M1 + M2 + M5
        
        # We need to sum contributions from all k.
        # For a specific m_{k,l}, it contributes to M_k elements.
        # Each M_k element contributes to specific C elements.
        
        # C00 (Block 0,0) gets M0[0,0] + M3[0,0] - M4[0,0] + M6[0,0]
        # C01 (Block 0,1) gets M2[0,0] + M4[0,0] (and [0,1] parts)
        # Actually, C is 4x4.
        # C[0:2, 0:2] = M0[0,0] + M3[0,0] - M4[0,0] + M6[0,0]
        # C[0:2, 2:4] = M2[0,0] + M4[0,0] (Wait, M2 is a 2x2 block. So C[0:2, 2:4] = M2 + M4? No.
        # Strassen result:
        # C11 (top-left) = M1 + M4 - M5 + M7 (using 1-based)
        # Let's stick to 0-based:
        # C00 = M0 + M3 - M4 + M6
        # C01 = M2 + M4
        # C10 = M1 + M3
        # C11 = M0 - M1 + M2 + M5
        
        # So for a specific M_k element (say M_k[0,0]), it contributes to:
        # If k=0: C00[0,0] += 1, C11[0,0] += 1
        # If k=1: C10[0,0] += 1, C11[0,0] += -1
        # If k=2: C01[0,0] += 1, C11[0,0] += 1
        # If k=3: C00[0,0] += 1, C10[0,0] += 1
        # If k=4: C00[0,0] += -1, C01[0,0] += 1
        # If k=5: C11[0,0] += 1
        # If k=6: C00[0,0] += 1
        
        # We need to map M_k[i,j] to C global indices.
        # M_k is a 2x2 block.
        # M_k[0,0] corresponds to C[bi*2+0, bj*2+0] for the block position (bi,bj) of M_k?
        # No, M_k is one of the 7 Strassen products.
        # The result C is composed of 4 blocks: C00, C01, C10, C11.
        # Each C block is a 2x2 matrix.
        # C00 = M0 + M3 - M4 + M6.
        # So C00[i,j] = M0[i,j] + M3[i,j] - M4[i,j] + M6[i,j].
        # Global index for C00[i,j] is (0*2+i, 0*2+j) = (i, j).
        
        # So for a given k and M_k element (mi, mj), we add to C:
        # If k=0: C[mi,mj] += 1, C[2+mi, 2+mj] += 1 (from C11)
        # If k=1: C[2+mi, mj] += 1 (C10), C[2+mi, 2+mj] += -1 (C11)
        # If k=2: C[mi, 2+mj] += 1 (C01), C[2+mi, 2+mj] += 1 (C11)
        # If k=3: C[mi, mj] += 1 (C00), C[2+mi, mj] += 1 (C10)
        # If k=4: C[mi, mj] += -1 (C00), C[mi, 2+mj] += 1 (C01)
        # If k=5: C[2+mi, 2+mj] += 1 (C11)
        # If k=6: C[mi, mj] += 1 (C00)
        
        # We iterate over l (inner strassen index)
        for l in range(7):
            LA, LB = InnerStrassen[l]
            # Update U and V
            for val, i, j in LA:
                add_to_U(i, j, val, col)
            for val, j, k_idx in LB:
                add_to_V(j, k_idx, val, col)
                
            # Update W
            # Get coefficients of s_l in M_k elements
            for mi in range(2):
                for mj in range(2):
                    coeff_in_M = M_coeffs[mi*2+mj].get(l, 0)
                    if coeff_in_M == 0: continue
                    
                    # Determine C contributions
                    # M_k[mi, mj] contributes to C blocks based on k
                    if k == 0:
                        # C00[mi,mj] += 1, C11[mi,mj] += 1
                        add_to_W(0*2+mj, 0*2+mi, coeff_in_M * 1, col)
                        add_to_W(2*2+mj, 2*2+mi, coeff_in_M * 1, col)
                    elif k == 1:
                        # C10[mi,mj] += 1, C11[mi,mj] += -1
                        add_to_W(0*2+mj, 2*2+mi, coeff_in_M * 1, col)
                        add_to_W(2*2+mj, 2*2+mi, coeff_in_M * -1, col)
                    elif k == 2:
                        # C01[mi,mj] += 1, C11[mi,mj] += 1
                        add_to_W(2*2+mj, 0*2+mi, coeff_in_M * 1, col)
                        add_to_W(2*2+mj, 2*2+mi, coeff_in_M * 1, col)
                    elif k == 3:
                        # C00[mi,mj] += 1, C10[mi,mj] += 1
                        add_to_W(0*2+mj, 0*2+mi, coeff_in_M * 1, col)
                        add_to_W(0*2+mj, 2*2+mi, coeff_in_M * 1, col)
                    elif k == 4:
                        # C00[mi,mj] += -1, C01[mi,mj] += 1
                        add_to_W(0*2+mj, 0*2+mi, coeff_in_M * -1, col)
                        add_to_W(2*2+mj, 0*2+mi, coeff_in_M * 1, col)
                    elif k == 5:
                        # C11[mi,mj] += 1
                        add_to_W(2*2+mj, 2*2+mi, coeff_in_M * 1, col)
                    elif k == 6:
                        # C00[mi,mj] += 1
                        add_to_W(0*2+mj, 0*2+mi, coeff_in_M * 1, col)
                        
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

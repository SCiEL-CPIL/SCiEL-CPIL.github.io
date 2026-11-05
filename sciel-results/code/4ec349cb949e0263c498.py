import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    DEADLINE = 130  # seconds safety margin

    # Strategy: Laderman rank-49 for 4x4x4 + schoolbook rank-16 for 4x4x1 = rank 65
    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65

    def add_term(col, u_list, v_list, w_list):
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c

    # Laderman's rank-49 decomposition for 4x4x4 matrix multiplication
    # Using nested Strassen: 8 outer Strassen terms, each with 7 inner Strassen terms = 56
    # But we want 49, so use the actual Laderman construction.
    # 
    # Alternative: Use the known 49-term decomposition.
    # Let's use a clean construction based on Strassen applied to 2x2 blocks of 2x2.
    # This gives 8*7 = 56 terms for 4x4x4, plus 16 for 4x4x1 = 72 total.
    # That's worse than 65.
    #
    # Let me use the explicit Laderman 49-term construction.
    # The Laderman algorithm for 4x4x4 has 49 multiplications.
    
    # Actually, let me use a verified approach: 
    # Split 4x4x4 into four 2x2x2 problems with cross terms.
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    #
    # Each 2x2x2 can be done with Strassen (rank 7).
    # C11 needs: A11*B11 (7) + A12*B21 (7) = 14, but we can share:
    # Use Strassen on the 2x2 block matrix multiplication directly.
    #
    # For the 2x2 block Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 - M4 - M6 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 - M5 + M7
    #
    # Each Mi is a 2x2x2 product, done with Strassen (7 terms each).
    # Total: 7 * 7 = 49 terms for 4x4x4.
    
    # Strassen for 2x2x2:
    # S1 = (a11+a22)(b11+b22)
    # S2 = (a21+a22)b11
    # S3 = a11(b12-b22)
    # S4 = a22(b21-b11)
    # S5 = (a11+a12)b22
    # S6 = (a21-a11)(b11+b12)
    # S7 = (a12-a22)(b21+b22)
    #
    # c11 = S1 - S4 - S6 + S7
    # c12 = S3 + S5
    # c21 = S2 + S4
    # c22 = S1 - S3 - S5 + S7
    
    # Define Strassen 2x2x2 terms
    # Each term: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: list of (i, j, coeff) for A[i,j]
    # v_coeffs: list of (j, k, coeff) for B[j,k]
    # w_coeffs: list of (k, i, coeff) for C[i,k] (note: W index is k*n+i)
    
    strassen_2x2 = [
        # S1 = (a11+a22)(b11+b22)
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # S2 = (a21+a22)b11
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1)]),
        # S3 = a11(b12-b22)
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1)]),
        # S4 = a22(b21-b11)
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(1,0,1),(0,0,-1)]),
        # S5 = (a11+a12)b22
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,1,1)]),
        # S6 = (a21-a11)(b11+b12)
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,0,1)]),
        # S7 = (a12-a22)(b21+b22)
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    # Strassen reconstruction for 2x2x2:
    # c11 = S1 - S4 - S6 + S7
    # c12 = S3 + S5
    # c21 = S2 + S4
    # c22 = S1 - S3 - S5 + S7
    strassen_recon = [
        # (output_pos, [(strassen_idx, sign), ...])
        ((0,0), [(0,1),(3,-1),(5,-1),(6,1)]),   # c11 = S1 - S4 - S6 + S7
        ((0,1), [(2,1),(4,1)]),                   # c12 = S3 + S5
        ((1,0), [(1,1),(3,1)]),                   # c21 = S2 + S4
        ((1,1), [(0,1),(2,-1),(4,-1),(6,1)]),     # c22 = S1 - S3 - S5 + S7
    ]
    
    # Outer Strassen for 4x4x4 (treating 4x4 as 2x2 blocks of 2x2)
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # where A00 is rows 0-1, cols 0-1; A01 is rows 0-1, cols 2-3; etc.
    
    # Outer Strassen terms (each is a 2x2 block product):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each Mi is computed using inner Strassen (7 terms)
    # Total: 7 outer * 7 inner = 49 terms
    
    # Block index mapping:
    # Block (br, bc) where br in {0,1}, bc in {0,1}
    # Block (br, bc) corresponds to rows [br*2, br*2+1], cols [bc*2, bc*2+1]
    
    def block_to_global(br, bc, local_i, local_j):
        """Convert block indices to global indices."""
        gi = br * 2 + local_i
        gj = bc * 2 + local_j
        return gi, gj
    
    col = 0
    # Outer Strassen definitions: (A_block_coeffs, B_block_coeffs, C_block_contribs)
    # A_block_coeffs: list of (br, bc, coeff)
    # B_block_coeffs: list of (br, bc, coeff)
    # C_block_contribs: list of (br, bc, sign) for how this M contributes to C blocks
    
    outer_strassen = [
        # M1 = (A00+A11)(B00+B11)
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M2 = (A10+A11)B00
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1)]),
        # M3 = A00(B01-B11)
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1)]),
        # M4 = A11(B10-B00)
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(1,0,1),(0,0,-1)]),
        # M5 = (A00+A01)B11
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,1,1)]),
        # M6 = (A10-A00)(B00+B01)
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,0,1)]),
        # M7 = (A01-A11)(B10+B11)
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    for outer_idx, (a_blocks, b_blocks, c_blocks) in enumerate(outer_strassen):
        for inner_idx, (inner_u, inner_v, inner_w) in enumerate(strassen_2x2):
            u_list = []
            v_list = []
            w_list = []
            
            # Expand A blocks
            for (abr, abc, acoeff) in a_blocks:
                for (li, lj, lc) in inner_u:
                    gi, gj = block_to_global(abr, abc, li, lj)
                    u_list.append((gi, gj, acoeff * lc))
            
            # Expand B blocks
            for (bbr, bbc, bcoeff) in b_blocks:
                for (li, lj, lc) in inner_v:
                    gi, gj = block_to_global(bbr, bbc, li, lj)
                    v_list.append((gi, gj, bcoeff * lc))
            
            # Expand C contributions
            for (cbr, cbc, csign) in c_blocks:
                for (li, lj, lc) in inner_w:
                    gi, gj = block_to_global(cbr, cbc, li, lj)
                    # W index is k*n + i where k is column, i is row
                    w_list.append((gj, gi, csign * lc))
            
            if u_list and v_list and w_list:
                add_term(col, u_list, v_list, w_list)
                col += 1
    
    # Part 2: 4x4x1 schoolbook (rank 16)
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1
    
    # Verify the decomposition is correct
    if time.time() - start_time < DEADLINE - 10:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        rng = np.random.RandomState(42)
        A_test = rng.rand(4, 4)
        B_test = rng.rand(4, 5)
        
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((4, 5))
        for alpha in range(rank):
            a_part = sum(A_test[i, k] * U_arr[i * 4 + k, alpha] for i in range(4) for k in range(4))
            b_part = sum(B_test[k, j] * V_arr[k * 5 + j, alpha] for k in range(4) for j in range(5))
            for i in range(4):
                for j in range(5):
                    C_computed[i, j] += a_part * b_part * W_arr[j * 4 + i, alpha]
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # Fallback: pure schoolbook rank 80
            rank = 80
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(4):
                for k in range(4):
                    for j in range(5):
                        U[i * 4 + k][col] = 1
                        V[k * 5 + j][col] = 1
                        W[j * 4 + i][col] = 1
                        col += 1
    
    return {"U": U, "V": V, "W": W}

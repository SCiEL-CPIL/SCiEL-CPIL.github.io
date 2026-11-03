import numpy as np

def construct():
    n, m, p = 3, 4, 4
    rank = 44
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # 1. Schoolbook for i=0 (1x4x4)
    # C[0, k] = sum_j A[0, j] B[j, k]
    r = 0
    for j in range(m):
        for k in range(p):
            U[0*m + j, r] = 1
            V[j*p + k, r] = 1
            W[k*n + 0, r] = 1
            r += 1
            
    # 2. Strassen for i=1,2
    # We need to compute C[1:3, 0:4]
    # Split into 4 blocks:
    # C[1:3, 0:2] = A[1:3, 0:2] B[0:2, 0:2] + A[1:3, 2:4] B[2:4, 0:2]
    # C[1:3, 2:4] = A[1:3, 0:2] B[0:2, 2:4] + A[1:3, 2:4] B[2:4, 2:4]
    
    # Define Strassen helper
    def add_strassen_block(U, V, W, i_offset, j_offset_A, j_offset_B, k_offset, r_start):
        # M = A[i_offset:i_offset+2, j_offset_A:j_offset_A+2]
        # N = B[j_offset_B:j_offset_B+2, k_offset:k_offset+2]
        # R = C[i_offset:i_offset+2, k_offset:k_offset+2]
        
        # Strassen terms
        # P1 = (M00+M11)(N00+N11)
        # P2 = (M10+M11)N00
        # P3 = M00(N01-N11)
        # P4 = M11(N10-N00)
        # P5 = (M00+M01)N11
        # P6 = (M10-M00)(N00+N01)
        # P7 = (M01-M11)(N10+N11)
        
        # Coefficients for M (rows 0,1 cols 0,1)
        # Coefficients for N (rows 0,1 cols 0,1)
        # Coefficients for R (rows 0,1 cols 0,1)
        
        terms = [
            # (M_coeffs, N_coeffs, R_coeffs)
            # M_coeffs: [M00, M01, M10, M11]
            # N_coeffs: [N00, N01, N10, N11]
            # R_coeffs: [R00, R01, R10, R11]
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # P1 -> R00, R11
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # P2 -> R10
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]), # P3 -> R01
            ([0, 0, 0, 1], [0, 0, 1, -1], [0, 0, 1, 1]), # P4 -> R10, R11
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 0, 0]), # P5 -> R00
            ([1, 0, -1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # P6 -> R11
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])  # P7 -> R00
        ]
        
        # Wait, the R_coeffs above are just the contribution of that P to the R elements.
        # P1 contributes to R00 (+1) and R11 (+1).
        # P2 contributes to R10 (+1).
        # P3 contributes to R01 (+1).
        # P4 contributes to R10 (+1) and R11 (+1).
        # P5 contributes to R00 (-1).
        # P6 contributes to R11 (+1).
        # P7 contributes to R00 (+1).
        
        # Let's re-verify Strassen formulas:
        # R00 = P1 + P4 - P5 + P7
        # R01 = P3 + P5
        # R10 = P2 + P4
        # R11 = P1 - P2 + P3 + P6
        
        # So for each P, we need to add its contribution to the corresponding R elements.
        # P1: R00 += 1, R11 += 1
        # P2: R10 += 1, R11 += -1
        # P3: R01 += 1, R11 += 1
        # P4: R00 += 1, R10 += 1
        # P5: R00 += -1, R01 += 1
        # P6: R11 += 1
        # P7: R00 += 1
        
        # My list above:
        # P1: R=[1,0,0,1] -> R00=1, R11=1. Correct.
        # P2: R=[0,0,1,0] -> R10=1. Missing R11=-1.
        # P3: R=[0,1,0,0] -> R01=1. Missing R11=1.
        # P4: R=[0,0,1,1] -> R10=1, R11=1. Missing R00=1.
        # P5: R=[-1,0,0,0] -> R00=-1. Missing R01=1.
        # P6: R=[0,0,0,1] -> R11=1. Correct.
        # P7: R=[1,0,0,0] -> R00=1. Correct.
        
        # I need to fix the R coefficients in the `terms` list.
        
        corrected_terms = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # P1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # P2
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),  # P3
            ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 1]),  # P4
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # P5
            ([1, 0, -1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # P6
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])   # P7
        ]
        
        r_idx = r_start
        for M_c, N_c, R_c in corrected_terms:
            # M_c = [M00, M01, M10, M11]
            # Indices in M: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
            # Global A index: (i_offset + mi, j_offset_A + mj)
            # U index: (i_offset + mi)*m + (j_offset_A + mj)
            
            # N_c = [N00, N01, N10, N11]
            # Global B index: (j_offset_B + ni, k_offset + nk)
            # V index: (j_offset_B + ni)*p + (k_offset + nk)
            
            # R_c = [R00, R01, R10, R11]
            # Global C index: (i_offset + ri, k_offset + rk)
            # W index: (k_offset + rk)*n + (i_offset + ri)
            
            for mi in range(2):
                for mj in range(2):
                    coeff = M_c[mi*2 + mj]
                    if coeff != 0:
                        U[(i_offset + mi)*m + (j_offset_A + mj), r_idx] += coeff
            
            for ni in range(2):
                for nk in range(2):
                    coeff = N_c[ni*2 + nk]
                    if coeff != 0:
                        V[(j_offset_B + ni)*p + (k_offset + nk), r_idx] += coeff
            
            for ri in range(2):
                for rk in range(2):
                    coeff = R_c[ri*2 + rk]
                    if coeff != 0:
                        W[(k_offset + rk)*n + (i_offset + ri), r_idx] += coeff
            
            r_idx += 1
        
        return r_idx

    # Block 1: A[1:3, 0:2] * B[0:2, 0:2] -> C[1:3, 0:2]
    r = add_strassen_block(U, V, W, 1, 0, 0, 0, r)
    # Block 2: A[1:3, 2:4] * B[2:4, 0:2] -> C[1:3, 0:2]
    r = add_strassen_block(U, V, W, 1, 2, 2, 0, r)
    # Block 3: A[1:3, 0:2] * B[0:2, 2:4] -> C[1:3, 2:4]
    r = add_strassen_block(U, V, W, 1, 0, 0, 2, r)
    # Block 4: A[1:3, 2:4] * B[2:4, 2:4] -> C[1:3, 2:4]
    r = add_strassen_block(U, V, W, 1, 2, 2, 2, r)
    
    assert r == rank, f"Rank mismatch: {r} vs {rank}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

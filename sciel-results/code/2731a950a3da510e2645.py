import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 decomposition using recursive Strassen
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # C_00 = S1 + S4 - S5 + S7
    # C_01 = S3 + S5
    # C_10 = S2 + S4
    # C_11 = S1 - S2 + S3 + S6
    # S1 = (A00+A11)(B00+B11)
    # S2 = (A10+A11)B00
    # S3 = A00(B01-B11)
    # S4 = A11(B10-B00)
    # S5 = (A00+A01)B11
    # S6 = (A10-A00)(B00+B01)
    # S7 = (A01-A11)(B10+B11)
    
    # Coefficients for C entries from S terms
    # c[i][k][t] is coefficient of S_t in C_ik
    c = [
        [
            [1, 0, 0, 1, -1, 0, 1], # C00
            [0, 0, 1, 0, 1, 0, 0]   # C01
        ],
        [
            [0, 1, 0, 1, 0, 0, 0], # C10
            [1, -1, 1, 0, 0, 1, 0] # C11
        ]
    ]
    
    # Linear forms for S terms
    # A_forms[t][i][j] is coefficient of A_ij in the A-factor of S_t
    # B_forms[t][i][j] is coefficient of B_ij in the B-factor of S_t
    A_forms = [
        [[1, 0], [0, 1]], # S1: A00+A11
        [[0, 0], [1, 1]], # S2: A10+A11
        [[1, 0], [0, 0]], # S3: A00
        [[0, 0], [0, 1]], # S4: A11
        [[1, 1], [0, 0]], # S5: A00+A01
        [[-1, 0], [1, 0]],# S6: A10-A00
        [[0, -1], [0, 1]] # S7: A01-A11
    ]
    B_forms = [
        [[1, 0], [0, 1]], # S1: B00+B11
        [[1, 0], [0, 0]], # S2: B00
        [[0, 1], [0, -1]],# S3: B01-B11
        [[0, -1], [1, 0]],# S4: B10-B00
        [[0, 0], [0, 1]], # S5: B11
        [[1, 1], [0, 0]], # S6: B00+B01
        [[0, 1], [1, 1]]  # S7: B10+B11
    ]
    
    # Outer Strassen coefficients (same as c)
    # c_outer[I][K][s]
    c_outer = c
    
    # Inner Strassen coefficients (same as c)
    # c_inner[i][k][t]
    c_inner = c
    
    # We need to map the inner Strassen A/B forms to the full A/B matrices
    # For a given outer term s, X_s and Y_s are 2x2 matrices of linear forms.
    # X_s[i][j] is a linear form on A.
    # X_s[i][j] = sum_{I,J} A_forms_outer[s][i][j] * A_block[I][J]
    # Wait, the outer Strassen A_forms are for the blocks.
    # Let's define outer A_forms and B_forms.
    
    A_outer_forms = A_forms
    B_outer_forms = B_forms
    
    for s in range(7):
        # Construct X_s and Y_s as 2x2 matrices of linear forms (vectors of length 16)
        # X_s[i][j] is a vector of length 16
        X_s = [[0]*16 for _ in range(2)]
        Y_s = [[0]*16 for _ in range(2)]
        
        for i in range(2):
            for j in range(2):
                # X_s[i][j] is a linear combination of A blocks
                # The coefficient is A_outer_forms[s][i][j]
                coeff = A_outer_forms[s][i][j]
                if coeff != 0:
                    # Add coeff * A_block[i][j] to X_s[i][j]
                    # A_block[i][j] corresponds to A[2i+u][2j+v]
                    for u in range(2):
                        for v in range(2):
                            idx_a = (2*i + u) * 4 + (2*j + v)
                            X_s[i][j][idx_a] += coeff
                            
                coeff_b = B_outer_forms[s][i][j]
                if coeff_b != 0:
                    # Add coeff_b * B_block[i][j] to Y_s[i][j]
                    for u in range(2):
                        for v in range(2):
                            idx_b = (2*i + u) * 4 + (2*j + v)
                            Y_s[i][j][idx_b] += coeff_b
        
        # Now apply inner Strassen to X_s and Y_s
        # This generates 7 scalar terms P_st
        for t in range(7):
            r = s * 7 + t
            
            # Get the linear forms for P_st
            # L_A is a linear form on X_s entries
            # L_B is a linear form on Y_s entries
            L_A = [0]*16
            L_B = [0]*16
            
            # A-factor of S_t is a 2x2 matrix of coefficients
            # L_A = sum_{i,j} A_inner_forms[t][i][j] * X_s[i][j]
            for i in range(2):
                for j in range(2):
                    coeff = A_inner_forms[t][i][j]
                    if coeff != 0:
                        for k in range(16):
                            L_A[k] += coeff * X_s[i][j][k]
                            
            # B-factor of S_t
            for i in range(2):
                for j in range(2):
                    coeff = B_inner_forms[t][i][j]
                    if coeff != 0:
                        for k in range(16):
                            L_B[k] += coeff * Y_s[i][j][k]
                            
            # Set U and V
            for u in range(16):
                U[u][r] = L_A[u]
            for v in range(16):
                V[v][r] = L_B[v]
                
            # Set W
            # P_st contributes to M_s
            # M_s is a 2x2 matrix
            # M_s[i][k] = sum_t c_inner[i][k][t] * P_st
            # M_s contributes to C blocks
            # C[I][K] = sum_s c_outer[I][K][s] * M_s
            # So P_st contributes to C[2I+i][2K+k] with coeff c_outer[I][K][s] * c_inner[i][k][t]
            
            for I in range(2):
                for K in range(2):
                    coeff_outer = c_outer[I][K][s]
                    if coeff_outer == 0:
                        continue
                    for i in range(2):
                        for k in range(2):
                            coeff_inner = c_inner[i][k][t]
                            if coeff_inner == 0:
                                continue
                            total_coeff = coeff_outer * coeff_inner
                            if total_coeff != 0:
                                i_full = 2*I + i
                                k_full = 2*K + k
                                w_idx = k_full * 4 + i_full
                                W[w_idx][r] += total_coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

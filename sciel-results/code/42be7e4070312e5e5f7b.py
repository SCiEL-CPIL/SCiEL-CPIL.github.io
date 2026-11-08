import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A10)B11
    # M6 = (A11-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # u_coeffs[s][ij] for A_ij
    # v_coeffs[s][jk] for B_jk
    # w_coeffs[s][ik] for C_ik
    
    # Indices 00, 01, 10, 11 -> 0, 1, 2, 3
    u_strassen = [
        [1, 0, 0, 1], # M1: A00+A11
        [0, 0, 1, 1], # M2: A10+A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 0, 1, 0], # M5: A00+A10
        [-1, 0, 0, 1],# M6: A11-A00
        [0, 1, 0, -1] # M7: A01-A11
    ]
    
    v_strassen = [
        [1, 1, 0, 0], # M1: B00+B11
        [1, 0, 0, 0], # M2: B00
        [0, 0, 1, -1],# M3: B10-B11
        [-1, 1, 0, 0],# M4: B01-B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00+B01
        [0, 0, 1, 1]  # M7: B10+B11
    ]
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M1 - M2 + M6
    # C10 = M2 + M4 - M1 + M5
    # C11 = M3 + M7 - M4 + M6
    w_strassen = [
        [1, 0, 0, 0], # M1
        [0, 0, 0, 0], # M2
        [0, 0, 0, 0], # M3
        [0, 0, 0, 0], # M4
        [0, 0, 0, 0], # M5
        [0, 0, 0, 0], # M6
        [0, 0, 0, 0]  # M7
    ]
    # Actually, w_strassen should be the coefficients for C_ik in terms of M_s.
    # C_00 = 1*M1 + 0*M2 + 0*M3 + 1*M4 - 1*M5 + 0*M6 + 1*M7
    # C_01 = 0*M1 - 1*M2 + 1*M3 + 0*M4 + 0*M5 + 1*M6 + 0*M7
    # C_10 = 0*M1 + 1*M2 + 0*M3 + 1*M4 + 1*M5 + 0*M6 + 0*M7
    # C_11 = 0*M1 + 0*M2 + 1*M3 - 1*M4 + 0*M5 + 1*M6 + 1*M7
    
    w_coeff = [
        [1, -1, 0, 1], # C00: M1-M2+M4+M7? No.
        # Let's re-verify Strassen output.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M1 - M2 + M6
        # C10 = M2 + M4 - M1 + M5
        # C11 = M3 + M7 - M4 + M6
        
        # So w_coeff[ik][s]
        [1, 0, 0, 1, -1, 0, 1], # C00
        [1, -1, 1, 0, 0, 1, 0], # C01
        [-1, 1, 0, 1, 1, 0, 0], # C10
        [0, 0, 1, -1, 0, 1, 1]  # C11
    ]
    
    # Now for the 4x4 case.
    # Blocks A00, A01, A10, A11.
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01-B00)
    # P5 = (A00+A10)B11
    # P6 = (A11-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # c_u[s][IJ] coefficient of block A_IJ in A^(s)
    c_u = [
        [1, 0, 0, 1], # P1: A00+A11
        [0, 0, 1, 1], # P2: A10+A11
        [1, 0, 0, 0], # P3: A00
        [0, 0, 0, 1], # P4: A11
        [1, 0, 1, 0], # P5: A00+A10
        [-1, 0, 0, 1],# P6: A11-A00
        [0, 1, 0, -1] # P7: A01-A11
    ]
    
    c_v = [
        [1, 1, 0, 0], # P1: B00+B11
        [1, 0, 0, 0], # P2: B00
        [0, 0, 1, -1],# P3: B10-B11
        [-1, 1, 0, 0],# P4: B01-B00
        [0, 0, 0, 1], # P5: B11
        [1, 1, 0, 0], # P6: B00+B01
        [0, 0, 1, 1]  # P7: B10+B11
    ]
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P1 - P2 + P6
    # C10 = P2 + P4 - P1 + P5
    # C11 = P3 + P7 - P4 + P6
    
    c_w = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [1, -1, 1, 0, 0, 1, 0], # C01
        [-1, 1, 0, 1, 1, 0, 0], # C10
        [0, 0, 1, -1, 0, 1, 1]  # C11
    ]
    
    col = 0
    for s in range(7):
        for l in range(7):
            # U part
            # U[i*m+j, col]
            # i = 2*I + alpha, j = 2*J + beta
            # U = sum_uv u_strassen[l][uv] * c_u[s][IJ]
            # u_strassen[l] is for A^(s) entries.
            # A^(s)_uv is a block.
            # So U_{IJ, alpha beta} = sum_uv u_strassen[l][uv] * c_u[s][IJ]
            # This is independent of alpha, beta.
            
            for I in range(2):
                for J in range(2):
                    val_u = 0
                    for uv in range(4):
                        # uv corresponds to block index in A^(s)
                        # u_strassen[l][uv] is coeff for A^(s)_uv
                        # c_u[s][IJ] is coeff for A_IJ in A^(s)_uv?
                        # No, A^(s) is a 2x2 matrix of blocks.
                        # A^(s)_uv = sum_IJ c_u[s][IJ] A_IJ.
                        # So the coefficient of A_IJ in A^(s)_uv is c_u[s][IJ].
                        # The linear form for M_l is sum_uv u_strassen[l][uv] A^(s)_uv.
                        # = sum_uv u_strassen[l][uv] (sum_IJ c_u[s][IJ] A_IJ)
                        # = sum_IJ (sum_uv u_strassen[l][uv] c_u[s][IJ]) A_IJ.
                        val_u += u_strassen[l][uv] * c_u[s][IJ]
                    
                    for alpha in range(2):
                        for beta in range(2):
                            i = 2*I + alpha
                            j = 2*J + beta
                            U[i*m+j][col] = val_u
                            
            # V part
            # V[j*p+k, col]
            # j = 2*J + beta, k = 2*K + gamma
            # V_{JK, beta gamma} = sum_uv v_strassen[l][uv] * c_v[s][JK]
            
            for J in range(2):
                for K in range(2):
                    val_v = 0
                    for uv in range(4):
                        val_v += v_strassen[l][uv] * c_v[s][JK]
                    
                    for beta in range(2):
                        for gamma in range(2):
                            j = 2*J + beta
                            k = 2*K + gamma
                            V[j*p+k][col] = val_v
                            
            # W part
            # W[k*n+i, col]
            # k = 2*K + gamma, i = 2*I + alpha
            # W_{KI, gamma alpha} = w_strassen[l][gamma alpha] * c_w[s][KI]
            # w_strassen[l] is coeff for C_ik in M_l.
            # C_ik = sum_l w_strassen[l][ik] M_l.
            # M_l = sum_s c_w[s][KI] P_s? No.
            # C_KI = sum_s c_w[s][KI] P_s.
            # P_s = sum_l w_strassen[l][uv] M_{s,l}?
            # P_s is a block matrix.
            # P_s,uv = sum_l w_strassen[l][uv] M_{s,l}.
            # C_KI, alpha beta = sum_s c_w[s][KI] P_s, alpha beta?
            # No, C_KI is the block at KI.
            # C_KI = sum_s c_w[s][KI] P_s.
            # So C_KI, alpha beta = sum_s c_w[s][KI] P_s, alpha beta.
            # P_s, alpha beta = sum_l w_strassen[l][alpha beta] M_{s,l}.
            # So C_KI, alpha beta = sum_l (sum_s c_w[s][KI] w_strassen[l][alpha beta]) M_{s,l}.
            # So W_{KI, gamma alpha} = sum_s c_w[s][KI] w_strassen[l][alpha beta]?
            # Wait, w_strassen[l][ik] is the coeff of M_l in C_ik.
            # Here C_ik refers to the 2x2 matrix C^(s).
            # So yes.
            
            for K in range(2):
                for I in range(2):
                    for gamma in range(2):
                        for alpha in range(2):
                            val_w = 0
                            # w_strassen[l][alpha*2+beta]?
                            # w_strassen is 4x7. w_strassen[ik][l].
                            # We need w_strassen[alpha*2+beta][l]?
                            # No, w_strassen[l] is the vector of coeffs for C_00..C_11.
                            # w_strassen[l][uv] is coeff of M_l in C_uv.
                            # Here uv corresponds to local indices alpha, beta.
                            val_w += c_w[s][K*2+I] * w_strassen[alpha*2+beta][l]
                            
                            k = 2*K + gamma
                            i = 2*I + alpha
                            W[k*n+i][col] = val_w
                            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

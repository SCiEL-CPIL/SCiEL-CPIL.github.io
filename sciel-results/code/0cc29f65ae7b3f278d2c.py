import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 49
    rank = 49
    
    # Initialize U, V, W
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Outer Strassen coefficients w[ab][r]
    # ab is 0..3 for blocks (00, 01, 10, 11) -> a,b in 0..1
    # r is 0..6 for M1..M7
    w = [[0.0]*7 for _ in range(4)]
    # C11 = M1 + M4 - M5 + M7
    w[0][0] = 1.0; w[0][3] = 1.0; w[0][4] = -1.0; w[0][6] = 1.0
    # C12 = M3 + M5
    w[1][2] = 1.0; w[1][4] = 1.0
    # C21 = M2 + M4
    w[2][1] = 1.0; w[2][3] = 1.0
    # C22 = M1 - M2 + M3 + M6
    w[3][0] = 1.0; w[3][1] = -1.0; w[3][2] = 1.0; w[3][5] = 1.0
    
    # Outer M_r definitions u[r][ab] and v[r][cd]
    u = [[0.0]*4 for _ in range(7)]
    v = [[0.0]*4 for _ in range(7)]
    
    # M1: (A11+A22)(B11+B22)
    u[0][0] = 1.0; u[0][3] = 1.0
    v[0][0] = 1.0; v[0][3] = 1.0
    # M2: (A21+A22)B11
    u[1][2] = 1.0; u[1][3] = 1.0
    v[1][0] = 1.0
    # M3: A11(B12-B22)
    u[2][0] = 1.0
    v[2][1] = 1.0; v[2][3] = -1.0
    # M4: A22(B21-B11)
    u[3][3] = 1.0
    v[3][2] = 1.0; v[3][0] = -1.0
    # M5: (A11+A12)B22
    u[4][0] = 1.0; u[4][1] = 1.0
    v[4][3] = 1.0
    # M6: (A21-A11)(B11+B12)
    u[5][2] = 1.0; u[5][0] = -1.0
    v[5][0] = 1.0; v[5][1] = 1.0
    # M7: (A12-A22)(B21+B22)
    u[6][1] = 1.0; u[6][3] = -1.0
    v[6][2] = 1.0; v[6][3] = 1.0
    
    # Inner Strassen products P_s definitions
    # P_s = L_X(X) * L_Y(Y)
    # L_X and L_Y are linear forms on 2x2 matrices.
    # We represent L_X by coefficients x[4] for x00, x01, x10, x11
    # and L_Y by coefficients y[4] for y00, y01, y10, y11
    
    inner_u = [[0.0]*4 for _ in range(7)]
    inner_v = [[0.0]*4 for _ in range(7)]
    
    # P1: (x00+x11)(y00+y11)
    inner_u[0][0] = 1.0; inner_u[0][3] = 1.0
    inner_v[0][0] = 1.0; inner_v[0][3] = 1.0
    # P2: (x10+x11)y00
    inner_u[1][2] = 1.0; inner_u[1][3] = 1.0
    inner_v[1][0] = 1.0
    # P3: x00(y01-y11)
    inner_u[2][0] = 1.0
    inner_v[2][1] = 1.0; inner_v[2][3] = -1.0
    # P4: x11(y10-y00)
    inner_u[3][3] = 1.0
    inner_v[3][2] = 1.0; inner_v[3][0] = -1.0
    # P5: (x00+x01)y11
    inner_u[4][0] = 1.0; inner_u[4][1] = 1.0
    inner_v[4][3] = 1.0
    # P6: (x10-x00)(y00+y01)
    inner_u[5][2] = 1.0; inner_u[5][0] = -1.0
    inner_v[5][0] = 1.0; inner_v[5][1] = 1.0
    # P7: (x01-x11)(y10+y11)
    inner_u[6][1] = 1.0; inner_u[6][3] = -1.0
    inner_v[6][2] = 1.0; inner_v[6][3] = 1.0
    
    # Inner output coefficients w_prime[alpha_beta][s]
    # alpha_beta is 0..3 for C00, C01, C10, C11
    w_prime = [[0.0]*7 for _ in range(4)]
    # C00 = P1 + P4 - P5 + P7
    w_prime[0][0] = 1.0; w_prime[0][3] = 1.0; w_prime[0][4] = -1.0; w_prime[0][6] = 1.0
    # C01 = P3 + P5
    w_prime[1][2] = 1.0; w_prime[1][4] = 1.0
    # C10 = P2 + P4
    w_prime[2][1] = 1.0; w_prime[2][3] = 1.0
    # C11 = P1 - P2 + P3 + P6
    w_prime[3][0] = 1.0; w_prime[3][1] = -1.0; w_prime[3][2] = 1.0; w_prime[3][5] = 1.0
    
    # Construct U, V, W
    col = 0
    for r in range(7):
        for s in range(7):
            # Calculate coefficients for A and B for this term (r, s)
            # L_A^{(r,s)} = sum_{i,j} U_{ij, col} A_{ij}
            # L_B^{(r,s)} = sum_{j,k} V_{jk, col} B_{jk}
            
            # L_A is derived from inner_u[s] applied to X_r
            # X_r[alpha, beta] = sum_{a,b} u[r][ab] A^{ab}[alpha, beta]
            # A^{ab}[alpha, beta] is A_{2a+alpha, 2b+beta}
            
            # So coeff of A_{ij} where i=2a+alpha, j=2b+beta is:
            # inner_u[s][alpha*2+beta] * u[r][a*2+b]
            
            # L_B is derived from inner_v[s] applied to Y_r
            # Y_r[gamma, delta] = sum_{c,d} v[r][cd] B^{cd}[gamma, delta]
            # B^{cd}[gamma, delta] is B_{2c+gamma, 2d+delta}
            
            # So coeff of B_{jk} where j=2c+gamma, k=2d+delta is:
            # inner_v[s][gamma*2+delta] * v[r][c*2+d]
            
            # W coefficient
            # K_{ik}^{(r,s)} where i=2a+alpha, k=2b+beta
            # is w[a*2+b][r] * w_prime[alpha*2+beta][s]
            
            for a in range(2):
                for b in range(2):
                    for alpha in range(2):
                        for beta in range(2):
                            i = 2*a + alpha
                            j = 2*b + beta
                            coeff_U = inner_u[s][alpha*2+beta] * u[r][a*2+b]
                            if coeff_U != 0:
                                U[i*4 + j][col] = coeff_U
                            
                            # Note: j in B is the same j as in A?
                            # In matrix mult C_ik = sum_j A_ij B_jk.
                            # The term is A_ij * B_jk.
                            # Our construction:
                            # A part: A_ij
                            # B part: B_jk
                            # The index j in A_ij is 2b+beta.
                            # The index j in B_jk must be the same.
                            # In Y_r, the row index is gamma.
                            # B_jk has row j. So j = 2c+gamma.
                            # So we must have 2b+beta = 2c+gamma => b=c and beta=gamma.
                            
                            # Wait, the tensor term is A_ij B_jk.
                            # The linear form L_A picks A_ij.
                            # The linear form L_B picks B_jk.
                            # The product is A_ij B_jk.
                            # This term contributes to C_ik.
                            # The coefficient in W is for C_ik.
                            
                            # So for a fixed (r,s), we sum over all i,j,k?
                            # No, U_{ij, col} and V_{jk, col} are fixed for this col.
                            # The product U_{ij, col} V_{jk, col} is non-zero only if
                            # L_A has A_ij and L_B has B_jk.
                            
                            # Let's fill V and W as well.
                            
            # Fill V
            for c in range(2):
                for d in range(2):
                    for gamma in range(2):
                        for delta in range(2):
                            j = 2*c + gamma
                            k = 2*d + delta
                            coeff_V = inner_v[s][gamma*2+delta] * v[r][c*2+d]
                            if coeff_V != 0:
                                V[j*4 + k][col] = coeff_V
            
            # Fill W
            # W index is k*n + i = k*4 + i.
            # K_{ik} = w[a*2+b][r] * w_prime[alpha*2+beta][s]
            # i = 2a+alpha, k = 2b+beta?
            # Wait, C^{ab} is block at row a, col b.
            # C_ik with i=2a+alpha, k=2b+beta.
            # So for each a, b, alpha, beta:
            for a in range(2):
                for b in range(2):
                    for alpha in range(2):
                        for beta in range(2):
                            i = 2*a + alpha
                            k = 2*b + beta
                            coeff_W = w[a*2+b][r] * w_prime[alpha*2+beta][s]
                            if coeff_W != 0:
                                W[k*4 + i][col] = coeff_W
            
            col += 1
            
    return {"U": U, "V": V, "W": W}

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Part 1: Strassen 4x4x4 for A[0:4,0:4] * B[0:4,0:4] -> C[0:4,0:4]
    # Rank 49
    # Base 2x2x2 Strassen
    # U_base, V_base, W_base are 4x7
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    U_base = [
        [1, 1, 1, 0, 1, -1, 0], # i=0,j=0
        [0, 0, 0, 0, 0, 0, -1], # i=0,j=1
        [0, 0, 0, 0, 0, 1, 1],  # i=1,j=0
        [1, 1, 0, 1, 1, 0, -1]  # i=1,j=1
    ]
    # Wait, let's use the standard Strassen coefficients carefully.
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # U_base[c, i*2+j] is coeff of A_ij in P_c
    # V_base[c, j*2+k] is coeff of B_jk in Q_c
    # W_base[c, k*2+i] is coeff of M_c in C_ki
    
    # c=0 (M1): P=A11+A22, Q=B11+B22
    # c=1 (M2): P=A21+A22, Q=B11
    # c=2 (M3): P=A11, Q=B12-B22
    # c=3 (M4): P=A22, Q=B21-B11
    # c=4 (M5): P=A11+A12, Q=B22
    # c=5 (M6): P=A21-A11, Q=B11+B12
    # c=6 (M7): P=A12-A22, Q=B21+B22
    
    U_base = [
        [1, 0, 0, 1], # M1: A11+A22
        [0, 0, 1, 1], # M2: A21+A22
        [1, 0, 0, 0], # M3: A11
        [0, 0, 0, 1], # M4: A22
        [1, 1, 0, 0], # M5: A11+A12
        [-1, 0, 1, 0],# M6: A21-A11
        [0, 1, 0, -1] # M7: A12-A22
    ]
    V_base = [
        [1, 1, 0, 0], # M1: B11+B22
        [1, 0, 0, 0], # M2: B11
        [0, 1, 0, -1],# M3: B12-B22
        [0, 0, 1, -1],# M4: B21-B11
        [0, 0, 0, 1], # M5: B22
        [1, 1, 0, 0], # M6: B11+B12
        [0, 0, 1, 1]  # M7: B21+B22
    ]
    W_base = [
        [1, 0, 0, 0], # M1 in C11
        [0, 0, 0, 0], # M2 in C11? No.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # W_base[c, k*2+i]
        # c=0 (M1): C11+=1, C22+=1
        # c=1 (M2): C21+=1, C22+=-1
        # c=2 (M3): C12+=1, C22+=1
        # c=3 (M4): C11+=1, C21+=1
        # c=4 (M5): C11+=-1, C12+=1
        # c=5 (M6): C22+=1
        # c=6 (M7): C11+=1
    ]
    W_base = [
        [1, 0, 0, 1], # M1: C11, C22
        [0, 0, 1, -1],# M2: C21, C22
        [0, 1, 0, 1], # M3: C12, C22
        [1, 0, 1, 0], # M4: C11, C21
        [-1, 1, 0, 0],# M5: C11, C12
        [0, 0, 0, 1], # M6: C22
        [1, 0, 0, 0]  # M7: C11
    ]
    
    # 4x4 Strassen Coefficients
    # P_j = sum IJ p[j, IJ, alpha, beta] A[I*2+alpha, J*2+beta]
    # Q_j = sum JK q[j, JK, alpha, beta] B[J*2+alpha, K*2+beta]
    # C_IK = sum j w[j, IK, alpha, beta] M_j[alpha, beta]
    
    # p[j, I, J, a, b]
    p_coeffs = {}
    q_coeffs = {}
    w_coeffs = {}
    
    # M1: P=A11+A22, Q=B11+B22
    p_coeffs[(0, 0, 0, 0, 0)] = 1; p_coeffs[(0, 0, 0, 1, 1)] = 1
    p_coeffs[(0, 1, 1, 0, 0)] = 1; p_coeffs[(0, 1, 1, 1, 1)] = 1
    # Wait, indices: p[j, I, J, a, b]
    # A11 is I=0, J=0. A22 is I=1, J=1.
    # P1 = A11 + A22.
    # p[0, 0, 0, a, b] = 1 for all a,b
    # p[0, 1, 1, a, b] = 1 for all a,b
    for a in range(2):
        for b in range(2):
            p_coeffs[(0, 0, 0, a, b)] = 1
            p_coeffs[(0, 1, 1, a, b)] = 1
            q_coeffs[(0, 0, 0, a, b)] = 1 # B11
            q_coeffs[(0, 1, 1, a, b)] = 1 # B22
            # w[0, I, K, a, b]
            w_coeffs[(0, 0, 0, a, b)] = 1 # C11
            w_coeffs[(0, 1, 1, a, b)] = 1 # C22

    # M2: P=A21+A22, Q=B11
    for a in range(2):
        for b in range(2):
            p_coeffs[(1, 1, 0, a, b)] = 1 # A21
            p_coeffs[(1, 1, 1, a, b)] = 1 # A22
            q_coeffs[(1, 0, 0, a, b)] = 1 # B11
            w_coeffs[(1, 1, 0, a, b)] = 1 # C21
            w_coeffs[(1, 1, 1, a, b)] = -1 # C22

    # M3: P=A11, Q=B12-B22
    for a in range(2):
        for b in range(2):
            p_coeffs[(2, 0, 0, a, b)] = 1
            q_coeffs[(2, 0, 1, a, b)] = 1 # B12
            q_coeffs[(2, 1, 1, a, b)] = -1 # B22
            w_coeffs[(2, 0, 1, a, b)] = 1 # C12
            w_coeffs[(2, 1, 1, a, b)] = 1 # C22

    # M4: P=A22, Q=B21-B11
    for a in range(2):
        for b in range(2):
            p_coeffs[(3, 1, 1, a, b)] = 1
            q_coeffs[(3, 1, 0, a, b)] = 1 # B21
            q_coeffs[(3, 0, 0, a, b)] = -1 # B11
            w_coeffs[(3, 0, 0, a, b)] = 1 # C11
            w_coeffs[(3, 1, 0, a, b)] = 1 # C21

    # M5: P=A11+A12, Q=B22
    for a in range(2):
        for b in range(2):
            p_coeffs[(4, 0, 0, a, b)] = 1
            p_coeffs[(4, 0, 1, a, b)] = 1 # A12
            q_coeffs[(4, 1, 1, a, b)] = 1 # B22
            w_coeffs[(4, 0, 0, a, b)] = -1 # C11
            w_coeffs[(4, 0, 1, a, b)] = 1 # C12

    # M6: P=A21-A11, Q=B11+B12
    for a in range(2):
        for b in range(2):
            p_coeffs[(5, 1, 0, a, b)] = 1
            p_coeffs[(5, 0, 0, a, b)] = -1
            q_coeffs[(5, 0, 0, a, b)] = 1
            q_coeffs[(5, 0, 1, a, b)] = 1
            w_coeffs[(5, 1, 1, a, b)] = 1 # C22

    # M7: P=A12-A22, Q=B21+B22
    for a in range(2):
        for b in range(2):
            p_coeffs[(6, 0, 1, a, b)] = 1
            p_coeffs[(6, 1, 1, a, b)] = -1
            q_coeffs[(6, 1, 0, a, b)] = 1
            q_coeffs[(6, 1, 1, a, b)] = 1
            w_coeffs[(6, 0, 0, a, b)] = 1 # C11

    rank1 = 49
    U1 = [[0]*rank1 for _ in range(n*m)]
    V1 = [[0]*rank1 for _ in range(m*p)]
    W1 = [[0]*rank1 for _ in range(p*n)]
    
    col = 0
    for j in range(7):
        for c in range(7):
            # U part
            for i in range(4):
                for jj in range(4):
                    I = i // 2
                    J = jj // 2
                    a = i % 2
                    b = jj % 2
                    coeff = p_coeffs.get((j, I, J, a, b), 0)
                    if coeff != 0:
                        U1[i*5+jj][col] += coeff * U_base[c][i*2+jj] # Wait, U_base index is local i*2+j
                        # Local index in U_base is a*2+b
                        # U_base[c][a*2+b]
                        U1[i*5+jj][col] = U1[i*5+jj][col] # Reset? No, accumulate.
                        # Actually, U_base[c] is a vector of length 4.
                        # The term is sum_{a,b} U_base[c, a*2+b] * p_coeffs[j, I, J, a, b] * A[I*2+a, J*2+b]
                        # So U1[I*2+a, J*2+b] gets contribution U_base[c, a*2+b] * p_coeffs...
                        pass
            
            # Let's re-do the accumulation properly.
            # U1[row, col] = sum_{j,c} U_base[c, a*2+b] * p_coeffs[j, I, J, a, b]
            # where row = I*2+a (local in 4x4) -> global row = I*2+a * 5 + (J*2+b)
            # Wait, A is 4x5. A1 is 4x4.
            # Global row index for A[I*2+a, J*2+b] is (I*2+a)*5 + (J*2+b).
            
            for a in range(2):
                for b in range(2):
                    ub = U_base[c][a*2+b]
                    if ub == 0: continue
                    for I in range(2):
                        for J in range(2):
                            pc = p_coeffs.get((j, I, J, a, b), 0)
                            if pc == 0: continue
                            row = (I*2+a)*5 + (J*2+b)
                            U1[row][col] += ub * pc

            # V part
            # V1[row, col] = sum_{j,c} V_base[c, a*2+b] * q_coeffs[j, J, K, a, b]
            # Global row for B[J*2+a, K*2+b] is (J*2+a)*5 + (K*2+b).
            for a in range(2):
                for b in range(2):
                    vb = V_base[c][a*2+b]
                    if vb == 0: continue
                    for J in range(2):
                        for K in range(2):
                            qc = q_coeffs.get((j, J, K, a, b), 0)
                            if qc == 0: continue
                            row = (J*2+a)*5 + (K*2+b)
                            V1[row][col] += vb * qc

            # W part
            # W1[row, col] = sum_{j,c} W_base[c, a*2+b] * w_coeffs[j, I, K, a, b]
            # Global row for C[I*2+a, K*2+b] is (K*2+b)*4 + (I*2+a).
            for a in range(2):
                for b in range(2):
                    wb = W_base[c][a*2+b]
                    if wb == 0: continue
                    for I in range(2):
                        for K in range(2):
                            wc = w_coeffs.get((j, I, K, a, b), 0)
                            if wc == 0: continue
                            row = (K*2+b)*4 + (I*2+a)
                            W1[row][col] += wb * wc
                            
            col += 1

    # Part 2: Schoolbook A[0:4,0:4] * B[0:4,4] -> C[0:4,4]
    rank2 = 16
    U2 = [[0]*rank2 for _ in range(n*m)]
    V2 = [[0]*rank2 for _ in range(m*p)]
    W2 = [[0]*rank2 for _ in range(p*n)]
    
    col = 0
    for i in range(4):
        for j in range(4):
            U2[i*5+j][col] = 1
            V2[j*5+4][col] = 1
            W2[4*4+i][col] = 1
            col += 1

    # Part 3: Schoolbook A[0:4,4] * B[4,0:5] -> C[0:4,0:5]
    rank3 = 20
    U3 = [[0]*rank3 for _ in range(n*m)]
    V3 = [[0]*rank3 for _ in range(m*p)]
    W3 = [[0]*rank3 for _ in range(p*n)]
    
    col = 0
    for i in range(4):
        for k in range(5):
            U3[i*5+4][col] = 1
            V3[4*5+k][col] = 1
            W3[k*4+i][col] = 1
            col += 1

    total_rank = rank1 + rank2 + rank3
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    for i in range(n*m):
        for c in range(rank1):
            U[i][c] = U1[i][c]
        for c in range(rank2):
            U[i][c+rank1] = U2[i][c]
        for c in range(rank3):
            U[i][c+rank1+rank2] = U3[i][c]
            
    for i in range(m*p):
        for c in range(rank1):
            V[i][c] = V1[i][c]
        for c in range(rank2):
            V[i][c+rank1] = V2[i][c]
        for c in range(rank3):
            V[i][c+rank1+rank2] = V3[i][c]
            
    for i in range(p*n):
        for c in range(rank1):
            W[i][c] = W1[i][c]
        for c in range(rank2):
            W[i][c+rank1] = W2[i][c]
        for c in range(rank3):
            W[i][c+rank1+rank2] = W3[i][c]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

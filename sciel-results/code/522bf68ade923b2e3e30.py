"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Rank-7 Strassen decomposition for 2x2x2
    # A index: i*2+j (i,j in 0..1) -> [0,1,2,3]
    # B index: j*2+k (j,k in 0..1) -> [0,1,2,3]  
    # C index: k*2+i (k,i in 0..1) -> [0,1,2,3]
    # C[k,i] = sum_j A[i,j] * B[j,k]
    
    # Strassen:
    # m0 = (a00+a11)(b00+b11)
    # m1 = (a10+a11)(b00)
    # m2 = (a00)(b01-b11)
    # m3 = (a11)(b10-b00)
    # m4 = (a00+a01)(b11)
    # m5 = (a10-a00)(b00+b01)
    # m6 = (a01-a11)(b10+b11)
    #
    # c00 = m0 + m3 - m4 + m6
    # c01 = m2 + m4
    # c10 = m1 + m3
    # c11 = m0 - m1 + m2 + m5
    
    U2 = [
        [1, 0, 0, 1],  # m0: a00+a11
        [0, 0, 1, 1],  # m1: a10+a11
        [1, 0, 0, 0],  # m2: a00
        [0, 0, 0, 1],  # m3: a11
        [1, 1, 0, 0],  # m4: a00+a01
        [1, 0, 1, 0],  # m5: a10-a00
        [0, 1, 0, 1],  # m6: a01-a11
    ]
    V2 = [
        [1, 0, 0, 1],  # m0: b00+b11
        [1, 0, 0, 0],  # m1: b00
        [0, 1, 0, 1],  # m2: b01-b11
        [0, 0, 1, 1],  # m3: b10-b00
        [0, 0, 0, 1],  # m4: b11
        [1, 1, 0, 0],  # m5: b00+b01
        [0, 0, 1, 1],  # m6: b10+b11
    ]
    W2 = [
        [1, 0, 0, 1],   # m0: c00+c11
        [0, 0, 1, 0],   # m1: c10
        [0, 1, 0, 1],   # m2: c01+c11
        [1, 0, 1, 0],   # m3: c00+c10
        [-1, 1, 0, 0],  # m4: -c00+c01
        [0, 0, 0, 1],   # m5: c11
        [1, 0, 0, 0],   # m6: c00
    ]
    
    # For 4x4x4: split into 2x2 blocks
    # A[i,j] with i=2*i1+i2, j=2*j1+j2
    # B[j,k] with j=2*j1+j2, k=2*k1+k2
    # C[k,i] with k=2*k1+k2, i=2*i1+i2
    #
    # Block structure:
    # A = [[A00, A01], [A10, A11]] where Axy is 2x2
    # B = [[B00, B01], [B10, B11]] where Bxy is 2x2
    # C = A*B:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # Strassen on blocks:
    # P0 = (A00+A11)(B00+B11)
    # P1 = (A10+A11)B00
    # P2 = A00(B01-B11)
    # P3 = A11(B10-B00)
    # P4 = (A00+A01)B11
    # P5 = (A10-A00)(B00+B01)
    # P6 = (A01-A11)(B10+B11)
    #
    # C00 = P0 + P3 - P4 + P6
    # C01 = P2 + P4
    # C10 = P1 + P3
    # C11 = P0 - P1 + P2 + P5
    #
    # Each Px is a 2x2 * 2x2 multiplication, rank 7 each.
    # Total: 7*7 = 49 terms for 4x4x4.
    
    # For 4x4x5: C has 5 columns.
    # C[:, 0:4] = A * B[:, 0:4] (4x4x4 part, rank 49)
    # C[:, 4] = A * B[:, 4] (4x4x1 part, rank 16)
    # Total rank: 65
    
    rank_total = 65
    U_out = [[0] * rank_total for _ in range(16)]  # n*m = 4*4 = 16
    V_out = [[0] * rank_total for _ in range(20)]  # m*p = 4*5 = 20
    W_out = [[0] * rank_total for _ in range(20)]  # p*n = 5*4 = 20
    
    # Part 1: 4x4x4 Strassen (49 terms)
    # A index: i*4+j where i=2*i1+i2, j=2*j1+j2
    #   A_idx = (2*i1+i2)*4 + (2*j1+j2) = 8*i1 + 4*i2 + 2*j1 + j2
    # B index (for 4x4 part): j*4+k where j=2*j1+j2, k=2*k1+k2
    #   B_idx = (2*j1+j2)*4 + (2*k1+k2) = 8*j1 + 4*j2 + 2*k1 + k2
    # C index (for 4x4 part): k*4+i where k=2*k1+k2, i=2*i1+i2
    #   C_idx = (2*k1+k2)*4 + (2*i1+i2) = 8*k1 + 4*k2 + 2*i1 + i2
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            # s-th Strassen product for outer 2x2, t-th for inner 2x2
            
            # U coefficient for A[i,j]:
            # A is split into blocks A00, A01, A10, A11
            # Block Axy: A[2*x+dx, 2*y+dy]
            # The s-th Strassen term uses certain blocks with coefficients from U2[s]
            # U2[s] gives coefficients for a00, a01, a10, a11 in the 2x2 case
            # So for outer block (x,y) with x,y in 0..1:
            #   coefficient = U2[s][x*2+y]
            # For inner block (dx,dy) with dx,dy in 0..1:
            #   coefficient = U2[t][dx*2+dy]
            
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            A_idx = 8*i1 + 4*i2 + 2*j1 + j2
                            # Outer block: (i1, j1), inner: (i2, j2)
                            coeff_u = U2[s][i1*2 + j1] * U2[t][i2*2 + j2]
                            U_out[A_idx][col] = coeff_u
            
            # V coefficient for B[j,k] (4x4 part, k in 0..3):
            # B block Bxy: B[2*x+jx, 2*y+ky]
            # coefficient = V2[s][x*2+y] * V2[t][jx*2+ky]
            
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(2):
                            B_idx = 8*j1 + 4*j2 + 2*k1 + k2  # j*4+k for 4x4
                            # Outer block: (j1, k1), inner: (j2, k2)
                            coeff_v = V2[s][j1*2 + k1] * V2[t][j2*2 + k2]
                            V_out[B_idx][col] = coeff_v
            
            # W coefficient for C[k,i] (4x4 part):
            # C block Cxy: C[2*x+kx, 2*y+iy]
            # coefficient = W2[s][x*2+y] * W2[t][kx*2+iy]
            
            for k1 in range(2):
                for k2 in range(2):
                    for i1 in range(2):
                        for i2 in range(2):
                            C_idx = 8*k1 + 4*k2 + 2*i1 + i2  # k*4+i for 4x4
                            # Outer block: (k1, i1), inner: (k2, i2)
                            coeff_w = W2[s][k1*2 + i1] * W2[t][k2*2 + i2]
                            W_out[C_idx][col] = coeff_w
    
    # Part 2: 4x4x1 schoolbook for 5th column (16 terms)
    # C[:,4] = A * B[:,4]
    # C[i,4] = sum_j A[i,j] * B[j,4]
    # C index: 4*4 + i = 16 + i (k=4, i in 0..3)
    # A index: i*4 + j
    # B index: j*5 + 4
    
    for i in range(4):
        for j in range(4):
            col = 49 + i * 4 + j
            A_idx = i * 4 + j
            B_idx = j * 5 + 4
            C_idx = 4 * 4 + i  # k=4, i in 0..3 -> 16+i
            
            U_out[A_idx][col] = 1
            V_out[B_idx][col] = 1
            W_out[C_idx][col] = 1
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Define 2x2 Strassen factors
    # U2, V2, W2 have shape (4, 7)
    # Indices for 2x2: 
    # A: 00, 01, 10, 11 -> indices 0, 1, 2, 3 (i*m + j)
    # B: 00, 01, 10, 11 -> indices 0, 1, 2, 3 (j*p + k)
    # C: 00, 01, 10, 11 -> indices 0, 1, 2, 3 (k*n + i) ? 
    # Let's check the W index convention from the prompt: W[k*n+i, r]
    # For 2x2, n=2. W index = k*2 + i.
    # If C is 2x2, C_{ki} is at k*2+i.
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # Strassen 7 products
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Indices in 2x2:
    # A00=0, A01=1, A10=2, A11=3
    # B00=0, B01=1, B10=2, B11=3
    # C00=0 (k=0,i=0), C01=1 (k=0,i=1), C10=2 (k=1,i=0), C11=3 (k=1,i=1)
    
    # Term 0 (M1)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1 # C00 and C11 get M1? 
    # Wait, C00 = M1 + M4 - M5 + M7
    # C11 = M1 - M2 + M3 + M6
    # So W2[0][0]=1, W2[3][0]=1 is correct for M1 contribution to C00 and C11.
    
    # Term 1 (M2)
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[3][1] = -1 # C11 gets -M2
    # C10 = M2 + M4. So W2[2][1] = 1
    W2[2][1] = 1
    
    # Term 2 (M3)
    U2[0][2] = 1
    V2[2][2] = 1; V2[3][2] = -1
    W2[1][2] = 1 # C01 gets M3
    W2[3][2] = 1 # C11 gets M3
    
    # Term 3 (M4)
    U2[3][3] = 1
    V2[1][3] = 1; V2[0][3] = -1
    W2[0][3] = 1 # C00 gets M4
    W2[2][3] = 1 # C10 gets M4
    
    # Term 4 (M5)
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1 # C00 gets -M5
    W2[1][4] = 1  # C01 gets M5
    
    # Term 5 (M6)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1  # C11 gets M6
    
    # Term 6 (M7)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1  # C00 gets M7
    
    # Now construct 4x4 factors using Kronecker product
    # Rank = 7 * 7 = 49
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for i0 in range(2):
                    # Index in 4x4 A: i = 2*i1 + i0, j = 2*j1 + j0
                    # U index: i*4 + j
                    # U2 index: i1*2 + i0
                    u_val = U2[i1 * 2 + i0][r1]
                    if u_val != 0:
                        for j1 in range(2):
                            for j0 in range(2):
                                # V2 index: j1*2 + j0
                                v_val = V2[j1 * 2 + j0][r1]
                                if v_val != 0:
                                    # U index in 4x4: (2*i1+i0)*4 + (2*j1+j0)
                                    u_idx = (2 * i1 + i0) * 4 + (2 * j1 + j0)
                                    U[u_idx][r] = u_val * v_val
                                # Wait, U only depends on i,j and r1? No.
                                # U factor in Kronecker product: U4[i,j, r1,r2] = U2[i1,j1, r1] * U2[i0,j0, r2]?
                                # No, the tensor is T = T2 \otimes T2.
                                # T2_{i,j,k} = sum_r U2_{i,j,r} V2_{j,k,r} W2_{k,i,r}
                                # T4_{i,j,k} = sum_{r1,r2} (U2_{i1,j1,r1} U2_{i0,j0,r2}) (V2_{j1,k1,r1} V2_{j0,k0,r2}) (W2_{k1,i1,r1} W2_{k0,i0,r2})
                                # So U4_{i,j, r1,r2} = U2_{i1,j1,r1} * U2_{i0,j0,r2}
                                # This means U4 depends on r1 and r2 separately.
                                # My loop structure above was mixing them.
                                pass
                                
    # Correct construction:
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    u1 = U2[i1 * 2 + j1][r1]
                    if u1 == 0: continue
                    for i0 in range(2):
                        for j0 in range(2):
                            u0 = U2[i0 * 2 + j0][r2]
                            if u0 == 0: continue
                            u_val = u1 * u0
                            u_idx = (2 * i1 + i0) * 4 + (2 * j1 + j0)
                            U[u_idx][r] = u_val
                            
                            for k1 in range(2):
                                v1 = V2[j1 * 2 + k1][r1]
                                if v1 == 0: continue
                                for k0 in range(2):
                                    v0 = V2[j0 * 2 + k0][r2]
                                    if v0 == 0: continue
                                    v_val = v1 * v0
                                    v_idx = (2 * j1 + j0) * 4 + (2 * k1 + k0)
                                    V[v_idx][r] = v_val
                                    
                                    for _ in range(1): # Just to fill W
                                        # W index: k*n + i
                                        # k = 2*k1 + k0, i = 2*i1 + i0
                                        w1 = W2[k1 * 2 + i1][r1]
                                        w0 = W2[k0 * 2 + i0][r2]
                                        w_val = w1 * w0
                                        w_idx = (2 * k1 + k0) * 4 + (2 * i1 + i0)
                                        W[w_idx][r] = w_val
                                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

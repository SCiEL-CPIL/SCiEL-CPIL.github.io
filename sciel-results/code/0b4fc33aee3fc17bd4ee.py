"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 decomposition using recursive Strassen
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrices
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U, V, W for 2x2 Strassen
    # Indices for 2x2: i,j in {0,1}
    # U[i*2+j, t], V[j*2+k, t], W[k*2+i, t]
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    def set2(u, v, w, i, j, k, t, val):
        U2[i*2+j][t] = val
        V2[j*2+k][t] = val
        W2[k*2+i][t] = val
        
    # t=0: M1
    # A00, A11; B00, B11
    set2(1, 1, 1, 0, 0, 0, 0, 1)
    set2(1, 1, 1, 1, 1, 1, 0, 1)
    # Note: M1 = (A00+A11)(B00+B11)
    # Terms: A00B00, A00B11, A11B00, A11B11
    # This expansion is not how Strassen works. Strassen uses 7 products.
    # The decomposition is linear.
    # U, V, W are linear forms.
    
    # Let's use the standard Strassen linear forms.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need U, V, W such that sum U V W = C.
    # For 2x2, rank 7.
    
    # Re-initialize U2, V2, W2
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # t=0 corresponds to M1
    # U part: A00+A11. V part: B00+B11.
    # This is not a single term in the decomposition.
    # The decomposition is:
    # C = sum_{t=1}^7 U_t V_t W_t
    # Where U_t, V_t, W_t are vectors.
    
    # Let's define the 7 terms explicitly for 2x2.
    # Term 1: M1
    # U1: A00+A11 -> [1,0,0,1] (for A00,A01,A10,A11)
    # V1: B00+B11 -> [1,0,0,1] (for B00,B01,B10,B11)
    # W1: C00+C11? No.
    # M1 contributes to C00 and C11.
    # C00 += M1, C11 += M1.
    # So W1 must have 1 at C00 and 1 at C11.
    # W1: [1,0,0,1] (for C00,C01,C10,C11)
    
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # Term 2: M2 = (A10+A11)B00
    # U2: [0,0,1,1]
    # V2: [1,0,0,0]
    # W2: C10 += M2. So W2[2]=1.
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1
    
    # Term 3: M3 = A00(B01-B11)
    # U3: [1,0,0,0]
    # V3: [0,1,0,-1]
    # W3: C01 += M3. So W3[1]=1.
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1
    
    # Term 4: M4 = A11(B10-B00)
    # U4: [0,0,0,1]
    # V4: [0,0,1,-1]
    # W4: C00 += M4, C10 += M4. So W4[0]=1, W4[2]=1.
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1
    
    # Term 5: M5 = (A00+A01)B11
    # U5: [1,1,0,0]
    # V5: [0,0,0,1]
    # W5: C01 += M5, C00 -= M5. So W5[1]=1, W5[0]=-1.
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[1][4] = 1; W2[0][4] = -1
    
    # Term 6: M6 = (A10-A00)(B00+B01)
    # U6: [0,0,1,-1]
    # V6: [1,1,0,0]
    # W6: C11 += M6. So W6[3]=1.
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # Term 7: M7 = (A01-A11)(B10+B11)
    # U7: [0,1,0,-1]
    # V7: [0,0,1,1]
    # W7: C00 += M7. So W7[0]=1.
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1

    # Now construct 4x4 decomposition
    # Outer Strassen terms
    # S1: (A00+A11)(B00+B11)
    # S2: (A10+A11)B00
    # S3: A00(B01-B11)
    # S4: A11(B10-B00)
    # S5: (A00+A01)B11
    # S6: (A10-A00)(B00+B01)
    # S7: (A01-A11)(B10+B11)
    
    # Block indices: 00, 01, 10, 11
    # For each outer term s, we have linear forms for blocks.
    # Then for each inner term t, we combine.
    
    # Outer Coefficients
    # U_outer[s][block_idx]
    # V_outer[s][block_idx]
    # W_outer[s][block_idx] (for output blocks)
    
    U_out = [[0]*4 for _ in range(7)]
    V_out = [[0]*4 for _ in range(7)]
    W_out = [[0]*4 for _ in range(7)]
    
    # S1
    U_out[0][0] = 1; U_out[0][3] = 1
    V_out[0][0] = 1; V_out[0][3] = 1
    W_out[0][0] = 1; W_out[0][3] = 1
    
    # S2
    U_out[1][2] = 1; U_out[1][3] = 1
    V_out[1][0] = 1
    W_out[1][2] = 1
    
    # S3
    U_out[2][0] = 1
    V_out[2][1] = 1; V_out[2][3] = -1
    W_out[2][1] = 1
    
    # S4
    U_out[3][3] = 1
    V_out[3][2] = 1; V_out[3][0] = -1
    W_out[3][0] = 1; W_out[3][2] = 1
    
    # S5
    U_out[4][0] = 1; U_out[4][1] = 1
    V_out[4][3] = 1
    W_out[4][1] = 1; W_out[4][0] = -1
    
    # S6
    U_out[5][2] = 1; U_out[5][0] = -1
    V_out[5][0] = 1; V_out[5][1] = 1
    W_out[5][3] = 1
    
    # S7
    U_out[6][1] = 1; U_out[6][3] = -1
    V_out[6][2] = 1; V_out[6][3] = 1
    W_out[6][0] = 1

    # Fill U, V, W
    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            for i in range(4):
                for j in range(4):
                    br_A = i // 2
                    bc_A = j // 2
                    lr_A = i % 2
                    lc_A = j % 2
                    block_A = br_A * 2 + bc_A
                    
                    u_val = U_out[s][block_A] * U2[lr_A * 2 + lc_A][t]
                    U[i * 4 + j][r] = u_val
                    
            for j in range(4):
                for k in range(4):
                    br_B = j // 2
                    bc_B = k // 2
                    lr_B = j % 2
                    lc_B = k % 2
                    block_B = br_B * 2 + bc_B
                    
                    v_val = V_out[s][block_B] * V2[lr_B * 2 + lc_B][t]
                    V[j * 4 + k][r] = v_val
                    
            for k in range(4):
                for i in range(4):
                    br_C = k // 2
                    bc_C = i // 2
                    lr_C = k % 2
                    lc_C = i % 2
                    block_C = br_C * 2 + bc_C
                    
                    w_val = W_out[s][block_C] * W2[lr_C * 2 + lc_C][t]
                    W[k * 4 + i][r] = w_val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

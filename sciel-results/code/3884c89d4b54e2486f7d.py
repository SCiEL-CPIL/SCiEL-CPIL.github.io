"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # We construct a rank-98 decomposition for 4x4x8 matrix multiplication.
    # This is based on two independent rank-49 decompositions for 4x4x4,
    # using the block Strassen algorithm (rank 7 for 2x2x2 composed with itself).
    
    # Step 1: Define the 2x2x2 Strassen decomposition (Rank 7)
    # Indices i,j,k in {0,1}
    # U2 shape (4, 7), V2 shape (4, 7), W2 shape (4, 7)
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # Term 1: (A00+A11)(B00+B11) -> C00+=1, C11+=1
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # Term 2: (A10+A11)B00 -> C10+=1, C11-=1
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1; W2[3][1] = -1
    
    # Term 3: A00(B01-B11) -> C01+=1, C11+=1
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1; W2[3][2] = 1
    
    # Term 4: A11(B10-B00) -> C00+=1, C10+=1
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[1][3] = 1
    
    # Term 5: (A00+A01)B11 -> C00-=1, C01+=1
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[2][4] = 1
    
    # Term 6: (A10-A00)(B00+B01) -> C11+=1
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # Term 7: (A01-A11)(B10+B11) -> C00+=1
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1

    # Step 2: Compose to 4x4x4 (Rank 49)
    # Block indices I,J in {0,1} for A, J,K in {0,1} for B, I,K in {0,1} for C
    # U4 shape (16, 49), V4 shape (16, 49), W4 shape (16, 49)
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    # Coefficients for A blocks in the 7 Strassen terms
    # M1: A00+A11 -> coeff[0][0]=1, coeff[0][1]=1 (I=0,J=0 and I=1,J=1)
    # M2: A10+A11 -> coeff[1][0]=1, coeff[1][1]=1 (I=1,J=0 and I=1,J=1)
    # M3: A00      -> coeff[2][0]=1
    # M4: A11      -> coeff[3][1]=1
    # M5: A00+A01  -> coeff[4][0]=1, coeff[4][1]=1 (I=0,J=0 and I=0,J=1)
    # M6: A10-A00  -> coeff[5][0]=1, coeff[5][1]=-1 (I=1,J=0 and I=0,J=0)
    # M7: A01-A11  -> coeff[6][0]=1, coeff[6][1]=-1 (I=0,J=1 and I=1,J=1)
    
    A_coeffs = [
        {0: 1, 3: 1},  # M1: A00 (idx 0), A11 (idx 3)
        {2: 1, 3: 1},  # M2: A10 (idx 2), A11 (idx 3)
        {0: 1},        # M3: A00
        {3: 1},        # M4: A11
        {0: 1, 1: 1},  # M5: A00, A01
        {2: 1, 0: -1}, # M6: A10, -A00
        {1: 1, 3: -1}  # M7: A01, -A11
    ]
    
    # Coefficients for B blocks
    # M1: B00+B11 -> {0:1, 3:1}
    # M2: B00     -> {0:1}
    # M3: B01-B11 -> {1:1, 3:-1}
    # M4: B10-B00 -> {2:1, 0:-1}
    # M5: B11     -> {3:1}
    # M6: B00+B01 -> {0:1, 1:1}
    # M7: B10+B11 -> {2:1, 3:1}
    
    B_coeffs = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]
    
    # Coefficients for C blocks (result)
    # C00: M1+M4-M5+M7 -> {0:1, 3:1, 4:-1, 6:1}
    # C01: M3+M5       -> {2:1, 4:1}
    # C10: M2+M4       -> {1:1, 3:1}
    # C11: M1-M2+M3+M6 -> {0:1, 1:-1, 2:1, 5:1}
    
    C_coeffs = [
        {0: 1, 3: 1, 4: -1, 6: 1}, # C00
        {2: 1, 4: 1},               # C01
        {1: 1, 3: 1},               # C10
        {0: 1, 1: -1, 2: 1, 5: 1}  # C11
    ]
    
    for r in range(7):
        for s in range(7):
            R_idx = 7 * r + s
            # Update U4
            for a_block_idx, coeff in A_coeffs[r].items():
                I = a_block_idx // 2
                J = a_block_idx % 2
                for i_prime in range(2):
                    for j_prime in range(2):
                        global_i = I * 2 + i_prime
                        global_j = J * 2 + j_prime
                        global_a_idx = global_i * 4 + global_j
                        local_a_idx = i_prime * 2 + j_prime
                        val = coeff * U2[local_a_idx][s]
                        if val != 0:
                            U4[global_a_idx][R_idx] += val
                            
            # Update V4
            for b_block_idx, coeff in B_coeffs[r].items():
                J = b_block_idx // 2
                K = b_block_idx % 2
                for j_prime in range(2):
                    for k_prime in range(2):
                        global_j = J * 2 + j_prime
                        global_k = K * 2 + k_prime
                        global_b_idx = global_j * 4 + global_k
                        local_b_idx = j_prime * 2 + k_prime
                        val = coeff * V2[local_b_idx][s]
                        if val != 0:
                            V4[global_b_idx][R_idx] += val
                            
            # Update W4
            for c_block_idx, coeff in C_coeffs[r].items():
                I = c_block_idx // 2
                K = c_block_idx % 2
                for i_prime in range(2):
                    for k_prime in range(2):
                        global_i = I * 2 + i_prime
                        global_k = K * 2 + k_prime
                        global_c_idx = global_k * 4 + global_i # Trace convention k*n+i
                        local_c_idx = k_prime * 2 + i_prime
                        val = coeff * W2[local_c_idx][s]
                        if val != 0:
                            W4[global_c_idx][R_idx] += val

    # Step 3: Extend to 4x4x8 (Rank 98)
    # U_final shape (16, 98)
    # V_final shape (32, 98)
    # W_final shape (32, 98)
    
    U_final = [[0]*98 for _ in range(16)]
    V_final = [[0]*98 for _ in range(32)]
    W_final = [[0]*98 for _ in range(32)]
    
    # Left half (cols 0-3 of B, cols 0-3 of C)
    for i in range(16):
        for r in range(49):
            U_final[i][r] = U4[i][r]
            U_final[i][r + 49] = U4[i][r]
            
    for j in range(4):
        for k in range(4):
            b_idx = j * 4 + k
            # Left half V
            V_final[b_idx][r] = V4[b_idx][r] # Wait, V4 index is j*4+k
            # Right half V
            # Global B index for right half: j*8 + (4+k)
            V_final[j * 8 + 4 + k][r + 49] = V4[b_idx][r]
            
    for k in range(4):
        for i in range(4):
            c_idx = k * 4 + i
            # Left half W
            W_final[c_idx][r] = W4[c_idx][r]
            # Right half W
            # Global C index for right half: (4+k)*4 + i
            W_final[(4 + k) * 4 + i][r + 49] = W4[c_idx][r]

    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END

import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Part 1: 4x4x4 multiplication using Rank 49 (Recursive Strassen)
    # Part 2: 4x4x1 multiplication using Rank 16 (Schoolbook)
    
    # --- Build 2x2 Rank 7 Factors ---
    # U2: (2*2, 7), V2: (2*2, 7), W2: (2*2, 7)
    # Indices for 2x2: i in {0,1}, j in {0,1}, k in {0,1}
    # U index: i*2+j, V index: j*2+k, W index: k*2+i
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1 = (A11 + A22)(B11 + B22)
    # A11 is U[0], A22 is U[3]
    # B11 is V[0], B22 is V[3]
    # C11 is W[0], C22 is W[3]
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2 = (A21 + A22)B11
    # A21 is U[2], A22 is U[3]
    # B11 is V[0]
    # C21 is W[2]
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1
    
    # M3 = A11(B12 - B22)
    # A11 is U[0]
    # B12 is V[1], B22 is V[3]
    # C12 is W[1]
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1
    
    # M4 = A22(B21 - B11)
    # A22 is U[3]
    # B21 is V[2], B11 is V[0]
    # C11 is W[0], C21 is W[2]
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # M5 = (A11 + A12)B22
    # A11 is U[0], A12 is U[1]
    # B22 is V[3]
    # C11 is W[0], C12 is W[1]
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1 # Note: C11 = M1+M4-M5+M7, so coeff is -1
    
    # M6 = (A21 - A11)(B11 + B12)
    # A21 is U[2], A11 is U[0]
    # B11 is V[0], B12 is V[1]
    # C22 is W[3]
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # M7 = (A12 - A22)(B21 + B22)
    # A12 is U[1], A22 is U[3]
    # B21 is V[2], B22 is V[3]
    # C11 is W[0], C22 is W[3]
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1; W2[3, 6] = -1 # Note: C22 = M1-M2+M3+M6, wait.
    # C22 = M1 - M2 + M3 + M6. M7 is NOT in C22.
    # Let's re-check Strassen formulas.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # So M7 is only in C11.
    # My W2 assignment for M7: W2[0, 6] = 1 (C11), W2[3, 6] = 0 (C22).
    # Let's correct W2 for M7.
    W2[3, 6] = 0
    
    # --- Build 4x4 Rank 49 Factors ---
    # We treat the 4x4 matrix as 2x2 blocks of 2x2 matrices.
    # The "variables" for the 4x4 Strassen are the 2x2 blocks.
    # We need to map the 4x4 indices to the 2x2 block indices.
    # Block (I, J) where I, J in {0,1} contains entries (i,j) where i=2I+i', j=2J+j'.
    # U4 index: i*4+j. Block index: I*2+J. Inner index: i'*2+j'.
    # The factor for a block product is the Kronecker product of the "block coefficient" and the "inner factor".
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # Strassen formulas for 4x4 (using blocks)
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We iterate over the 7 outer Strassen products.
    # For each outer product r (0..6), we have 7 inner Strassen products s (0..6).
    # Total 49 terms.
    
    # Define the outer coefficients for U, V, W for each of the 7 outer terms.
    # Outer U coeff for block (I, J)
    outer_U = {
        0: {(0,0): 1, (1,1): 1}, # M1: A11+A22
        1: {(1,0): 1, (1,1): 1}, # M2: A21+A22
        2: {(0,0): 1},           # M3: A11
        3: {(1,1): 1},           # M4: A22
        4: {(0,0): 1, (0,1): 1}, # M5: A11+A12
        5: {(1,0): 1, (0,0): -1},# M6: A21-A11
        6: {(0,1): 1, (1,1): -1} # M7: A12-A22
    }
    
    outer_V = {
        0: {(0,0): 1, (1,1): 1}, # M1: B11+B22
        1: {(0,0): 1},           # M2: B11
        2: {(0,1): 1, (1,1): -1},# M3: B12-B22
        3: {(1,0): 1, (0,0): -1},# M4: B21-B11
        4: {(1,1): 1},           # M5: B22
        5: {(0,0): 1, (0,1): 1}, # M6: B11+B12
        6: {(1,0): 1, (1,1): 1}  # M7: B21+B22
    }
    
    outer_W = {
        0: {(0,0): 1, (1,1): 1}, # M1: C11+C22 (coeff 1,1)
        1: {(1,0): 1},           # M2: C21 (coeff 1)
        2: {(0,1): 1},           # M3: C12 (coeff 1)
        3: {(0,0): 1, (1,0): 1}, # M4: C11+C21 (coeff 1,1)
        4: {(0,0): -1, (0,1): 1},# M5: C11-C12? No. C11 coeff -1, C12 coeff 1.
        5: {(1,1): 1},           # M6: C22 (coeff 1)
        6: {(0,0): 1}            # M7: C11 (coeff 1)
    }
    
    # Fill U4, V4, W4
    for r in range(7):
        for s in range(7):
            c = r * 7 + s
            # For each block (I,J) in outer_U[r], multiply by U2[s]
            for (I, J), coeff_u in outer_U[r].items():
                # U2[s] has non-zero entries at specific (i', j')
                # We need to map (I, J, i', j') to global (i, j)
                # i = 2*I + i', j = 2*J + j'
                # U2 index is i'*2 + j'
                # U4 index is i*4 + j
                for u_idx in range(4):
                    val = U2[u_idx, s]
                    if val != 0:
                        i_prime = u_idx // 2
                        j_prime = u_idx % 2
                        i_global = 2 * I + i_prime
                        j_global = 2 * J + j_prime
                        u4_idx = i_global * 4 + j_global
                        U4[u4_idx, c] += coeff_u * val
                        
            # Similarly for V
            for (J, K), coeff_v in outer_V[r].items(): # Note: V blocks are (J,K) where J is row, K is col of B block
                # Wait, B is 4x4. Blocks are B_JK.
                # V2 index is j'*2 + k'.
                # Global j = 2*J + j', k = 2*K + k'.
                # V4 index is j*4 + k.
                for v_idx in range(4):
                    val = V2[v_idx, s]
                    if val != 0:
                        j_prime = v_idx // 2
                        k_prime = v_idx % 2
                        j_global = 2 * J + j_prime
                        k_global = 2 * K + k_prime
                        v4_idx = j_global * 4 + k_global
                        V4[v4_idx, c] += coeff_v * val
                        
            # Similarly for W
            for (K, I), coeff_w in outer_W[r].items(): # Note: W blocks are (K,I) where K is col of C, I is row of C
                # W2 index is k'*2 + i'.
                # Global k = 2*K + k', i = 2*I + i'.
                # W4 index is k*4 + i.
                for w_idx in range(4):
                    val = W2[w_idx, s]
                    if val != 0:
                        k_prime = w_idx // 2
                        i_prime = w_idx % 2
                        k_global = 2 * K + k_prime
                        i_global = 2 * I + i_prime
                        w4_idx = k_global * 4 + i_global
                        W4[w4_idx, c] += coeff_w * val

    # --- Build 4x4x1 Rank 16 Factors ---
    U1 = np.zeros((16, 16))
    V1 = np.zeros((4, 16)) # m*p = 4*1 = 4
    W1 = np.zeros((4, 16)) # p*n = 1*4 = 4
    
    col = 0
    for i in range(4):
        for j in range(4):
            U1[i*4+j, col] = 1
            V1[j*1+0, col] = 1
            W1[0*4+i, col] = 1
            col += 1

    # --- Combine ---
    # U: (16, 49+16)
    # V: (16, 49+16) -- Wait, V for 4x4x4 is (16, 49). V for 4x4x1 is (4, 16).
    # The dimensions of V must match m*p.
    # For 4x4x4, m=4, p=4. V is (16, 49).
    # For 4x4x1, m=4, p=1. V is (4, 16).
    # We need to embed the 4x4x1 V into the 4x4x5 V space.
    # The total p is 5. So V total is (4*5, 65) = (20, 65).
    # The 4x4x4 part uses the first 4 columns of B (k=0..3).
    # The 4x4x1 part uses the 5th column of B (k=4).
    
    U_total = np.zeros((16, 65))
    V_total = np.zeros((20, 65))
    W_total = np.zeros((20, 65)) # p*n = 5*4 = 20
    
    # Copy 4x4x4 part
    U_total[:, 0:49] = U4
    V_total[0:16, 0:49] = V4 # Rows 0..15 correspond to j*4+k where k=0..3
    # W4 is (16, 49). Indices k*4+i. k=0..3.
    # W_total rows 0..15 correspond to k*4+i where k=0..3.
    W_total[0:16, 0:49] = W4
    
    # Copy 4x4x1 part
    U_total[:, 49:65] = U1
    # V1 is (4, 16). Indices j*1+k where k=0.
    # In V_total, the 5th column of B is k=4.
    # So V_total index is j*5+4.
    for j in range(4):
        V_total[j*5+4, 49:65] = V1[j, :]
        
    # W1 is (4, 16). Indices k*4+i where k=0.
    # In W_total, the 5th column of C? No, k is the column index of B/C.
    # W index is k*n+i.
    # For the 4x4x1 part, the output is C[:, 4] (5th column).
    # So k=4.
    # W_total index is 4*4+i = 16+i.
    for i in range(4):
        W_total[16+i, 49:65] = W1[i, :]

    # Convert to lists of lists
    U_list = U_total.tolist()
    V_list = V_total.tolist()
    W_list = W_total.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

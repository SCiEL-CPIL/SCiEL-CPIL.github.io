import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    # Initialize U, V, W with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen 2x2 decomposition coefficients
    # U_2[i*2+j, r], V_2[j*2+k, r], W_2[k*2+i, r]
    # r is 0..6 corresponding to M1..M7
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1 = (A11 + A22)(B11 + B22)
    # A11: i=0,j=0 -> idx 0. A22: i=1,j=1 -> idx 3.
    U2[0][0] = 1
    U2[3][0] = 1
    # B11: j=0,k=0 -> idx 0. B22: j=1,k=1 -> idx 3.
    V2[0][0] = 1
    V2[3][0] = 1
    # C11: k=0,i=0 -> idx 0. C22: k=1,i=1 -> idx 3.
    W2[0][0] = 1
    W2[3][0] = 1
    
    # M2 = (A21 + A22)B11
    # A21: i=1,j=0 -> idx 2. A22: i=1,j=1 -> idx 3.
    U2[2][1] = 1
    U2[3][1] = 1
    # B11: j=0,k=0 -> idx 0.
    V2[0][1] = 1
    # C21: k=1,i=0 -> idx 2.
    W2[2][1] = 1
    
    # M3 = A11(B12 - B22)
    # A11: i=0,j=0 -> idx 0.
    U2[0][2] = 1
    # B12: j=0,k=1 -> idx 1. B22: j=1,k=1 -> idx 3.
    V2[1][2] = 1
    V2[3][2] = -1
    # C12: k=0,i=1 -> idx 1.
    W2[1][2] = 1
    
    # M4 = A22(B21 - B11)
    # A22: i=1,j=1 -> idx 3.
    U2[3][3] = 1
    # B21: j=1,k=0 -> idx 2. B11: j=0,k=0 -> idx 0.
    V2[2][3] = 1
    V2[0][3] = -1
    # C21: k=1,i=0 -> idx 2.
    W2[2][3] = 1
    
    # M5 = (A11 + A12)B22
    # A11: i=0,j=0 -> idx 0. A12: i=0,j=1 -> idx 1.
    U2[0][4] = 1
    U2[1][4] = 1
    # B22: j=1,k=1 -> idx 3.
    V2[3][4] = 1
    # C11: k=0,i=0 -> idx 0. Coeff -1.
    W2[0][4] = -1
    
    # M6 = (A21 - A11)(B11 + B12)
    # A21: i=1,j=0 -> idx 2. A11: i=0,j=0 -> idx 0.
    U2[2][5] = 1
    U2[0][5] = -1
    # B11: j=0,k=0 -> idx 0. B12: j=0,k=1 -> idx 1.
    V2[0][5] = 1
    V2[1][5] = 1
    # C22: k=1,i=1 -> idx 3. Coeff 1.
    W2[3][5] = 1
    
    # M7 = (A12 - A22)(B21 + B22)
    # A12: i=0,j=1 -> idx 1. A22: i=1,j=1 -> idx 3.
    U2[1][6] = 1
    U2[3][6] = -1
    # B21: j=1,k=0 -> idx 2. B22: j=1,k=1 -> idx 3.
    V2[2][6] = 1
    V2[3][6] = 1
    # C11: k=0,i=0 -> idx 0. Coeff 1.
    W2[0][6] = 1
    
    # Construct 4x4 decomposition using Kronecker product
    for r1 in range(7):
        for r2 in range(7):
            r4 = r1 * 7 + r2
            
            # Collect non-zero indices for U2, V2, W2 for r1 and r2
            u1_indices = [(idx, U2[idx][r1]) for idx in range(4) if U2[idx][r1] != 0]
            u2_indices = [(idx, U2[idx][r2]) for idx in range(4) if U2[idx][r2] != 0]
            
            v1_indices = [(idx, V2[idx][r1]) for idx in range(4) if V2[idx][r1] != 0]
            v2_indices = [(idx, V2[idx][r2]) for idx in range(4) if V2[idx][r2] != 0]
            
            w1_indices = [(idx, W2[idx][r1]) for idx in range(4) if W2[idx][r1] != 0]
            w2_indices = [(idx, W2[idx][r2]) for idx in range(4) if W2[idx][r2] != 0]
            
            # Fill U
            for idx1, val1 in u1_indices:
                i1 = idx1 // 2
                j1 = idx1 % 2
                for idx2, val2 in u2_indices:
                    i2 = idx2 // 2
                    j2 = idx2 % 2
                    idx4 = i1 * 8 + i2 * 4 + j1 * 2 + j2
                    U[idx4][r4] += val1 * val2
            
            # Fill V
            for idx1, val1 in v1_indices:
                j1 = idx1 // 2
                k1 = idx1 % 2
                for idx2, val2 in v2_indices:
                    j2 = idx2 // 2
                    k2 = idx2 % 2
                    idx4 = j1 * 8 + j2 * 4 + k1 * 2 + k2
                    V[idx4][r4] += val1 * val2
            
            # Fill W
            for idx1, val1 in w1_indices:
                k1 = idx1 // 2
                i1 = idx1 % 2
                for idx2, val2 in w2_indices:
                    k2 = idx2 // 2
                    i2 = idx2 % 2
                    idx4 = k1 * 8 + k2 * 4 + i1 * 2 + i2
                    W[idx4][r4] += val1 * val2
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

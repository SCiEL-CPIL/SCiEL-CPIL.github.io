import numpy as np

def construct():
    n, m, p = 4, 4, 8
    rank = 98
    
    # 1. Generate U2, V2, W2 for 2x2x2 Strassen
    # Indices: U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)
    
    # Strassen coefficients
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b01)
    # m7 = (a01-a11)(b10+b11)
    
    # A coeffs (alpha)
    # r, i, j -> coeff
    alpha = [
        {(0,0):1, (1,1):1}, # m1
        {(1,0):1, (1,1):1}, # m2
        {(0,0):1},          # m3
        {(1,1):1},          # m4
        {(0,0):1, (0,1):1}, # m5
        {(1,0):1, (0,0):-1},# m6
        {(0,1):1, (1,1):-1} # m7
    ]
    
    # B coeffs (beta)
    # r, j, k -> coeff
    beta = [
        {(0,0):1, (1,1):1}, # m1
        {(0,0):1},          # m2
        {(0,1):1, (1,1):-1},# m3
        {(1,0):1, (0,0):-1},# m4
        {(1,1):1},          # m5
        {(0,0):1, (0,1):1}, # m6
        {(1,0):1, (1,1):1}  # m7
    ]
    
    # C coeffs (gamma)
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    # r, k, i -> coeff (Note: W index is k*2+i, so k is col, i is row of C)
    # C_{i,k}
    gamma = {
        (0,0): {0:1, 3:1, 4:-1, 6:1}, # c00 (i=0,k=0)
        (0,1): {2:1, 4:1},             # c01 (i=0,k=1)
        (1,0): {1:1, 3:1},             # c10 (i=1,k=0)
        (1,1): {0:1, 1:-1, 2:1, 5:1}   # c11 (i=1,k=1)
    }
    
    for r in range(7):
        for (i,j), val in alpha[r].items():
            U2[i*2+j, r] = val
        for (j,k), val in beta[r].items():
            V2[j*2+k, r] = val
        for (i,k), coeffs in gamma.items():
            if r in coeffs:
                W2[k*2+i, r] = coeffs[r]

    # 2. Generate U4, V4, W4 for 4x4x4 using recursive Strassen
    # U4[i*4+j, R], V4[j*4+k, R], W4[k*4+i, R]
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    # Outer Strassen coefficients for 4x4 blocks
    # A_r = sum alpha_r[I,J] A[I,J]
    # B_r = sum beta_r[J,K] B[J,K]
    # C[P,Q] = sum gamma_r[P,Q] M_r
    
    # alpha_outer: r -> { (I,J): coeff }
    alpha_outer = [
        {(0,0):1, (1,1):1},
        {(1,0):1, (1,1):1},
        {(0,0):1},
        {(1,1):1},
        {(0,0):1, (0,1):1},
        {(1,0):1, (0,0):-1},
        {(0,1):1, (1,1):-1}
    ]
    
    # beta_outer: r -> { (J,K): coeff }
    beta_outer = [
        {(0,0):1, (1,1):1},
        {(0,0):1},
        {(0,1):1, (1,1):-1},
        {(1,0):1, (0,0):-1},
        {(1,1):1},
        {(0,0):1, (0,1):1},
        {(1,0):1, (1,1):1}
    ]
    
    # gamma_outer: (P,Q) -> { r: coeff }
    # C[P,Q] is 2x2 block
    gamma_outer = {
        (0,0): {0:1, 3:1, 4:-1, 6:1},
        (0,1): {2:1, 4:1},
        (1,0): {1:1, 3:1},
        (1,1): {0:1, 1:-1, 2:1, 5:1}
    }
    
    for r in range(7):
        for s in range(7):
            R = r*7 + s
            
            # Update U4
            # U4[i*4+j, R] = alpha_outer[r][I,J] * U2[i_loc*2+j_loc, s]
            for i in range(4):
                for j in range(4):
                    I = i // 2
                    J = j // 2
                    i_loc = i % 2
                    j_loc = j % 2
                    if (I,J) in alpha_outer[r]:
                        U4[i*4+j, R] = alpha_outer[r][(I,J)] * U2[i_loc*2+j_loc, s]
                        
            # Update V4
            # V4[j*4+k, R] = beta_outer[r][J,K] * V2[j_loc*2+k_loc, s]
            for j in range(4):
                for k in range(4):
                    J = j // 2
                    K = k // 2
                    j_loc = j % 2
                    k_loc = k % 2
                    if (J,K) in beta_outer[r]:
                        V4[j*4+k, R] = beta_outer[r][(J,K)] * V2[j_loc*2+k_loc, s]
                        
            # Update W4
            # W4[k*4+i, R] = gamma_outer[P,Q][r] * W2[k_loc*2+i_loc, s]
            for k in range(4):
                for i in range(4):
                    P = k // 2
                    Q = i // 2
                    k_loc = k % 2
                    i_loc = i % 2
                    if (P,Q) in gamma_outer and r in gamma_outer[(P,Q)]:
                        W4[k*4+i, R] = gamma_outer[(P,Q)][r] * W2[k_loc*2+i_loc, s]

    # 3. Construct final U, V, W for 4x4x8
    U = np.zeros((16, 98), dtype=float)
    V = np.zeros((32, 98), dtype=float)
    W = np.zeros((32, 98), dtype=float)
    
    # Part 1: k in 0..3
    for r in range(49):
        U[:, r] = U4[:, r]
        # V: j*8+k
        for j in range(4):
            for k in range(4):
                V[j*8+k, r] = V4[j*4+k, r]
        # W: k*4+i
        for k in range(4):
            for i in range(4):
                W[k*4+i, r] = W4[k*4+i, r]
                
    # Part 2: k in 4..7
    for r in range(49):
        R = 49 + r
        U[:, R] = U4[:, r]
        # V: j*8+k (k in 4..7)
        for j in range(4):
            for k in range(4, 8):
                k_prime = k - 4
                V[j*8+k, R] = V4[j*4+k_prime, r]
        # W: k*4+i (k in 4..7)
        for k in range(4, 8):
            for i in range(4):
                k_prime = k - 4
                W[k*4+i, R] = W4[k_prime*4+i, r]

    # Convert to lists as required by interface
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

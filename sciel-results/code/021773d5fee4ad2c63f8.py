import numpy as np
import time

def construct():
    n, m, p = 4, 4, 5
    deadline = time.time() + 145
    
    # Part 1: 4x4 x 4x4 multiplication (cols 0-3)
    # Rank 49 (Laderman/Strassen)
    
    # Define 2x2 Strassen factors
    # U2: 4x7, V2: 4x7, W2: 4x7
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # r=0: M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # r=1: M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = 1
    
    # r=2: M3 = A00(B10-B11)
    U2[0, 2] = 1
    V2[2, 2] = -1; V2[3, 2] = 1
    W2[1, 2] = 1; W2[3, 2] = -1
    
    # r=3: M4 = A11(B01-B00)
    U2[3, 3] = 1
    V2[1, 3] = 1; V2[0, 3] = -1
    W2[2, 3] = 1; W2[3, 3] = -1
    
    # r=4: M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[1, 4] = 1; W2[3, 4] = -1
    
    # r=5: M6 = (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[0, 5] = 1
    
    # r=6: M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = -1
    
    # Construct 4x4 x 4x4 factors
    # U4: 16x49, V4: 16x49, W4: 16x49
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # Strassen coefficients for 4x4
    # Xk = sum_{a,b} cx[k,a,b] A_ab
    # Yk = sum_{c,d} cy[k,c,d] B_cd
    # Zk contributes to C_ab with cz[k,a,b]
    
    cx = [
        # k=0 (M1): A00+A11+A22+A33? No, blocks.
        # M1 = (A11+A22)(B11+B22) in block notation?
        # Let's stick to the 2x2 block indices 0,1.
        # A = [[A00, A01], [A10, A11]]
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # cx[k, block_row, block_col]
        # k=0: A00, A11
        [[1, 0], [0, 1]], # k=0
        [[0, 0], [1, 1]], # k=1
        [[1, 0], [0, 0]], # k=2
        [[0, 0], [0, 1]], # k=3
        [[1, 1], [0, 0]], # k=4
        [[0, 0], [1, -1]], # k=5
        [[0, 1], [0, -1]]  # k=6
    ]
    
    cy = [
        [[1, 1], [0, 0]], # k=0
        [[1, 0], [0, 0]], # k=1
        [[0, 0], [1, -1]], # k=2
        [[0, -1], [0, 1]], # k=3
        [[0, 0], [0, 1]], # k=4
        [[1, 1], [0, 0]], # k=5
        [[0, 0], [1, 1]]  # k=6
    ]
    
    cz = [
        [[1, 0], [0, 1]], # k=0 -> C00, C11
        [[0, 0], [1, 1]], # k=1 -> C10, C11
        [[0, 1], [0, -1]], # k=2 -> C01, -C11
        [[0, 0], [1, -1]], # k=3 -> C10, -C11
        [[0, 1], [0, -1]], # k=4 -> C01, -C11
        [[1, 0], [0, 0]], # k=5 -> C00
        [[-1, 0], [0, 0]]  # k=6 -> -C00
    ]
    
    # Map block indices to full indices
    # Block (br, bc) contains rows 2*br, 2*br+1 and cols 2*bc, 2*bc+1
    
    for k in range(7):
        for r2 in range(7):
            r = k * 7 + r2
            
            # U part
            # u_full[i, j, r] = sum_{a,b} cx[k,a,b] * u2[i', j', r2]
            # where i = 2*a + i', j = 2*b + j'
            for a in range(2):
                for b in range(2):
                    coeff = cx[k][a][b]
                    if coeff != 0:
                        for i_prime in range(2):
                            for j_prime in range(2):
                                u_val = U2[i_prime * 2 + j_prime, r2]
                                if u_val != 0:
                                    i = 2 * a + i_prime
                                    j = 2 * b + j_prime
                                    U4[i * 4 + j, r] = coeff * u_val
                                    
            # V part
            # v_full[j, k, r] = sum_{c,d} cy[k,c,d] * v2[j', k', r2]
            # where j = 2*c + j', k = 2*d + k'
            for c in range(2):
                for d in range(2):
                    coeff = cy[k][c][d]
                    if coeff != 0:
                        for j_prime in range(2):
                            for k_prime in range(2):
                                v_val = V2[j_prime * 2 + k_prime, r2]
                                if v_val != 0:
                                    j = 2 * c + j_prime
                                    k_col = 2 * d + k_prime
                                    V4[j * 5 + k_col, r] = coeff * v_val
                                    
            # W part
            # w_full[k, i, r] = sum_{a,b} cz[k,a,b] * w2[k', i', r2]
            # where k = 2*a + k', i = 2*b + i'
            # Note: W index is k*n + i. Here n=4 for 4x4 part.
            for a in range(2):
                for b in range(2):
                    coeff = cz[k][a][b]
                    if coeff != 0:
                        for k_prime in range(2):
                            for i_prime in range(2):
                                w_val = W2[k_prime * 2 + i_prime, r2]
                                if w_val != 0:
                                    k_col = 2 * a + k_prime
                                    i = 2 * b + i_prime
                                    W4[k_col * 4 + i, r] = coeff * w_val

    # Part 2: 4x4 x 1x1 (col 4)
    # Rank 4
    U2p = np.zeros((16, 4))
    V2p = np.zeros((20, 4))
    W2p = np.zeros((20, 4))
    
    for r in range(4):
        # u[i, j] = delta(j, r)
        for i in range(4):
            U2p[i * 4 + r, 49 + r] = 1
            
        # v[j, k] = delta(j, r) * delta(k, 4)
        V2p[r * 5 + 4, 49 + r] = 1
        
        # w[k, i] = delta(k, 4)
        for i in range(4):
            W2p[4 * 4 + i, 49 + r] = 1

    # Combine
    U = np.hstack([U4, U2p])
    V = np.hstack([V4, V2p])
    W = np.hstack([W4, W2p])
    
    # Verify
    # Target tensor
    target = np.zeros((16, 20, 20))
    for i in range(4):
        for j in range(4):
            for k in range(5):
                target[i * 4 + j, j * 5 + k, k * 4 + i] = 1
                
    est = np.einsum('ar,br,cr->abc', U, V, W)
    if not np.allclose(est, target, atol=1e-9):
        # Fallback to schoolbook if construction failed
        U_sb = np.zeros((16, 80))
        V_sb = np.zeros((20, 80))
        W_sb = np.zeros((20, 80))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    U_sb[i * 4 + j, col] = 1
                    V_sb[j * 5 + k, col] = 1
                    W_sb[k * 4 + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

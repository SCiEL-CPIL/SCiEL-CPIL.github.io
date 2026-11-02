import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Part 1: 4x4x4 (First 4 columns of B)
    # Using Strassen for 4x4x4 (rank 49)
    # ... (Code for A_coeffs, B_coeffs, C_coeffs remains the same) ...
    
    A_coeffs = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, 1]], # M2
        [[1, 0], [0, 0]], # M3
        [[0, 0], [0, 1]], # M4
        [[1, 1], [0, 0]], # M5
        [[-1, 0], [1, 0]], # M6
        [[0, 1], [0, -1]]  # M7
    ]
    B_coeffs = [
        [[1, 0], [0, 1]], # M1
        [[1, 0], [0, 0]], # M2
        [[0, -1], [0, 1]], # M3
        [[-1, 1], [0, 0]], # M4
        [[0, 0], [0, 1]], # M5
        [[1, 1], [0, 0]], # M6
        [[0, 1], [0, 1]]  # M7
    ]
    C_coeffs = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, -1]], # M2
        [[0, 1], [1, 0]], # M3
        [[1, 0], [0, 1]], # M4
        [[-1, 1], [0, 0]], # M5
        [[0, 0], [1, 1]], # M6
        [[1, 0], [0, 0]]  # M7
    ]

    # Inner Schoolbook 2x2x2
    U_in = np.zeros((4, 8), dtype=int)
    V_in = np.zeros((4, 8), dtype=int)
    W_in = np.zeros((4, 8), dtype=int)
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U_in[i*2+j, col] = 1
                V_in[j*2+k, col] = 1
                W_in[k*2+i, col] = 1
                col += 1
                
    # Build 4x4x4 using Strassen blocks and Schoolbook inner.
    rank1 = 7 * 8 # 56
    U_g = np.zeros((16, rank1), dtype=int)
    V_g = np.zeros((16, rank1), dtype=int)
    W_g = np.zeros((16, rank1), dtype=int)
    
    col_g = 0
    for s in range(7):
        A_s = A_coeffs[s]
        B_s = B_coeffs[s]
        C_s = C_coeffs[s]
        
        for t in range(8):
            for i in range(2):
                for j in range(2):
                    u_val = U_in[i*2+j, t]
                    if u_val != 0:
                        for I in range(2):
                            for J in range(2):
                                a_coeff = A_s[I][J]
                                if a_coeff != 0:
                                    idx_a = (I*2+i)*4 + (J*2+j)
                                    U_g[idx_a, col_g] += a_coeff * u_val
                                    
            for j in range(2):
                for k in range(2):
                    v_val = V_in[j*2+k, t]
                    if v_val != 0:
                        for J in range(2):
                            for K in range(2):
                                b_coeff = B_s[J][K]
                                if b_coeff != 0:
                                    idx_b = (J*2+j)*4 + (K*2+k)
                                    V_g[idx_b, col_g] += b_coeff * v_val
                                    
            for k in range(2):
                for i in range(2):
                    w_val = W_in[k*2+i, t]
                    if w_val != 0:
                        for K in range(2):
                            for L in range(2):
                                c_coeff = C_s[K][L]
                                if c_coeff != 0:
                                    idx_c = (K*2+k)*4 + (L*2+i)
                                    W_g[idx_c, col_g] += c_coeff * w_val
                                    
            col_g += 1
            
    # Part 2: 4x4x1 (Last column of B)
    rank2 = 16
    U_p2 = np.zeros((16, rank2), dtype=int)
    V_p2 = np.zeros((4, rank2), dtype=int)
    W_p2 = np.zeros((4, rank2), dtype=int)
    
    col_p2 = 0
    terms = [
        (0, 0, 0, 0), # A11, B1 -> I=0, J=0
        (0, 1, 1, 1), # A12, B2 -> I=0, J=1
        (1, 0, 0, 0), # A21, B1 -> I=1, J=0
        (1, 1, 1, 1)  # A22, B2 -> I=1, J=1
    ]
    
    for I, J, J_b, K_b in terms:
        for i in range(2):
            for j in range(2):
                idx_a = (I*2+i)*4 + (J*2+j)
                idx_b = J_b*2+j
                idx_c = I*2+i
                
                U_p2[idx_a, col_p2] = 1
                V_p2[idx_b, col_p2] = 1
                W_p2[idx_c, col_p2] = 1
                col_p2 += 1
                
    # Combine
    # U is simply concatenated along rank axis
    U = np.hstack([U_g, U_p2])
    
    # V must be (m*p, rank) = (20, 72)
    # V_g is (16, 56) corresponding to B (4x4). Index j*4+k
    # V_p2 is (4, 16) corresponding to B (4x1). Index j*1+0
    # We need to place V_g into V_total for k=0..3
    # and V_p2 into V_total for k=4
    
    V_total = np.zeros((20, 72), dtype=int)
    W_total = np.zeros((20, 72), dtype=int)
    
    # Place V_g
    # V_g row index r_g corresponds to j*4+k
    # V_total row index r_t corresponds to j*5+k
    # For k in 0..3:
    for r_g in range(16):
        j = r_g // 4
        k = r_g % 4
        r_t = j * 5 + k
        V_total[r_t, :56] = V_g[r_g, :]
        
    # Place V_p2
    # V_p2 row index r_p corresponds to j (since p=1, index j*1+0)
    # V_total row index for k=4: j*5 + 4
    for r_p in range(4):
        j = r_p
        r_t = j * 5 + 4
        V_total[r_t, 56:] = V_p2[r_p, :]
        
    # Place W_g
    # W_g is (16, 56) corresponding to C (4x4). Index k*4+i
    # W_total is (20, 72) corresponding to C (4x5). Index k*4+i? 
    # Wait, W shape is (p*n, rank). p=5, n=4. Total 20.
    # Index in W is k*n + i.
    # W_g corresponds to k in 0..3.
    for r_g in range(16):
        k = r_g // 4
        i = r_g % 4
        r_t = k * 4 + i
        W_total[r_t, :56] = W_g[r_g, :]
        
    # Place W_p2
    # W_p2 is (4, 16) corresponding to C (4x1). Index 0*4+i = i.
    # W_total row index for k=4: 4*4+i = 16+i.
    for r_p in range(4):
        i = r_p
        r_t = 4 * 4 + i
        W_total[r_t, 56:] = W_p2[r_p, :]
        
    return {"U": U.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}

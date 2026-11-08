import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 15
    pn = p * n  # 10

    # Target rank 28
    target_rank = 28
    
    U = np.zeros((nm, target_rank))
    V = np.zeros((mp, target_rank))
    W = np.zeros((pn, target_rank))
    
    r = 0
    
    def add_strassen_block(U_idx_map, V_idx_map, W_idx_map, start_r):
        """
        Adds a Strassen 2x2x2 decomposition to the global matrices.
        U_idx_map: dict mapping local (i,j) to global U index
        V_idx_map: dict mapping local (j,k) to global V index
        W_idx_map: dict mapping local (k,i) to global W index
        """
        # Strassen terms
        # M1 = (X00+X11)(Y00+Y11)
        # M2 = (X10+X11)Y00
        # M3 = X00(Y10-Y11)
        # M4 = X11(Y01-Y00)
        # M5 = (X00+X01)Y11
        # M6 = (X10-X00)(Y00+Y01)
        # M7 = (X01-X11)(Y10+Y11)
        
        # Local indices:
        # X (A_left): i in 0..1, j in 0..1
        # Y (B_block): j in 0..1, k in 0..1
        
        # We need to populate U, V, W for 7 terms.
        
        # Term 1: M1
        # U: X00, X11
        # V: Y00, Y11
        # W: C00, C01, C10, C11 (with coeffs)
        
        # It's easier to iterate through the 7 multiplications and add to U, V, W.
        # But Strassen combines them.
        # We will assign 1 column of rank per Strassen term.
        
        # Term 1: (X00 + X11) * (Y00 + Y11)
        # U[:, r] += X00 + X11
        # V[:, r] += Y00 + Y11
        # W[:, r] += C00 + C01 - C10 + C11 ? No, W is the output.
        # The decomposition is sum_r U_i * V_j * W_k = T_ijk
        # Here T_ijk is the identity tensor for the block.
        # So we need to express the block multiplication tensor as sum of 7 rank-1 tensors.
        
        # Rank 1 tensor for M1:
        # u1 = e_00 + e_11 (in A space)
        # v1 = e_00 + e_11 (in B space)
        # w1 = e_00 + e_01 - e_10 + e_11 (in C space, coeff for M1 is 1 for C00, 1 for C01? No.)
        
        # Let's use the standard Strassen output reconstruction:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So the contribution to W for each M term is:
        # M1: C00 + C11
        # M2: C10 - C11
        # M3: C01 + C11
        # M4: C00 + C10
        # M5: -C00 + C01
        # M6: C11
        # M7: C00
        
        # Wait, the signs in W must match the reconstruction.
        # If C = sum M_t * W_t, then:
        # W1 = e_00 + e_11
        # W2 = e_10 - e_11
        # W3 = e_01 + e_11
        # W4 = e_00 + e_10
        # W5 = -e_00 + e_01
        # W6 = e_11
        # W7 = e_00
        
        # Let's verify C00: W1[0,0] + W4[0,0] - W5[0,0] + W7[0,0] = 1 + 1 - (-1) + 1 = 4? No.
        # Standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So W vectors (coeffs for M_t in each C_ij):
        # W1: [1, 0, 0, 1] (C00, C01, C10, C11)
        # W2: [0, 0, 1, -1]
        # W3: [0, 1, 0, 1]
        # W4: [1, 0, 1, 0]
        # W5: [-1, 1, 0, 0]
        # W6: [0, 0, 0, 1]
        # W7: [1, 0, 0, 0]
        
        # U vectors (coeffs for A_ij in M_t):
        # U1: [1, 0, 0, 1] (A00, A01, A10, A11)
        # U2: [0, 0, 1, 1]
        # U3: [1, 0, 0, 0]
        # U4: [0, 0, 0, 1]
        # U5: [1, 1, 0, 0]
        # U6: [-1, 0, 1, 0]
        # U7: [0, 1, 0, -1]
        
        # V vectors (coeffs for B_ij in M_t):
        # V1: [1, 0, 0, 1] (B00, B01, B10, B11)
        # V2: [1, 0, 0, 0]
        # V3: [0, -1, 1, 0]
        # V4: [-1, 1, 0, 0]
        # V5: [0, 0, 0, 1]
        # V6: [1, 1, 0, 0]
        # V7: [0, 1, 1, 0]
        
        # We need to map local (i,j) to global U index.
        # local A: i in 0..1, j in 0..1.
        # local B: j in 0..1, k in 0..1.
        # local C: k in 0..1, i in 0..1.
        
        u_vecs = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ]
        v_vecs = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, -1, 1, 0],
            [-1, 1, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 1, 1, 0]
        ]
        w_vecs = [
            [1, 0, 0, 1],
            [0, 0, 1, -1],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [-1, 1, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0]
        ]
        
        for t in range(7):
            # Populate U
            for li in range(2):
                for lj in range(2):
                    coeff = u_vecs[t][li*2 + lj]
                    if coeff != 0:
                        global_u_idx = U_idx_map[(li, lj)]
                        U[global_u_idx, start_r + t] += coeff
                        
            # Populate V
            for lj in range(2):
                for lk in range(2):
                    coeff = v_vecs[t][lj*2 + lk]
                    if coeff != 0:
                        global_v_idx = V_idx_map[(lj, lk)]
                        V[global_v_idx, start_r + t] += coeff
                        
            # Populate W
            for lk in range(2):
                for li in range(2):
                    coeff = w_vecs[t][lk*2 + li]
                    if coeff != 0:
                        global_w_idx = W_idx_map[(lk, li)]
                        W[global_w_idx, start_r + t] += coeff
                        
        return start_r + 7

    # Block 1: A_left (2x2) * B_top_L (2x2)
    # A_left: i in 0..1, j in 0..1. Global U: i*3 + j
    # B_top_L: j in 0..1, k in 0..1. Global V: j*5 + k
    # C_part: k in 0..1, i in 0..1. Global W: k*2 + i
    
    U_map_1 = {}
    for i in range(2):
        for j in range(2):
            U_map_1[(i, j)] = i * 3 + j
            
    V_map_1 = {}
    for j in range(2):
        for k in range(2):
            V_map_1[(j, k)] = j * 5 + k
            
    W_map_1 = {}
    for k in range(2):
        for i in range(2):
            W_map_1[(k, i)] = k * 2 + i
            
    r = add_strassen_block(U, V, W, r, U_map_1, V_map_1, W_map_1)
    
    # Block 2: A_left (2x2) * B_top_R1 (2x2)
    # A_left: same
    # B_top_R1: j in 0..1, k in 2..3. Global V: j*5 + k
    # C_part: k in 2..3, i in 0..1. Global W: k*2 + i
    
    U_map_2 = U_map_1
    
    V_map_2 = {}
    for j in range(2):
        for k in range(2, 4):
            V_map_2[(j, k-2)] = j * 5 + k # Local k 0..1 maps to global 2..3
            
    W_map_2 = {}
    for k in range(2, 4):
        for i in range(2):
            W_map_2[(k-2, i)] = k * 2 + i # Local k 0..1 maps to global 2..3
            
    r = add_strassen_block(U, V, W, r, U_map_2, V_map_2, W_map_2)
    
    # Block 3: A_left (2x2) * B_top_R2 (2x1)
    # A_left: same
    # B_top_R2: j in 0..1, k in 4. Global V: j*5 + 4
    # C_part: k in 4, i in 0..1. Global W: 4*2 + i
    
    # Naive rank 4
    for j in range(2):
        for i in range(2):
            U[i * 3 + j, r] = 1
            V[j * 5 + 4, r] = 1
            W[4 * 2 + i, r] = 1
            r += 1
            
    # Block 4: A_right (2x1) * B_bottom (1x5)
    # A_right: i in 0..1, j in 2. Global U: i*3 + 2
    # B_bottom: j in 2, k in 0..4. Global V: 2*5 + k
    # C_part: k in 0..4, i in 0..1. Global W: k*2 + i
    
    # Naive rank 10
    for k in range(5):
        for i in range(2):
            U[i * 3 + 2, r] = 1
            V[2 * 5 + k, r] = 1
            W[k * 2 + i, r] = 1
            r += 1
            
    # Verify rank
    if r != target_rank:
        # Fallback to naive if construction failed
        rank_naive = n * m * p
        U_naive = np.zeros((nm, rank_naive))
        V_naive = np.zeros((mp, rank_naive))
        W_naive = np.zeros((pn, rank_naive))
        r_naive = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_naive[i*m + j, r_naive] = 1
                    V_naive[j*p + k, r_naive] = 1
                    W_naive[k*n + i, r_naive] = 1
                    r_naive += 1
        return {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

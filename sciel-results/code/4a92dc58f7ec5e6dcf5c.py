import numpy as np
import time

DIMENSIONS = [4, 5, 5]
TIME_LIMIT = 140.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Schoolbook rank fallback
    r_school = n * m * p
    U_sb = np.zeros((n * m, r_school))
    V_sb = np.zeros((m * p, r_school))
    W_sb = np.zeros((p * n, r_school))
    
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    for c, (i, j, k) in enumerate(indices):
        U_sb[i * m + j, c] = 1.0
        V_sb[j * p + k, c] = 1.0
        W_sb[k * n + i, c] = 1.0

    def _pack(U, V, W, r):
        return U[:, :r].tolist(), V[:, :r].r_[:, :r].tolist(), W[:, :r].tolist()

    if time.time() - start_time > TIME_LIMIT:
        return {"U": _pack(U_sb, V_sb, W_sb, r_school)[0], 
                "V": _pack(U_sb, V_sb, W_sb, r_school)[1], 
                "W": _pack(U_sb, V_sb, W_sb, r_school)[2]}

    try:
        # --- Part 1: Strassen 4x4x4 (Rank 49) ---
        # Multiplying A[0:4, 0:4] by B[0:4, 0:4] to get C[0:4, 0:4]
        
        # Get Strassen 2x2x2 factors (Rank 7)
        def get_strassen_2x2():
            U = np.zeros((4, 7))
            V = np.zeros((4, 7))
            W = np.zeros((4, 7))
            
            # P1 = (A00+A11)(B00+B11)
            U[0, 0] = 1; U[3, 0] = 1
            V[0, 0] = 1; V[3, 0] = 1
            W[0, 0] = 1; W[3, 0] = 1
            
            # P2 = (A10+A11)B00
            U[2, 1] = 1; U[3, 1] = 1
            V[0, 1] = 1
            W[2, 1] = 1
            W[3, 1] = -1
            
            # P3 = A00(B01-B11)
            U[0, 2] = 1
            V[1, 2] = 1; V[3, 2] = -1
            W[1, 2] = 1
            W[3, 2] = 1
            
            # P4 = A11(B10-B00)
            U[3, 3] = 1
            V[2, 3] = 1; V[0, 3] = -1
            W[0, 3] = 1
            W[2, 3] = 1
            
            # P5 = (A00+A01)B11
            U[0, 4] = 1; U[1, 4] = 1
            V[3, 4] = 1
            W[0, 4] = -1
            W[1, 4] = 1
            
            # P6 = (A10-A00)(B00+B01)
            U[2, 5] = 1; U[0, 5] = -1
            V[0, 5] = 1; V[1, 5] = 1
            W[3, 5] = 1
            
            # P7 = (A01-A11)(B10+B11)
            U[1, 6] = 1; U[3, 6] = -1
            V[2, 6] = 1; V[3, 6] = 1
            W[0, 6] = 1
            
            return U, V, W

        U2, V2, W2 = get_strassen_2x2()
        
        # Block linear forms for Strassen 4x4
        # A blocks: 00, 01, 10, 11
        # B blocks: 00, 01, 10, 11
        # We define the coefficients for each block in the linear form for A and B for each of the 7 products.
        
        # P1: (A00+A11)(B00+B11)
        A_blocks_1 = [(0,0,1), (1,1,1)]
        B_blocks_1 = [(0,0,1), (1,1,1)]
        
        # P2: (A10+A11)B00
        A_blocks_2 = [(1,0,1), (1,1,1)]
        B_blocks_2 = [(0,0,1)]
        
        # P3: A00(B01-B11)
        A_blocks_3 = [(0,0,1)]
        B_blocks_3 = [(0,1,1), (1,1,-1)]
        
        # P4: A11(B10-B00)
        A_blocks_4 = [(1,1,1)]
        B_blocks_4 = [(1,0,1), (0,0,-1)]
        
        # P5: (A00+A01)B11
        A_blocks_5 = [(0,0,1), (0,1,1)]
        B_blocks_5 = [(1,1,1)]
        
        # P6: (A10-A00)(B00+B01)
        A_blocks_6 = [(1,0,1), (0,0,-1)]
        B_blocks_6 = [(0,0,1), (0,1,1)]
        
        # P7: (A01-A11)(B10+B11)
        A_blocks_7 = [(0,1,1), (1,1,-1)]
        B_blocks_7 = [(1,0,1), (1,1,1)]
        
        block_forms = [
            (A_blocks_1, B_blocks_1),
            (A_blocks_2, B_blocks_2),
            (A_blocks_3, B_blocks_3),
            (A_blocks_4, B_blocks_4),
            (A_blocks_5, B_blocks_5),
            (A_blocks_6, B_blocks_6),
            (A_blocks_7, B_blocks_7)
        ]
        
        # Result block coefficients for C
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # We need to map these to the W matrix.
        # W index for C block (bi, bj) local (li, lj) is (2*bj + lj)*4 + (2*bi + li)
        # Wait, W index is k*n + i.
        # C is 4x4. i is row (0..3), k is col (0..3).
        # Block (bi, bj) covers rows 2*bi..2*bi+1, cols 2*bj..2*bj+1.
        # Local (li, lj) -> i = 2*bi + li, k = 2*bj + lj.
        # W index = k*4 + i = (2*bj + lj)*4 + (2*bi + li).
        
        # For each Strassen product s (0..6), and each local rank r (0..6):
        # We add contributions to U, V, W.
        
        # We will accumulate U, V, W for the 4x4x4 part.
        # Size: U (16, 49), V (16, 49), W (16, 49)
        U_strassen = np.zeros((16, 49))
        V_strassen = np.zeros((16, 49))
        W_strassen = np.zeros((16, 49))
        
        for s in range(7):
            A_blocks, B_blocks = block_forms[s]
            
            # Precompute the global U and V vectors for the block linear forms
            # U_global_block is a 16x1 vector? No, for each local rank r, we scale by U2[local, r].
            # Let's iterate over local rank r.
            
            for r in range(7):
                col = s * 7 + r
                
                # Construct global U vector
                # U2[local_idx, r] gives the coefficient for the local entry.
                # For each block (bi, bj) in A_blocks with coeff c:
                # For each local entry (li, lj) in 2x2 block:
                # Global A index: i = 2*bi + li, j = 2*bj + lj.
                # U index: i*5 + j? No, U is for A (4x5).
                # But this part is only for A[0:4, 0:4].
                # So A index is (i, j) with i,j in 0..3.
                # U index in the 4x5 matrix is i*5 + j.
                
                # However, we are building a 16-dim vector for the 4x4 subpart.
                # We will later place this into the full 20-dim U.
                # Let's build the 16-dim vector for now.
                
                u_vec = np.zeros(16)
                v_vec = np.zeros(16)
                
                for (bi, bj, c) in A_blocks:
                    for li in range(2):
                        for lj in range(2):
                            local_idx = li * 2 + lj
                            coeff = c * U2[local_idx, r]
                            if coeff != 0:
                                global_i = 2 * bi + li
                                global_j = 2 * bj + lj
                                u_vec[global_i * 4 + global_j] = coeff
                
                for (bi, bj, c) in B_blocks:
                    for li in range(2):
                        for lj in range(2):
                            local_idx = li * 2 + lj
                            coeff = c * V2[local_idx, r]
                            if coeff != 0:
                                global_i = 2 * bi + li
                                global_j = 2 * bj + lj
                                v_vec[global_i * 4 + global_j] = coeff
                
                U_strassen[:, col] = u_vec
                V_strassen[:, col] = v_vec
                
                # Construct W vector
                # We need to know which C blocks this product contributes to, and with what sign.
                # This depends on s.
                c_coeffs = {}
                if s == 0: # P1
                    c_coeffs = {(0,0): 1, (1,1): 1}
                elif s == 1: # P2
                    c_coeffs = {(1,0): 1, (1,1): -1}
                elif s == 2: # P3
                    c_coeffs = {(0,1): 1, (1,1): 1}
                elif s == 3: # P4
                    c_coeffs = {(0,0): 1, (1,0): 1}
                elif s == 4: # P5
                    c_coeffs = {(0,0): -1, (0,1): 1}
                elif s == 5: # P6
                    c_coeffs = {(1,1): 1}
                elif s == 6: # P7
                    c_coeffs = {(0,0): 1}
                
                w_vec = np.zeros(16)
                for (bi, bj, c) in c_coeffs.items():
                    for li in range(2):
                        for lj in range(2):
                            local_idx = li * 2 + lj
                            coeff = c * W2[local_idx, r]
                            if coeff != 0:
                                global_i = 2 * bi + li
                                global_k = 2 * bj + lj
                                # W index for C (4x4) is k*4 + i
                                w_vec[global_k * 4 + global_i] = coeff
                
                W_strassen[:, col] = w_vec
        
        # --- Part 2: Schoolbook 4x4x1 (Rank 16) ---
        # A[0:4, 0:4] * B[0:4, 4] -> C[0:4, 4]
        # C[i, 4] = sum_j A[i, j] B[j, 4]
        # Rank 16: for each i, j in 0..3
        
        U_sb441 = np.zeros((16, 16)) # 16 is 4*4, but we need to map to full U
        V_sb441 = np.zeros((16, 16))
        W_sb441 = np.zeros((16, 16))
        
        for j in range(4):
            for i in range(4):
                col = j * 4 + i
                # U: A[i, j]
                U_sb441[i * 4 + j, col] = 1.0
                # V: B[j, 4]
                # In 4x4 subpart of B (which is B[0:4, 0:4]), this index is out of bounds.
                # We are handling the full B (5x5) in the final assembly.
                # Let's just store the local indices and map later.
                # V local index for B[j, 4] is not in 0..15.
                # Let's build the full vectors directly.
                
        # Let's build full vectors for Part 2
        U_p2 = np.zeros((20, 16))
        V_p2 = np.zeros((25, 16))
        W_p2 = np.zeros((20, 16))
        
        for j in range(4):
            for i in range(4):
                col = j * 4 + i
                # U: A[i, j]
                U_p2[i * 5 + j, col] = 1.0
                # V: B[j, 4]
                V_p2[j * 5 + 4, col] = 1.0
                # W: C[i, 4]
                W_p2[4 * 4 + i, col] = 1.0
        
        # --- Part 3: Outer Product 4x1x5 (Rank 20) ---
        # A[0:4, 4] * B[4, 0:5] -> C[0:4, 0:5]
        # C[i, k] += A[i, 4] B[4, k]
        # Rank 20: for each i, k in 0..3, 0..4
        
        U_p3 = np.zeros((20, 20))
        V_p3 = np.zeros((25, 20))
        W_p3 = np.zeros((20, 20))
        
        for i in range(4):
            for k in range(5):
                col = i * 5 + k
                # U: A[i, 4]
                U_p3[i * 5 + 4, col] = 1.0
                # V: B[4, k]
                V_p3[4 * 5 + k, col] = 1.0
                # W: C[i, k]
                W_p3[k * 4 + i, col] = 1.0
        
        # --- Assemble Full Matrices ---
        # Total Rank = 49 + 16 + 20 = 85
        
        U_full = np.zeros((20, 85))
        V_full = np.zeros((25, 85))
        W_full = np.zeros((20, 85))
        
        # Copy Part 1 (Strassen 4x4x4)
        # U_strassen is 16x49. Map to U_full[0:16, 0:49]
        # But U_full is 20x85. The first 16 rows correspond to A[0:4, 0:4].
        # Wait, A is 4x5.
        # A[0:4, 0:4] indices are i*5+j for i in 0..3, j in 0..3.
        # These are NOT contiguous 0..15.
        # i=0: 0,1,2,3
        # i=1: 5,6,7,8
        # i=2: 10,11,12,13
        # i=3: 15,16,17,18
        # So we need to map the 16-dim vector to the 20-dim vector.
        
        def map_4x4_to_4x5(vec_16):
            vec_20 = np.zeros(20)
            for i in range(4):
                for j in range(4):
                    vec_20[i * 5 + j] = vec_16[i * 4 + j]
            return vec_20
        
        for c in range(49):
            U_full[:, c] = map_4x4_to_4x5(U_strassen[:, c])
            # V_strassen is for B[0:4, 0:4].
            # B is 5x5.
            # B[0:4, 0:4] indices are j*5+k for j in 0..3, k in 0..3.
            # Same mapping.
            V_full[:, c] = map_4x4_to_4x5(V_strassen[:, c])
            
            # W_strassen is for C[0:4, 0:4].
            # C is 4x5.
            # C[0:4, 0:4] indices are k*4+i for k in 0..3, i in 0..3.
            # These are contiguous 0..15.
            W_full[0:16, c] = W_strassen[:, c]
            W_full[16:20, c] = 0
        
        # Copy Part 2 (Schoolbook 4x4x1)
        for c in range(16):
            U_full[:, 49 + c] = U_p2[:, c]
            V_full[:, 49 + c] = V_p2[:, c]
            W_full[:, 49 + c] = W_p2[:, c]
            
        # Copy Part 3 (Outer Product)
        for c in range(20):
            U_full[:, 49 + 16 + c] = U_p3[:, c]
            V_full[:, 49 + 16 + c] = V_p3[:, c]
            W_full[:, 49 + 16 + c] = W_p3[:, c]
            
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
        
    except Exception as e:
        return {"U": _pack(U_sb, V_sb, W_sb, r_school)[0], 
                "V": _pack(U_sb, V_sb, W_sb, r_school)[1], 
                "W": _pack(U_sb, V_sb, W_sb, r_school)[2]}


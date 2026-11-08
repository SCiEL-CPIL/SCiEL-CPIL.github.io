import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = 4, 4, 8
    
    # Strassen 2x2x2 decomposition (rank 7)
    # U_2[i*2+j, s], V_2[i*2+j, s], W_2[i*2+j, s]
    # where i,j in {0,1} and s in {0..6}
    
    # Standard Strassen for 2x2x2:
    # S0 = (A00+A11)(B00+B11)
    # S1 = (A10+A11)B00
    # S2 = A00(B01-B11)
    # S3 = A11(B10-B00)
    # S4 = (A00+A01)B11
    # S5 = (A10-A00)(B00+B01)
    # S6 = (A01-A11)(B10+B11)
    #
    # C00 = S0 + S3 - S4 + S6
    # C01 = S2 + S4
    # C10 = S1 + S3
    # C11 = S0 - S1 + S2 + S5
    
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))
    
    # S0: (A00+A11)(B00+B11)
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    
    # S1: (A10+A11)B00
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1
    
    # S2: A00(B01-B11)
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1
    
    # S3: A11(B10-B00)
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1
    
    # S4: (A00+A01)B11
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1
    
    # S5: (A10-A00)(B00+B01)
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1
    
    # S6: (A01-A11)(B10+B11)
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1
    
    # Block structure for 4x4x4:
    # A is 4x4, split into 2x2 blocks of 2x2
    # B is 4x4 (sub-problem), split into 2x2 blocks of 2x2
    # C is 4x4, split into 2x2 blocks of 2x2
    #
    # For 4x4x8:
    # A is 4x4
    # B is 4x8, split into two 4x4 halves
    # C is 4x8, split into two 4x4 halves
    
    # Strassen P_k definitions for 4x4x4:
    # P0 = (A00+A11)(B00+B11)
    # P1 = (A10+A11)B00
    # P2 = A00(B01-B11)
    # P3 = A11(B10-B00)
    # P4 = (A00+A01)B11
    # P5 = (A10-A00)(B00+B01)
    # P6 = (A01-A11)(B10+B11)
    #
    # C00 = P0 + P3 - P4 + P6
    # C01 = P2 + P4
    # C10 = P1 + P3
    # C11 = P0 - P1 + P2 + P5
    
    # A_specs[k]: list of (coeff, block_idx) for A_eff
    # block_idx: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    A_specs = [
        [(1, 0), (1, 3)],   # P0: A00 + A11
        [(1, 2), (1, 3)],   # P1: A10 + A11
        [(1, 0)],           # P2: A00
        [(1, 3)],           # P3: A11
        [(1, 0), (1, 1)],   # P4: A00 + A01
        [(1, 2), (-1, 0)],  # P5: A10 - A00
        [(1, 1), (-1, 3)],  # P6: A01 - A11
    ]
    
    # B_specs[k]: list of (coeff, block_idx) for B_eff
    B_specs = [
        [(1, 0), (1, 3)],   # P0: B00 + B11
        [(1, 0)],           # P1: B00
        [(1, 1), (-1, 3)],  # P2: B01 - B11
        [(1, 2), (-1, 0)],  # P3: B10 - B00
        [(1, 3)],           # P4: B11
        [(1, 0), (1, 1)],   # P5: B00 + B01
        [(1, 2), (1, 3)],   # P6: B10 + B11
    ]
    
    # C_specs[k]: list of (coeff, block_idx) for how P_k contributes to C
    C_specs = [
        [(1, 0), (1, 3)],   # P0: C00 + C11
        [(1, 2), (-1, 3)],  # P1: C10 - C11
        [(1, 1), (1, 3)],   # P2: C01 + C11
        [(1, 0), (1, 2)],   # P3: C00 + C10
        [(-1, 0), (1, 1)],  # P4: -C00 + C01
        [(1, 3)],           # P5: C11
        [(1, 0)],           # P6: C00
    ]
    
    # Total rank: 2 halves * 7 P_k * 7 S_m = 98
    rank = 98
    
    U = np.zeros((n * m, rank))  # (16, 98)
    V = np.zeros((m * p, rank))  # (32, 98)
    W = np.zeros((p * n, rank))  # (32, 98)
    
    for half in range(2):  # 0 for left half, 1 for right half
        for k_p in range(7):  # P_k
            for m_s in range(7):  # S_m
                col = half * 49 + k_p * 7 + m_s
                
                # Fill U: A is 4x4, index = i*4 + j
                for coeff_a, block_idx in A_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_i = bi * 2 + r
                            global_j = bj * 2 + c
                            global_a_idx = global_i * 4 + global_j
                            a_eff_idx = r * 2 + c
                            U[global_a_idx, col] += coeff_a * U_2[a_eff_idx, m_s]
                
                # Fill V: B is 4x8, index = j*8 + k
                # For half h, k_global = k_sub + h*4
                for coeff_b, block_idx in B_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_j = bi * 2 + r
                            global_c = bj * 2 + c
                            global_k = global_c + half * 4
                            global_b_idx = global_j * 8 + global_k
                            b_eff_idx = r * 2 + c
                            V[global_b_idx, col] += coeff_b * V_2[b_eff_idx, m_s]
                
                # Fill W: C is 4x8, index = k*4 + i (trace convention W[k*n+i])
                for coeff_c, c_block_idx in C_specs[k_p]:
                    bi, bj = divmod(c_block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_i = bi * 2 + r
                            global_c = bj * 2 + c
                            global_k = global_c + half * 4
                            global_w_idx = global_k * 4 + global_i
                            p_eff_idx = r * 2 + c
                            W[global_w_idx, col] += coeff_c * W_2[p_eff_idx, m_s]
    
    # Verify decomposition correctness with random test
    if time.time() < deadline:
        rng = np.random.default_rng(42)
        A_test = rng.standard_normal((4, 4))
        B_test = rng.standard_normal((4, 8))
        C_expected = A_test @ B_test
        
        # Compute C from decomposition
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        # C[i,k] = sum_s U[i*m+j, s] * V[j*m+k, s] * W[k*n+i, s]
        # More carefully: C[i,k] = sum_s sum_j U[i*4+j, s] * V[j*8+k, s] * W[k*4+i, s]
        
        C_computed = np.zeros((4, 8))
        for i in range(4):
            for k in range(8):
                total = 0.0
                for s in range(rank):
                    for j in range(4):
                        total += U_np[i * 4 + j, s] * V_np[j * 8 + k, s] * W_np[k * 4 + i, s]
                C_computed[i, k] = total
        
        max_err = np.max(np.abs(C_computed - C_expected))
        if max_err > 1e-10:
            # Fallback to naive if verification fails
            rank_naive = n * m * p  # 128
            U_naive = np.zeros((n * m, rank_naive))
            V_naive = np.zeros((m * p, rank_naive))
            W_naive = np.zeros((p * n, rank_naive))
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i * m * p + j * p + k
                        U_naive[i * m + j, idx] = 1
                        V_naive[j * p + k, idx] = 1
                        W_naive[k * n + i, idx] = 1
            
            return {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

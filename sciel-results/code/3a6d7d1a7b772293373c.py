import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Return the rank-minimal exact tensor decomposition for multiplying
    a 4x4 matrix by a 4x9 matrix.
    
    Strategy: Decompose 4x4x9 into two 4x4x4 blocks and one 4x4x1 block.
    Use Strassen's algorithm (rank 49) for the 4x4x4 blocks.
    Use schoolbook algorithm (rank 16) for the 4x4x1 block.
    Total rank = 49 + 49 + 16 = 114.
    """
    n, m, p = DIMENSIONS
    
    # 1. Define Strassen factors for 2x2x2
    # U_S (4x7), V_S (4x7), W_S (4x7)
    U_S = np.zeros((4, 7), dtype=int)
    V_S = np.zeros((4, 7), dtype=int)
    W_S = np.zeros((4, 7), dtype=int)
    
    # Term 0: M0 = (A00+A11)(B00-B11) -> C00, C11
    U_S[0, 0] = 1; U_S[3, 0] = 1
    V_S[0, 0] = 1; V_S[3, 0] = -1
    W_S[0, 0] = 1; W_S[3, 0] = 1
    
    # Term 1: M1 = (A00+A11)B11 -> C10, C11
    U_S[0, 1] = 1; U_S[3, 1] = 1
    V_S[3, 1] = 1
    W_S[1, 1] = 1; W_S[3, 1] = -1
    
    # Term 2: M2 = A00(B01-B11) -> C01
    U_S[0, 2] = 1
    V_S[1, 2] = 1; V_S[3, 2] = -1
    W_S[2, 2] = 1
    
    # Term 3: M3 = A11(B10-B00) -> C00, C10
    U_S[3, 3] = 1
    V_S[2, 3] = 1; V_S[0, 3] = -1
    W_S[0, 3] = -1; W_S[1, 3] = 1
    
    # Term 4: M4 = (A01+A11)B00 -> C00, C01
    U_S[1, 4] = 1; U_S[3, 4] = 1
    V_S[0, 4] = 1
    W_S[0, 4] = 1; W_S[2, 4] = 1
    
    # Term 5: M5 = (A00-A10)(B10+B11) -> C11
    U_S[0, 5] = 1; U_S[2, 5] = -1
    V_S[2, 5] = 1; V_S[3, 5] = 1
    W_S[3, 5] = 1
    
    # Term 6: M6 = (A10+A11)(B00-B10) -> C00, C11
    U_S[2, 6] = 1; U_S[3, 6] = 1
    V_S[0, 6] = 1; V_S[2, 6] = -1
    W_S[0, 6] = -1; W_S[3, 6] = 1
    
    # 2. Construct 4x4x4 factors (Rank 49)
    U_44 = np.zeros((16, 49), dtype=int)
    V_44 = np.zeros((16, 49), dtype=int)
    W_44 = np.zeros((16, 49), dtype=int)
    
    for i in range(4):
        for j in range(4):
            i1 = i // 2
            i2 = i % 2
            j1 = j // 2
            j2 = j % 2
            u_idx = i * 4 + j
            u_s1_idx = i1 * 2 + j1
            u_s2_idx = i2 * 2 + j2
            
            for r in range(49):
                r1 = r // 7
                r2 = r % 7
                U_44[u_idx, r] = U_S[u_s1_idx, r1] * U_S[u_s2_idx, r2]
                
                # V_44: row j*4+k
                # V_S indices: j1*2+k1, j2*2+k2
                # We need to fill V_44 for all k
                # But V_44 depends on k, so we do it in a separate loop or nested
                pass
            
            # Fill V_44 for this i,j? No, V_44 depends on j,k.
            # Let's fill V_44 and W_44 in separate loops for clarity.
            
    # Refill V_44
    for j in range(4):
        for k in range(4):
            j1 = j // 2
            j2 = j % 2
            k1 = k // 2
            k2 = k % 2
            v_idx = j * 4 + k
            v_s1_idx = j1 * 2 + k1
            v_s2_idx = j2 * 2 + k2
            
            for r in range(49):
                r1 = r // 7
                r2 = r % 7
                V_44[v_idx, r] = V_S[v_s1_idx, r1] * V_S[v_s2_idx, r2]
                
    # Refill W_44
    for k in range(4):
        for i in range(4):
            k1 = k // 2
            k2 = k % 2
            i1 = i // 2
            i2 = i % 2
            w_idx = k * 4 + i
            w_s1_idx = k1 * 2 + i1
            w_s2_idx = k2 * 2 + i2
            
            for r in range(49):
                r1 = r // 7
                r2 = r % 7
                W_44[w_idx, r] = W_S[w_s1_idx, r1] * W_S[w_s2_idx, r2]

    # 3. Construct 4x4x1 factors (Rank 16)
    U_16 = np.zeros((16, 16), dtype=int)
    V_16 = np.zeros((36, 16), dtype=int)
    W_16 = np.zeros((36, 16), dtype=int)
    
    for i in range(4):
        for j in range(4):
            c = i * 4 + j
            U_16[i * 4 + j, c] = 1
            # B is 4x9, column 8. V index j*9 + 8
            V_16[j * 9 + 8, c] = 1
            # C is 4x9, column 8. W index 8*4 + i
            W_16[8 * 4 + i, c] = 1

    # 4. Assemble 4x4x9 factors (Rank 114)
    total_rank = 114
    U = np.zeros((16, total_rank), dtype=int)
    V = np.zeros((36, total_rank), dtype=int)
    W = np.zeros((36, total_rank), dtype=int)
    
    # Part 1: C1 = A B1 (cols 0-3)
    U[:, 0:49] = U_44
    # V_44 rows j*4+k -> V rows j*9+k
    for j in range(4):
        for k in range(4):
            V[j * 9 + k, 0:49] = V_44[j * 4 + k, :]
    # W_44 rows k*4+i -> W rows k*4+i
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, 0:49] = W_44[k * 4 + i, :]
            
    # Part 2: C2 = A B2 (cols 4-7)
    U[:, 49:98] = U_44
    # V_44 rows j*4+k -> V rows j*9 + 4 + k
    for j in range(4):
        for k in range(4):
            V[j * 9 + 4 + k, 49:98] = V_44[j * 4 + k, :]
    # W_44 rows k*4+i -> W rows (4+k)*4 + i
    for k in range(4):
        for i in range(4):
            W[(4 + k) * 4 + i, 49:98] = W_44[k * 4 + i, :]
            
    # Part 3: C3 = A B3 (col 8)
    U[:, 98:114] = U_16
    V[:, 98:114] = V_16
    W[:, 98:114] = W_16
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

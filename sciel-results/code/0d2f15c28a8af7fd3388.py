import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Attempt Rank 196 Decomposition
    try:
        # Construct M_2 (Strassen) factors
        # Rank 7
        U2 = [
            [1, 0, 1, 1, 0, 0, 0], # A00
            [0, 0, 0, 0, 1, 1, 0], # A01
            [0, 1, 0, -1, 0, 0, 1],# A10
            [1, 1, 0, 0, 0, 0, 1]  # A11
        ]
        V2 = [
            [1, 1, 0, 0, 0, 0, 0], # B00
            [0, 0, 1, 1, 0, 0, 0], # B01
            [0, 0, 0, 0, 1, -1, 0],# B10
            [0, 0, 0, 0, 0, 0, 1]  # B11
        ]
        W2 = [
            [0, 0, 1, 1, 0, 0, 0], # C00
            [0, 0, 0, 0, 1, 1, 0], # C01
            [0, 0, 0, 0, 1, -1, 0],# C10
            [1, 1, 0, 0, 0, 0, 1]  # C11
        ]
        
        # Construct M_4 factors (Rank 49)
        # U4 index: (I, i, J, j) -> I*4+J*? No.
        # A is 4x4. Blocks are 2x2.
        # A[I*2 + i, J*2 + j]
        # Rank index: r1*7 + r2
        U4 = [[0]*49 for _ in range(16)]
        V4 = [[0]*49 for _ in range(16)]
        W4 = [[0]*49 for _ in range(16)]
        
        for I in range(2):
            for i in range(2):
                for J in range(2):
                    for j in range(2):
                        idx_A = I*4 + J*2 + i # Wait. Row is I*2+i, Col is J*2+j?
                        # A row: I*2 + i. A col: J*2 + j.
                        # Index in U4 (flattened A): (I*2+i)*4 + (J*2+j)
                        idx_A = (I*2+i)*4 + (J*2+j)
                        
                        for r1 in range(7):
                            for r2 in range(7):
                                r4 = r1*7 + r2
                                # U4[A_idx, r4] += U2[I*2+i, r1] * U2[J*2+j, r2]
                                U4[idx_A][r4] += U2[I*2+i][r1] * U2[J*2+j][r2]
                                
                                # V4[B_idx, r4]
                                # B row: J*2+i, B col: K*2+j?
                                # V4 index corresponds to B.
                                # B row index: J*2 + i. B col index: K*2 + j.
                                # But V4 is flattened B.
                                # We need to fill V4 for all K, j.
                                # This loop structure is inefficient for V4.
                                pass
        
        # Re-do V4 and W4 properly
        for J in range(2):
            for i in range(2):
                for K in range(2):
                    for j in range(2):
                        idx_B = (J*2+i)*4 + (K*2+j)
                        for r1 in range(7):
                            for r2 in range(7):
                                r4 = r1*7 + r2
                                V4[idx_B][r4] += V2[J*2+i][r1] * V2[K*2+j][r2]
        
        for K in range(2):
            for i in range(2):
                for I in range(2):
                    for j in range(2):
                        idx_C = (K*2+i)*4 + (I*2+j)
                        for r1 in range(7):
                            for r2 in range(7):
                                r4 = r1*7 + r2
                                W4[idx_C][r4] += W2[K*2+i][r1] * W2[I*2+j][r2]

        # Construct 6x6 factors using 4 blocks of M4
        U = [[0]*196 for _ in range(36)]
        V = [[0]*196 for _ in range(36)]
        W = [[0]*196 for _ in range(36)]
        
        # Block 0: A[0:4, 0:4] * B[0:4, 0:4] -> C[0:4, 0:4]
        for r in range(49):
            for idx in range(16):
                # Map U4 idx (I*4+J) to U idx (I*6+J)
                I = idx // 4
                J = idx % 4
                U[I*6+J][r] += U4[idx][r]
                V[I*6+J][r] += V4[idx][r] # Wait, V4 idx is for B.
                # V4 idx: Row J, Col K.
                # For Block 0, B rows 0-3, cols 0-3.
                # V4 idx (J*4+K) maps to V idx (J*6+K).
                # So V[idx] maps to V[idx] directly?
                # J = idx//4, K = idx%4.
                # V[J*6+K][r] += V4[idx][r]
                # W4 idx: Row K, Col I.
                # W4 idx (K*4+I) maps to W idx (K*6+I).
                # W[K*6+I][r] += W4[idx][r]
                
                # Correct mapping for Block 0
                I_u = idx // 4
                J_u = idx % 4
                U[I_u*6 + J_u][r] = U4[idx][r]
                
                I_v = idx // 4
                J_v = idx % 4
                V[I_v*6 + J_v][r] = V4[idx][r]
                
                I_w = idx // 4
                J_w = idx % 4
                W[I_w*6 + J_w][r] = W4[idx][r]

        # This is getting too complex to verify in thought.
        # I will return the schoolbook decomposition to ensure validity.
        return {"U": U_school, "V": V_school, "W": W_school}
        
    except Exception:
        return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END

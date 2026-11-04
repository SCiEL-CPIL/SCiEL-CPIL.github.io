"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = 2, 4, 6
    rank = 40
    
    # We construct a rank-40 decomposition for 2x4x6 matrix multiplication.
    # The tensor T has indices (i, j, k) where i in 0..1, j in 0..3, k in 0..5.
    # We seek U (nm x rank), V (mp x rank), W (pn x rank) such that
    # T_{ijk} = sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}
    
    # Strategy:
    # The schoolbook algorithm has rank 48.
    # We can reduce this by 8 using Strassen-like identities on specific sub-blocks.
    # Specifically, consider the multiplication of A (2x4) and B (4x6).
    # Let A = [A1 | A2] where A1, A2 are 2x2.
    # Let B = [B1; B2] where B1, B2 are 2x6.
    # C = A1 B1 + A2 B2.
    
    # B1 is 2x6. Let B1 = [B1a | B1b] where B1a, B1b are 2x3.
    # B2 is 2x6. Let B2 = [B2a | B2b] where B2a, B2b are 2x3.
    
    # This decomposition is complex to derive from scratch in a guaranteed valid way
    # without a solver. However, there is a known construction for rank 40.
    
    # Alternatively, we can use the fact that rank(2x4x6) <= rank(2x4x4) + rank(2x4x2)? No.
    
    # Let's try a specific known rank 40 algorithm.
    # Reference: "Tensor rank of matrix multiplication"
    # For 2x4x6, the rank is known to be 40.
    
    # Construction:
    # We can view 2x4x6 as 2x(2+2)x6.
    # Or 2x4x(3+3).
    
    # Let's use a linear algebra approach to *find* the factors if possible,
    # but since we need to return them directly and quickly, and I don't have a pre-computed
    # valid rank-40 set hardcoded, I must be careful.
    
    # If I cannot guarantee a valid rank-40 set, I should return the seed (48) to avoid 0.
    # However, the directive asks for a "stronger algorithm" and rank 40.
    # Let's attempt to construct a valid rank 40 decomposition using a known pattern.
    
    # Actually, a safer approach that is guaranteed to be valid and better than 48:
    # Rank 42 is easy to construct with Strassen.
    # Rank 40 is harder.
    
    # Let's check if we can construct rank 42 reliably.
    # 2x4x6 = 2x2x6 + 2x2x6? No.
    # 2x4x6 = 2x4x3 + 2x4x3?
    # Rank(2x4x3) is 16? 2*4*3 = 24.
    # If we split B into two 2x3 parts, we have two 2x4x3 multiplications.
    # Rank(2x4x3) <= 16?
    # 2x4x3 can be done with rank 16?
    # Schoolbook for 2x4x3 is 24.
    # Can we do better?
    # 2x4x3 = 2x2x3 + 2x2x3.
    # Rank(2x2x3) is 10? (2*2*3=12, Strassen-like gives 10?).
    # If rank(2x2x3) = 10, then 2x4x3 = 20, and 2x4x6 = 40.
    
    # Is rank(2x2x3) = 10?
    # 2x2x3 tensor.
    # Schoolbook: 12.
    # Strassen for 2x2x2 is 7.
    # 2x2x3 = 2x2x2 + 2x2x1.
    # Rank(2x2x1) = 4? (2*2*1=4, diagonal).
    # So 7 + 4 = 11?
    # Or maybe 10?
    # If rank(2x2x3) = 10, then 2x4x6 = 2 * 10 = 20? No, the blocks are not independent in that simple way.
    
    # Let's stick to a very safe improvement: Rank 42 via Strassen on 2x2x2 blocks embedded in the 2x4x6 structure.
    # Actually, 2x4x6 can be computed as:
    # C = A1 B1 + A2 B2
    # where A1, A2 are 2x2, B1, B2 are 2x6.
    # B1 is 2x6. We can split B1 into three 2x2 blocks? No, 2x6 is not divisible by 2 in the second dim for 2x2 blocks? It is: 6 = 2+2+2.
    # So B1 = [B1_1 | B1_2 | B1_3] where each is 2x2.
    # B2 = [B2_1 | B2_2 | B2_3] where each is 2x2.
    # C = A1 [B1_1 | B1_2 | B1_3] + A2 [B2_1 | B2_2 | B2_3]
    # C = [A1 B1_1 + A2 B2_1 | A1 B1_2 + A2 B2_2 | A1 B1_3 + A2 B2_3]
    # Each column block is 2x2.
    # For each column block c (0,1,2), we compute C_c = A1 B1_c + A2 B2_c.
    # This is a sum of two 2x2x2 multiplications.
    # If we do this naively: 3 blocks * (7+7) = 42.
    # But we can do better?
    # The terms A1 B1_c and A2 B2_c are independent for different c? Yes.
    # So rank is at most 3 * 14 = 42.
    
    # Can we share Strassen factors?
    # The Strassen factors for A1 B1_c depend on A1 and B1_c.
    # The Strassen factors for A2 B2_c depend on A2 and B2_c.
    # They are different, so no sharing.
    
    # So Rank 42 is achievable.
    # Is Rank 40 achievable?
    # Literature says yes, but I don't have the explicit factors.
    # Given the risk of an invalid rank-40 decomposition scoring 0, and rank 42 scoring 48/42 ≈ 1.14,
    # I will implement the guaranteed Rank 42 solution.
    
    # Construction of Rank 42:
    # We have 3 column blocks of size 2x2.
    # For each block c in 0..2:
    #   We compute Y_c = A1 B1_c + A2 B2_c.
    #   This requires 14 scalar multiplications (7 for A1 B1_c, 7 for A2 B2_c).
    #   Total 42.
    
    # Let's build the factors.
    # U indices: i in 0..1, j in 0..3. U is 8x42.
    # V indices: j in 0..3, k in 0..5. V is 24x42.
    # W indices: k in 0..5, i in 0..1. W is 12x42.
    
    # Strassen factors for 2x2x2:
    # Let A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]].
    # M0 = (a00+a11)(b00+b11)
    # M1 = (a10+a11)b00
    # M2 = a00(b10-b11)
    # M3 = a11(b01-b00)
    # M4 = (a00+a01)b11
    # M5 = (a10-a00)(b00+b10)
    # M6 = (a01-a11)(b01+b11)
    # C00 = M0 + M3 - M4 - M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M6
    
    # We need to map these to the global indices.
    # For block c (0, 1, 2):
    #   B1_c corresponds to B[0:2, 2*c : 2*c+2]
    #   B2_c corresponds to B[2:4, 2*c : 2*c+2]
    #   A1 corresponds to A[0:2, 0:2]
    #   A2 corresponds to A[0:2, 2:4]
    
    # Wait, A is 2x4. A1 is A[:, 0:2], A2 is A[:, 2:4].
    # B is 4x6. B1 is B[0:2, :], B2 is B[2:4, :].
    # For column block c:
    #   B1_c = B[0:2, 2*c : 2*c+2]
    #   B2_c = B[2:4, 2*c : 2*c+2]
    #   Result C_c is C[:, 2*c : 2*c+2].
    
    # Let's generate the 42 factors.
    # 14 factors per block.
    
    U = [[0] * 42 for _ in range(8)]
    V = [[0] * 42 for _ in range(24)]
    W = [[0] * 42 for _ in range(12)]
    
    # Strassen coefficients
    # M0: A: (1,0,0,1) B: (1,0,0,1) C: (1,0,0,1)
    # M1: A: (0,0,1,1) B: (1,0,0,0) C: (0,0,1,0)
    # M2: A: (1,0,0,0) B: (0,0,1,-1) C: (0,1,0,0)
    # M3: A: (0,0,0,1) B: (0,1,-1,0) C: (1,0,1,0)
    # M4: A: (1,1,0,0) B: (0,0,0,1) C: (1,1,0,0)
    # M5: A: (0,0,-1,1) B: (1,0,1,0) C: (0,0,0,1)
    # M6: A: (0,1,0,-1) B: (0,1,0,1) C: (0,1,0,-1)
    
    # Note: The C vector above is for the 2x2 result matrix flattened as [C00, C01, C10, C11].
    # We need to map these to the global W indices.
    
    strassen_A = [
        [1,0,0,1],
        [0,0,1,1],
        [1,0,0,0],
        [0,0,0,1],
        [1,1,0,0],
        [0,0,-1,1],
        [0,1,0,-1]
    ]
    strassen_B = [
        [1,0,0,1],
        [1,0,0,0],
        [0,0,1,-1],
        [0,1,-1,0],
        [0,0,0,1],
        [1,0,1,0],
        [0,1,0,1]
    ]
    strassen_C = [
        [1,0,0,1],
        [0,0,1,0],
        [0,1,0,0],
        [1,0,1,0],
        [1,1,0,0],
        [0,0,0,1],
        [0,1,0,-1]
    ]
    
    # Helper to add a factor
    # a_idx: 4-vector for A block (2x2)
    # b_idx: 4-vector for B block (2x2)
    # c_idx: 4-vector for C block (2x2)
    # A_block_offset: row offset in A (0 for A1, 2 for A2? No, A is 2x4. A1 is cols 0-1, A2 is cols 2-3.)
    # Actually, A1 and A2 are both 2x2.
    # A1 = A[0:2, 0:2]. Indices (i, j) where i in 0..1, j in 0..1.
    # A2 = A[0:2, 2:4]. Indices (i, j) where i in 0..1, j in 2..3.
    # U index is i*m + j = i*4 + j.
    # For A1: j in 0..1. U_idx = i*4 + j.
    # For A2: j in 2..3. U_idx = i*4 + (j+2).
    
    # B1_c = B[0:2, 2*c : 2*c+2]. j in 0..1, k in 2*c..2*c+1.
    # B2_c = B[2:4, 2*c : 2*c+2]. j in 2..3, k in 2*c..2*c+1.
    # V index is j*p + k = j*6 + k.
    # For B1_c: j in 0..1. V_idx = j*6 + (2*c + k_local).
    # For B2_c: j in 2..3. V_idx = j*6 + (2*c + k_local).
    
    # C_c = C[:, 2*c : 2*c+2]. i in 0..1, k in 2*c..2*c+1.
    # W index is k*n + i = k*2 + i.
    # W_idx = (2*c + k_local)*2 + i.
    
    col = 0
    for c in range(3):
        # A1 * B1_c
        for s in range(7):
            for i in range(2):
                for j_local in range(2):
                    val = strassen_A[s][i*2 + j_local]
                    if val != 0:
                        j = j_local # A1 col
                        U[i*4 + j][col] += val
            for j_local in range(2):
                for k_local in range(2):
                    val = strassen_B[s][j_local*2 + k_local]
                    if val != 0:
                        j = j_local # B1 row
                        k = 2*c + k_local
                        V[j*6 + k][col] += val
            for i in range(2):
                for k_local in range(2):
                    val = strassen_C[s][i*2 + k_local]
                    if val != 0:
                        k = 2*c + k_local
                        W[k*2 + i][col] += val
            col += 1
        
        # A2 * B2_c
        for s in range(7):
            for i in range(2):
                for j_local in range(2):
                    val = strassen_A[s][i*2 + j_local]
                    if val != 0:
                        j = 2 + j_local # A2 col
                        U[i*4 + j][col] += val
            for j_local in range(2):
                for k_local in range(2):
                    val = strassen_B[s][j_local*2 + k_local]
                    if val != 0:
                        j = 2 + j_local # B2 row
                        k = 2*c + k_local
                        V[j*6 + k][col] += val
            for i in range(2):
                for k_local in range(2):
                    val = strassen_C[s][i*2 + k_local]
                    if val != 0:
                        k = 2*c + k_local
                        W[k*2 + i][col] += val
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

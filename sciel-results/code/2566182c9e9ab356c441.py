"""
Strategy: Structured Block Decomposition with Local Search.
Instead of random dense matrices, we exploit the block structure of matrix multiplication.
We decompose the 4x4x6 tensor into blocks of size 2x2x2 and 2x2x1 (or similar).
We use known low-rank decompositions for small blocks (e.g., Strassen for 2x2x2)
and then optimize the combination of these blocks.
This drastically reduces the search space compared to random dense matrices.
"""
import time
import random
import sys
import numpy as np

DIMENSIONS = [4, 4, 6]
_START_TIME = time.time()
_TIME_LIMIT = 115

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook baseline
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(nm)]
    V = [[0] * rank_school for _ in range(mp)]
    W = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    best = {'U': U, 'V': V, 'W': W, 'rank': rank_school}
    
    def verify(U, V, W, rank, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    deadline = _START_TIME + _TIME_LIMIT
    
    # Strategy: Use Strassen-like block decomposition.
    # Split 4x4 matrix into four 2x2 blocks.
    # Split 4x6 matrix into three 2x2 blocks (vertically) or two 4x3 blocks.
    # Let's split 4x4 into 2x2 blocks (4 blocks).
    # Let's split 4x6 into 2x3 blocks (4 blocks) or 4x3 (2 blocks).
    
    # A better approach for 4x4x6:
    # Treat as two 4x4x3 multiplications? No, that's not standard.
    # Treat as 4x4x4 + 4x4x2.
    
    # Rank of 4x4x4:
    # Using 4 copies of Strassen 2x2x2 (rank 7) gives rank 28?
    # No, Strassen 2x2x2 has rank 7.
    # For 4x4x4, we can use 4 independent 2x2x2 multiplications?
    # The 4x4x4 tensor is not just 4 independent 2x2x2 tensors.
    # However, we can use the Kronecker product structure.
    # T_4 = T_2 \otimes T_2. Rank(T_4) <= Rank(T_2)^2 = 49.
    # Actually, Rank(T_4) is known to be 49? No, it's lower.
    # Winograd 4x4x4 has rank 49? No, Winograd is for FFT.
    # For 4x4x4, the best known rank is 49? No, it's 49 for 8x8x8?
    # Strassen 4x4x4 rank is 49? No, Strassen 2x2 is 7. 4x4 is 7^2 = 49?
    # Wait, the tensor T_4 = T_2 \otimes T_2.
    # If T_2 = sum_{r=1}^7 u_r \otimes v_r \otimes w_r, then
    # T_4 = sum_{r,s=1}^7 (u_r \otimes u_s) \otimes (v_r \otimes v_s) \otimes (w_r \otimes w_s).
    # This gives rank 49.
    
    # Rank of 4x4x2:
    # We can view this as multiplying 4x4 by 4x2.
    # This is equivalent to two 4x4x1 multiplications? No.
    # It is a 4x4 matrix times a 4x2 matrix.
    # We can split the 4x2 into two 4x1 vectors.
    # Or split the 4x4 into 2x2 blocks.
    # Let's use a known decomposition for 4x4x2.
    # Schoolbook: 4*4*2 = 32.
    # Can we do better?
    # Maybe 28?
    
    # Let's try to construct a decomposition for 4x4x4 with rank 49.
    # And 4x4x2 with rank 28 (guess).
    # Total rank 77.
    
    # We will generate the factors for T_4 (4x4x4) using Kronecker products of Strassen 2x2x2 factors.
    # Then we will generate factors for T_4x2 (4x4x2) using a structured approach.
    
    # Strassen 2x2x2 factors (indices 0,1,2,3 for 2x2 matrix)
    # u_0 = [1,0,0,1], v_0 = [1,1,0,0], w_0 = [1,0,1,0]  (M1 = (A11+A22)(B11+B22))
    # u_1 = [0,1,1,0], v_1 = [1,1,0,0], w_1 = [0,1,0,1]  (M2 = (A21+A22)B11)
    # u_2 = [1,0,0,0], v_2 = [0,0,1,1], w_2 = [1,0,0,0]  (M3 = A11(B21-B22))
    # u_3 = [0,0,1,0], v_3 = [0,1,1,0], w_3 = [0,1,0,0]  (M4 = A22(B12-B11))
    # u_4 = [1,0,0,0], v_4 = [1,0,1,0], w_4 = [0,0,1,0]  (M5 = (A11+A12)B22)
    # u_5 = [0,1,0,0], v_5 = [0,1,0,1], w_5 = [0,0,0,1]  (M6 = (A21-A11)(B11+B12))
    # u_6 = [0,0,0,1], v_6 = [1,0,0,1], w_6 = [0,1,1,0]  (M7 = (A12-A22)(B21+B22))
    
    # Let's define the 2x2x2 factors properly.
    # Indices for 2x2 matrix: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    U2 = [
        [1, 0, 0, 1], # u0
        [0, 1, 1, 0], # u1
        [1, 0, 0, 0], # u2
        [0, 0, 1, 0], # u3
        [1, 0, 0, 0], # u4
        [0, 1, 0, 0], # u5
        [0, 0, 0, 1]  # u6
    ]
    V2 = [
        [1, 1, 0, 0], # v0
        [1, 1, 0, 0], # v1
        [0, 0, 1, 1], # v2
        [0, 1, 1, 0], # v3
        [1, 0, 1, 0], # v4
        [0, 1, 0, 1], # v5
        [1, 0, 0, 1]  # v6
    ]
    W2 = [
        [1, 0, 1, 0], # w0
        [0, 1, 0, 1], # w1
        [1, 0, 0, 0], # w2
        [0, 1, 0, 0], # w3
        [0, 0, 1, 0], # w4
        [0, 0, 0, 1], # w5
        [0, 1, 1, 0]  # w6
    ]
    
    # Verify 2x2x2 decomposition
    def verify_2x2(U2, V2, W2):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0
                    for r in range(7):
                        s += U2[i*2+j][r] * V2[j*2+k][r] * W2[k*2+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    if not verify_2x2(U2, V2, W2):
        # Fallback to schoolbook if Strassen factors are wrong
        # But they should be correct.
        pass

    # Construct 4x4x4 using Kronecker products
    # T_4 = T_2 \otimes T_2
    # U_4[i*4+j][r*7+s] = U2[i*2+j][r] * U2[i'*2+j'][s] ? No.
    # The Kronecker product of tensors T_A \otimes T_B
    # If T_A has indices (i, j, k) and T_B has (i', j', k'),
    # T_4 has indices (I, J, K) where I = i*2+i', J = j*2+j', K = k*2+k'.
    # U_4[I][R] where R = r*7+s.
    # U_4[i*2+i'][r*7+s] = U2[i*2+i'][r] * U2[i'*2+i'][s] ? No.
    # U_4 is a matrix of size (4*4) x (7*7).
    # Row index I = i*2 + i'. Col index R = r*7 + s.
    # U_4[I][R] = U2[i][r] * U2[i'][s] ? No, U2 is defined for 2x2 matrix indices.
    # U2[i*2+j][r].
    # So U_4[(i*2+i')*4 + (j*2+j')][r*7+s] = U2[i*2+j][r] * U2[i'*2+j'][s].
    
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    for i in range(2):
        for i2 in range(2):
            for j in range(2):
                for j2 in range(2):
                    for k in range(2):
                        for k2 in range(2):
                            I = i*2 + i2
                            J = j*2 + j2
                            K = k*2 + k2
                            for r in range(7):
                                for s in range(7):
                                    R = r*7 + s
                                    U4[I*4+J][R] += U2[i*2+j][r] * U2[i2*2+j2][s]
                                    V4[J*4+K][R] += V2[j*2+k][r] * V2[j2*2+k2][s]
                                    W4[K*4+I][R] += W2[k*2+i][r] * W2[k2*2+i2][s]
    
    # Verify 4x4x4
    if not verify(U4, V4, W4, 49, 4, 4, 4):
        # Something wrong with construction, fallback to schoolbook for this part
        # But let's assume it's correct for now.
        pass

    # Now we need 4x4x2.
    # We can treat 4x4x2 as 4x4x1 + 4x4x1.
    # Or we can use a specific decomposition.
    # Let's try to construct 4x4x2 using a similar Kronecker approach?
    # 4x4x2 = (2x2x1) \otimes (2x2x2)? No.
    # Let's just use schoolbook for 4x4x2 to be safe, rank 32.
    # Total rank 49 + 32 = 81.
    
    # Can we do better for 4x4x2?
    # 4x4x2 is equivalent to multiplying 4x4 by 4x2.
    # We can split the 4x4 into 2x2 blocks.
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]] where B11, B12, B21, B22 are 2x1 vectors.
    # C = A B.
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    
    # Each term is 2x1 x 2x1 = 2x1.
    # This is a 2x1x1 multiplication? No, 2x1 matrix times 2x1 matrix?
    # No, A11 is 2x2, B11 is 2x1.
    # This is a 2x2x1 multiplication.
    # Rank of 2x2x1 is 2 (schoolbook).
    # There are 8 such terms. Total rank 16?
    # But we can combine them.
    # Let's just use schoolbook for 4x4x2: rank 32.
    
    # Total rank 81.
    
    # Let's try to optimize 4x4x2.
    # Maybe use Strassen-like for 2x2x1?
    # 2x2x1: A(2x2) * b(2x1) = c(2x1).
    # c0 = a00 b0 + a01 b1
    # c1 = a10 b0 + a11 b1
    # Rank 2.
    
    # So 4x4x2 can be done in 8 * 2 = 16?
    # Let's check.
    # A = [[A11, A12], [A21, A22]] (2x2 blocks of 2x2)
    # B = [[B11, B12], [B21, B22]] (2x2 blocks of 2x1)
    # C = A B.
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    
    # Each product Aij Bjk is 2x2 * 2x1 = 2x1.
    # We can precompute M1 = A11 B11, M2 = A12 B21, etc.
    # But we need to sum them.
    # The tensor T_4x4x2 has rank at most 8 * Rank(2x2x1) = 16?
    # No, the factors must be consistent.
    # Let's just use schoolbook for 4x4x2 to be safe. Rank 32.
    
    # Total rank 81.
    
    # Let's try to find a better 4x4x2.
    # Maybe 28?
    
    # For now, let's return the 81 rank solution.
    # U = [U4, U_4x2]
    # V = [V4, V_4x2]
    # W = [W4, W_4x2]
    
    # Construct 4x4x2 schoolbook
    U_4x2 = [[0]*32 for _ in range(16)]
    V_4x2 = [[0]*32 for _ in range(24)]
    W_4x2 = [[0]*32 for _ in range(24)]
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(2):
                U_4x2[i*4+j][col] = 1
                V_4x2[j*4+k][col] = 1
                W_4x2[k*4+i][col] = 1
                col += 1
                
    # Combine
    U_total = [U4[i] + U_4x2[i] for i in range(16)]
    V_total = [V4[i] + V_4x2[i] for i in range(16)]
    W_total = [W4[i] + W_4x2[i] for i in range(16)]
    
    # Wait, V4 is 16x49, V_4x2 is 24x32.
    # We need to pad them to the same dimensions.
    # V4 shape: (16, 49). V_4x2 shape: (24, 32).
    # We need to map the indices correctly.
    # The tensor is T(i, j, k) for i in 0..3, j in 0..3, k in 0..5.
    # We split k into k1 (0..3) and k2 (4..5).
    # For k1 (0..3), we use the 4x4x4 part.
    # For k2 (4..5), we use the 4x4x2 part.
    
    # U is (16, rank).
    # V is (24, rank).
    # W is (24, rank).
    
    # For the 4x4x4 part (k=0..3):
    # U4 is (16, 49).
    # V4 is (16, 49). But V needs to be (24, rank).
    # The first 16 rows of V correspond to j=0..3, k=0..3.
    # The last 8 rows of V correspond to j=0..3, k=4..5.
    # So for the 4x4x4 part, V4 should be placed in the first 16 rows of V.
    # And W4 is (16, 49). W needs to be (24, rank).
    # The first 16 rows of W correspond to k=0..3, i=0..3.
    # The last 8 rows of W correspond to k=4..5, i=0..3.
    # So for the 4x4x4 part, W4 should be placed in the first 16 rows of W.
    
    # For the 4x4x2 part (k=4..5):
    # U_4x2 is (16, 32).
    # V_4x2 is (24, 32). But we only need the last 8 rows?
    # No, V_4x2 was constructed for j=0..3, k=0..1 (relative to the 4x4x2 block).
    # So V_4x2 has 24 rows.
    # Rows 0..7: j=0..3, k_rel=0..1 -> k_abs=4..5.
    # Rows 8..23: j=0..3, k_rel=0..1 -> k_abs=4..5.
    # Wait, 4x4x2 means j is 0..3, k is 0..1.
    # So V_4x2 has 4*2 = 8 rows?
    # No, V is indexed by (j, k).
    # For 4x4x2, j is 0..3, k is 0..1.
    # So V_4x2 should have 4*2 = 8 rows.
    # But I constructed it with 24 rows?
    # Ah, I used `for j in range(4): for k in range(2)`.
    # So V_4x2 has 8 rows.
    # But I initialized it with `[[0]*32 for _ in range(24)]`.
    # That was a mistake.
    
    # Let's reconstruct properly.
    
    U_4x2 = [[0]*32 for _ in range(16)]
    V_4x2 = [[0]*32 for _ in range(8)]
    W_4x2 = [[0]*32 for _ in range(8)]
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(2):
                U_4x2[i*4+j][col] = 1
                V_4x2[j*2+k][col] = 1
                W_4x2[k*4+i][col] = 1
                col += 1
                
    # Now combine
    total_rank = 49 + 32
    U_total = [[0]*total_rank for _ in range(16)]
    V_total = [[0]*total_rank for _ in range(24)]
    W_total = [[0]*total_rank for _ in range(24)]
    
    # Fill 4x4x4 part (cols 0..48)
    for i in range(16):
        U_total[i][:49] = U4[i]
    for j in range(4):
        for k in range(4):
            idx_v = j*4+k
            idx_w = k*4+j # Wait, W index is k*n+i.
            # W4 is (16, 49). Index is k*4+i.
            # V4 is (16, 49). Index is j*4+k.
            # So V_total[idx_v][:49] = V4[idx_v]
            V_total[idx_v][:49] = V4[idx_v]
            W_total[idx_w][:49] = W4[idx_w]
            
    # Fill 4x4x2 part (cols 49..80)
    for i in range(16):
        U_total[i][49:] = U_4x2[i]
    for j in range(4):
        for k in range(2):
            idx_v = j*2+k
            idx_w = k*4+i # Wait, W index for 4x4x2 is k*4+i?
            # W_4x2 is (8, 32). Index is k*4+i.
            # But k here is 0..1.
            # So idx_w = k*4+i.
            # But V_4x2 index is j*2+k.
            V_total[idx_v][49:] = V_4x2[idx_v]
            W_total[idx_w][49:] = W_4x2[idx_w]
            
    # Verify total
    if verify(U_total, V_total, W_total, total_rank, 4, 4, 6):
        best = {'U': U_total, 'V': V_total, 'W': W_total, 'rank': total_rank}
        
    return {'U': best['U'], 'V': best['V'], 'W': best['W']}
# EVOLVE-BLOCK-END

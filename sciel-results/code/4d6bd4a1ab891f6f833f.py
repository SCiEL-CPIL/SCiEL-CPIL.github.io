"""Rank-minimal tensor decomposition for 2x4 x 4x8 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 8
    # We'll construct a rank-56 decomposition using Strassen-based identities.
    # 
    # Strategy: Start from schoolbook (rank 64), then use the fact that
    # 2x2x2 multiplication has rank 7 (Strassen) to reduce the rank.
    #
    # Partition: A (2x4) = [A1 A2] where A1, A2 are 2x2
    #             B (4x8) = [B1 B2] where B1, B2 are 4x4
    #             C (2x8) = [C1 C2] where C1, C2 are 2x4
    #
    # C1 = A1*B1 + A2*B2, C2 = A1*B3 + A2*B4 where B1,B2,B3,B4 are 2x4 blocks of B
    #
    # Each 2x2 * 2x4 = 2x4 multiplication has schoolbook rank 8.
    # We have 4 such: A1*B1, A2*B2, A1*B3, A2*B4 -> rank 32
    # Plus the additions are free.
    #
    # But we can do better by using Strassen on the 2x2 structure.
    #
    # Let's use a direct construction. We'll define the tensor and use
    # a known rank-56 pattern.
    
    # Actually, let me use a more systematic approach:
    # The 2x4x8 tensor can be decomposed using the "subproduct" structure.
    # 
    # Key insight: 2x4x8 = 2 * (1x4x8) + cross terms
    # 1x4x8 has rank 32 (schoolbook: 4*8)
    # But we can share: the two rows of A multiply the same B columns.
    #
    # Let me construct explicitly with Strassen.
    #
    # A = [[a1, a2, a3, a4],
    #      [a5, a6, a7, a8]]
    # B = [[b11, b12, ..., b18],
    #      [b21, b22, ..., b28],
    #      [b31, b32, ..., b38],
    #      [b41, b42, ..., b48]]
    #
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Strassen for 2x2: 
    #   M1 = (a1+a4)(b11+b22), M2 = (a3+a4)b11, M3 = a1(b21-b22),
    #   M4 = a4(b12-b21), M5 = (a1+a2)b22, M6 = (a3-a1)(b11+b12),
    #   M7 = (a2-a4)(b21+b22)
    #
    # For 2x4 x 4x2 = 2x2, we can use 7 multiplications instead of 8.
    #
    # 2x4 x 4x8: partition B into four 4x2 blocks: B = [B1 B2 B3 B4]
    # C = [C1 C2 C3 C4] where Ci = A * Bi, each Ci is 2x2.
    # Each 2x4 x 4x2 has rank 8 (schoolbook) or 7 (Strassen).
    # Total: 4 * 7 = 28? No, that's too optimistic because the A factors are shared.
    #
    # Actually each Ci = A * Bi is independent given A and Bi, so yes,
    # we can use Strassen for each: 4 * 7 = 28 rank-1 terms.
    # But wait - that would mean rank 28 for 2x4x8, which is impossible
    # since the border rank of 2x4x8 is at least 2*4*8/... 
    #
    # The issue: Strassen's 7 terms for 2x2x2 don't directly apply to 2x4x4x2
    # because the dimensions don't match. 2x4 x 4x2 is NOT a 2x2x2 tensor.
    #
    # Let me be more careful. 2x4 x 4x2:
    # A is 2x4, B is 4x2, C is 2x2.
    # C[i,k] = sum_{j=1}^{4} A[i,j] * B[j,k]
    # This is a 2x4x2 tensor. Its schoolbook rank is 2*4*2 = 16.
    # 
    # We can partition A into two 2x2 blocks: A = [A1 A2]
    # C = A1*B1 + A2*B2 where B1, B2 are 2x2 (top and bottom halves of B)
    # Each 2x2 x 2x2 = 2x2 has rank 7 (Strassen).
    # Total: 7 + 7 = 14 for 2x4x2.
    #
    # For 2x4x8: partition B into four 4x2 blocks.
    # C = A*B1 + A*B2 + A*B3 + A*B4 (block-wise)
    # Each A*Bi is 2x4x2 with rank 14.
    # Total: 4 * 14 = 56!
    #
    # So rank 56 is achievable! Let me construct it.
    
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for block in range(4):  # 4 blocks of B (each 4x2)
        # For each block, use Strassen on the two 2x2 sub-blocks of A
        # A = [A1 A2] where A1 = A[:, 0:2], A2 = A[:, 2:4]
        # B_block = [B1; B2] where B1 = B[0:2, block*2:(block+1)*2], B2 = B[2:4, block*2:(block+1)*2]
        # C_block = A1*B1 + A2*B2
        #
        # Strassen for A1*B1 (2x2 * 2x2):
        # M1 = (A1[0,0]+A1[1,1]) * (B1[0,0]+B1[1,1])
        # M2 = (A1[1,0]+A1[1,1]) * B1[0,0]
        # M3 = A1[0,0] * (B1[1,0]-B1[1,1])
        # M4 = A1[1,1] * (B1[0,1]-B1[1,1])
        # M5 = (A1[0,0]+A1[0,1]) * B1[1,1]
        # M6 = (A1[1,0]-A1[0,0]) * (B1[0,0]+B1[0,1])
        # M7 = (A1[0,1]-A1[1,1]) * (B1[1,0]+B1[1,1])
        #
        # Similarly for A2*B2.
        
        for sub in range(2):  # sub=0: A1*B1, sub=1: A2*B2
            a_row_start = sub * 2  # 0 for A1, 2 for A2
            b_row_start = sub * 2  # 0 for B1, 2 for B2
            
            # Strassen terms for 2x2 * 2x2
            strassen_U = [
                [1, 1],  # (a00 + a11)
                [1, 1],  # (a10 + a11)
                [1, 0],  # a00
                [0, 1],  # a11
                [1, 1],  # (a00 + a01)
                [1, -1], # (a10 - a00)
                [1, -1], # (a01 - a11)
            ]
            strassen_V = [
                [1, 1],  # (b00 + b11)
                [1, 0],  # b00
                [1, -1], # (b10 - b11)
                [1, -1], # (b01 - b11)
                [0, 1],  # b11
                [1, 1],  # (b00 + b01)
                [1, 1],  # (b10 + b11)
            ]
            
            for s in range(7):
                # U: A indices (i, j) where i in {0,1}, j in {a_row_start, a_row_start+1}
                # A is 2x4, so row i, col j -> index i*4 + j
                for di in range(2):
                    U[di * m + (a_row_start + s % 2 if s < 4 else (s - 4) % 2 + 1 if s >= 4 else 0)][col] = 0  # placeholder
                # This is getting complicated. Let me do it more carefully.
                
                col += 1
    
    # Let me redo this more carefully with explicit indexing.
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for block in range(4):  # 4 blocks of 2 columns in B
        for sub in range(2):  # 2 sub-blocks of 2 rows in B (and corresponding cols in A)
            # A sub-block: rows 0-1, cols sub*2 to sub*2+1
            # B sub-block: rows sub*2 to sub*2+1, cols block*2 to block*2+1
            # C sub-block: rows 0-1, cols block*2 to block*2+1
            
            # Strassen 7 terms
            # Term 1: (A00+A11)*(B00+B11)
            U[0 * m + sub * 2][col] = 1      # A[0, sub*2]
            U[1 * m + sub * 2 + 1][col] = 1  # A[1, sub*2+1]
            V[sub * 2 * p + block * 2][col] = 1      # B[sub*2, block*2]
            V[(sub * 2 + 1) * p + block * 2 + 1][col] = 1  # B[sub*2+1, block*2+1]
            W[block * 2 * n + 0][col] = 1  # C[0, block*2]
            W[block * 2 * n + 1][col] = 1  # C[1, block*2+1]
            W[block * 2 * n + 0 + 1][col] += 1  # C[0, block*2+1]
            W[block * 2 * n + 1 + 0][col] += 1  # C[1, block*2]
            col += 1
            
            # Term 2: (A10+A11)*B00
            U[1 * m + sub * 2][col] = 1      # A[1, sub*2]
            U[1 * m + sub * 2 + 1][col] = 1  # A[1, sub*2+1]
            V[sub * 2 * p + block * 2][col] = 1  # B[sub*2, block*2]
            W[block * 2 * n + 0][col] = 1  # C[0, block*2]
            W[block * 2 * n + 1][col] = 1  # C[1, block*2]
            col += 1
            
            # Term 3: A00*(B10-B11)
            U[0 * m + sub * 2][col] = 1  # A[0, sub*2]
            V[(sub * 2 + 1) * p + block * 2][col] = 1    # B[sub*2+1, block*2]
            V[(sub * 2 + 1) * p + block * 2 + 1][col] = -1  # B[sub*2+1, block*2+1]
            W[block * 2 * n + 0][col] = 1  # C[0, block*2]
            W[block * 2 * n + 0 + 1][col] = -1  # C[0, block*2+1]
            col += 1
            
            # Term 4: A11*(B01-B11)
            U[1 * m + sub * 2 + 1][col] = 1  # A[1, sub*2+1]
            V[sub * 2 * p + block * 2 + 1][col] = 1    # B[sub*2, block*2+1]
            V[(sub * 2 + 1) * p + block * 2 + 1][col] = -1  # B[sub*2+1, block*2+1]
            W[block * 2 * n + 1][col] = 1  # C[1, block*2]
            W[block * 2 * n + 1 + 1][col] = -1  # C[1, block*2+1]
            col += 1
            
            # Term 5: (A00+A01)*B11
            U[0 * m + sub * 2][col] = 1      # A[0, sub*2]
            U[0 * m + sub * 2 + 1][col] = 1  # A[0, sub*2+1]
            V[(sub * 2 + 1) * p + block * 2 + 1][col] = 1  # B[sub*2+1, block*2+1]
            W[block * 2 * n + 0 + 1][col] = 1  # C[0, block*2+1]
            W[block * 2 * n + 1 + 1][col] = 1  # C[1, block*2+1]
            col += 1
            
            # Term 6: (A10-A00)*(B00+B01)
            U[1 * m + sub * 2][col] = 1      # A[1, sub*2]
            U[0 * m + sub * 2][col] = -1     # A[0, sub*2]
            V[sub * 2 * p + block * 2][col] = 1    # B[sub*2, block*2]
            V[sub * 2 * p + block * 2 + 1][col] = 1  # B[sub*2, block*2+1]
            W[block * 2 * n + 1][col] = 1  # C[1, block*2]
            W[block * 2 * n + 1 + 1][col] = 1  # C[1, block*2+1]
            col += 1
            
            # Term 7: (A01-A11)*(B10+B11)
            U[0 * m + sub * 2 + 1][col] = 1  # A[0, sub*2+1]
            U[1 * m + sub * 2 + 1][col] = -1 # A[1, sub*2+1]
            V[(sub * 2 + 1) * p + block * 2][col] = 1    # B[sub*2+1, block*2]
            V[(sub * 2 + 1) * p + block * 2 + 1][col] = 1  # B[sub*2+1, block*2+1]
            W[block * 2 * n + 0 + 1][col] = 1  # C[0, block*2+1]
            W[block * 2 * n + 1 + 1][col] = 1  # C[1, block*2+1]
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

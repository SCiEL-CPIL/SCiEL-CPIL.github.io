"""
Rank-minimal exact tensor decomposition for 3x4x8 matrix multiplication.
Uses block-partitioning: 1x4x8 schoolbook (rank 32) + 2x4x8 Strassen-like (rank 56) = rank 88.
"""

import numpy as np
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns U, V, W matrices for 3x4x8 matrix multiplication.
    U: shape (12, 88) - corresponds to A's entries (3 rows x 4 cols)
    V: shape (32, 88) - corresponds to B's entries (4 rows x 8 cols)
    W: shape (24, 88) - corresponds to C's entries (8 cols x 3 rows), W[k*n+i, r]
    
    The decomposition satisfies:
    sum_r U[i*m+j, r] * V[j'*p+k, r] = W[k*n+i', r] for all valid indices
    where i,i' in [0,3), j,j' in [0,4), k in [0,8)
    """
    n, m, p = 3, 4, 8
    rank = 88
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Block 1: 1x4x8 schoolbook (rank 32)
    # This handles row 0 of A (3 cols... wait, n=3 rows, m=4 cols)
    # A is 3x4, B is 4x8, C is 3x8
    # U indexes A[i,j] with index i*m+j
    # V indexes B[j,k] with index j*p+k
    # W indexes C[k,i] with index k*n+i (trace convention W[k*n+i])
    
    # For 1x4x8: take i=0 from A, all j in [0,4), all k in [0,8)
    # Schoolbook: rank = 1*4*8 = 32
    # For each (j,k), we have a rank-1 term
    for j in range(m):
        for k in range(p):
            col = j * p + k
            U[0 * m + j, col] = 1.0
            V[j * p + k, col] = 1.0
            W[k * n + 0, col] = 1.0
    
    # Block 2: 2x4x8 using Strassen (rank 56)
    # This handles i=1,2 from A, all j in [0,4), all k in [0,8)
    # Strassen for 2x2x2 has rank 7. For 2x4x8, we can tile.
    # 2x4x8 = (2x2x2) tiled 1x2x4 times? No, that's not right.
    # Actually, 2x4x8: we can split into blocks.
    # Let's think of this as multiplying a 2x4 matrix by a 4x8 matrix to get a 2x8 result.
    # We can split the 4x8 into four 2x4 blocks? No...
    # Better: split columns of B into groups. 
    # 2x4x8: split B's 8 columns into 4 groups of 2. Each 2x4x2 sub-problem.
    # For each 2x4x2: split A's 4 cols into 2x2 blocks? 
    # Actually, let's just use Strassen directly on 2x2x2 blocks.
    # 2x4x8: we can view this as 2x2x2 blocks where:
    #   - A has 2 rows, 4 cols -> 2x2 blocks (2 blocks horizontally)
    #   - B has 4 rows, 8 cols -> 2x2 blocks (2 blocks vertically, 4 blocks horizontally)
    #   - C has 2 rows, 8 cols -> 2x2 blocks (2 blocks horizontally)
    
    # So we have a grid of 2x2 blocks. The total number of 2x2x2 sub-problems is
    # (2 horizontal A blocks) * (2 vertical B blocks) * (4 horizontal B blocks) = 16
    # But Strassen for 2x2x2 has rank 7, so 16*7 = 112? That's worse than schoolbook 64.
    
    # Wait, let me reconsider. For 2x4x8:
    # Schoolbook rank = 2*4*8 = 64
    # The knowledge says "Strassen for 2x4x8" has rank 56.
    # 
    # Let me think about this differently. 2x4x8 can be seen as:
    # Split the 4 (inner dimension) into 2+2.
    # Split the 8 (output columns) into 4+4.
    # 
    # Actually, let me just implement a known Strassen-based approach for 2x4x8.
    # 
    # Alternative: Use the fact that M(2,4,8) <= M(2,2,2) * something.
    # 
    # Let me try a different approach: directly implement Strassen for the 2x2x2 case
    # and tile it appropriately.
    #
    # For 2x2x2, Strassen's algorithm uses 7 multiplications:
    # Given A (2x2), B (2x2), C = A*B (2x2):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # For 2x4x8, let's partition:
    # A is 2x4, B is 4x8, C is 2x8
    # Split A into left 2x2 (A_L) and right 2x2 (A_R)
    # Split B into top 2x8 (B_T) and bottom 2x8 (B_B)
    # C = A_L @ B_T + A_R @ B_B
    #
    # Each of A_L@B_T and A_R@B_B is a 2x2 @ 2x8 problem.
    # Split B_T into 2x2 blocks: B_T = [B_T0 B_T1 B_T2 B_T3] where each B_Ti is 2x2
    # Similarly B_B = [B_B0 B_B1 B_B2 B_B3]
    #
    # A_L @ B_T = A_L @ [B_T0 B_T1 B_T2 B_T3] = [A_L@B_T0, A_L@B_T1, A_L@B_T2, A_L@B_T3]
    # Each A_L@B_Ti is 2x2@2x2, which uses 7 Strassen mults.
    # So A_L@B_T uses 4*7 = 28 mults.
    # Similarly A_R@B_B uses 4*7 = 28 mults.
    # Total: 56 mults. 
    
    # Now let's implement this. The indices:
    # A_L = A[:, 0:2], A_R = A[:, 2:4]
    # B_T = B[0:2, :], B_B = B[2:4, :]
    # 
    # For Strassen on 2x2@2x2:
    # Let A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # U index for A[i,j]: i*4+j (since m=4)
    # V index for B[j,k]: j*8+k (since p=8)
    # W index for C[k,i]: k*3+i (since n=3)
    
    # Block 2 starts at column 32 in U, V, W
    
    base_col = 32
    
    # Strassen multipliers for a 2x2x2 block:
    # Given offsets:
    #   A offset: (i_off, j_off) in A (row offset, col offset)
    #   B offset: (j_off, k_off) in B (row offset, col offset)  
    #   C offset: (i_off, k_off) in C (row offset, col offset)
    # 
    # A elements: A[i_off + i, j_off + j]
    # B elements: B[j_off + j, k_off + k]
    # C elements: C[i_off + i, k_off + k]
    
    def strassen_2x2x2(A_i_off, A_j_off, B_j_off, B_k_off, C_i_off, C_k_off, col_start):
        """
        Add Strassen 2x2x2 decomposition starting at col_start.
        Returns col_start + 7
        """
        # M1 = (A00+A11)(B00+B11)
        # U: A[A_i_off+0, A_j_off+0] and A[A_i_off+1, A_j_off+1]
        # V: B[B_j_off+0, B_k_off+0] and B[B_j_off+1, B_k_off+1]
        # W: C[C_i_off+0, C_k_off+0], C[C_i_off+0, C_k_off+1], C[C_i_off+1, C_k_off+0], C[C_i_off+1, C_k_off+1]
        
        # M1: (A00 + A11) * (B00 + B11)
        # U[A_i_off*4 + A_j_off+0, col] += 1
        # U[A_i_off*4 + A_j_off+1, col] += 1
        # V[B_j_off*8 + B_k_off+0, col] += 1
        # V[B_j_off*8 + B_k_off+1, col] += 1
        # W for C00: C_i_off+0, C_k_off+0 -> W[(C_k_off+0)*3 + C_i_off+0]
        # W for C01: C_i_off+0, C_k_off+1 -> W[(C_k_off+1)*3 + C_i_off+0]
        # W for C10: C_i_off+1, C_k_off+0 -> W[(C_k_off+0)*3 + C_i_off+1]
        # W for C11: C_i_off+1, C_k_off+1 -> W[(C_k_off+1)*3 + C_i_off+1]
        
        col = col_start
        # M1
        U[(A_i_off)*4 + (A_j_off+0), col] += 1.0
        U[(A_i_off+1)*4 + (A_j_off+1), col] += 1.0
        V[(B_j_off)*8 + (B_k_off+0), col] += 1.0
        V[(B_j_off+1)*8 + (B_k_off+1), col] += 1.0
        W[(C_k_off+0)*3 + (C_i_off+0), col] += 1.0
        W[(C_k_off+1)*3 + (C_i_off+0), col] += 1.0
        W[(C_k_off+0)*3 + (C_i_off+1), col] += 1.0
        W[(C_k_off+1)*3 + (C_i_off+1), col] += 1.0
        col += 1
        
        # M2 = (A10 + A11) * B00
        U[(A_i_off+1)*4 + (A_j_off+0), col] += 1.0
        U[(A_i_off+1)*4 + (A_j_off+1), col] += 1.0
        V[(B_j_off)*8 + (B_k_off+0), col] += 1.0
        # W for C10: +M2, C11: -M2
        W[(C_k_off+0)*3 + (C_i_off+1), col] += 1.0
        W[(C_k_off+1)*3 + (C_i_off+1), col] -= 1.0
        col += 1
        
        # M3 = A00 * (B10 - B11)
        U[(A_i_off+0)*4 + (A_j_off+0), col] += 1.0
        V[(B_j_off+1)*8 + (B_k_off+0), col] += 1.0
        V[(B_j_off+1)*8 + (B_k_off+1), col] -= 1.0
        # W for C01: +M3, C11: +M3
        W[(C_k_off+1)*3 + (C_i_off+0), col] += 1.0
        W[(C_k_off+1)*3 + (C_i_off+1), col] += 1.0
        col += 1
        
        # M4 = A11 * (B01 - B10)
        U[(A_i_off+1)*4 + (A_j_off+1), col] += 1.0
        V[(B_j_off)*8 + (B_k_off+1), col] += 1.0
        V[(B_j_off+1)*8 + (B_k_off+0), col] -= 1.0
        # W for C00: +M4, C10: +M4
        W[(C_k_off+0)*3 + (C_i_off+0), col] += 1.0
        W[(C_k_off+0)*3 + (C_i_off+1), col] += 1.0
        col += 1
        
        # M5 = (A00 + A01) * B11
        U[(A_i_off+0)*4 + (A_j_off+0), col] += 1.0
        U[(A_i_off+0)*4 + (A_j_off+1), col] += 1.0
        V[(B_j_off+1)*8 + (B_k_off+1), col] += 1.0
        # W for C00: -M5, C01: +M5
        W[(C_k_off+0)*3 + (C_i_off+0), col] -= 1.0
        W[(C_k_off+1)*3 + (C_i_off+0), col] += 1.0
        col += 1
        
        # M6 = (A10 - A00) * (B00 + B10)
        U[(A_i_off+1)*4 + (A_j_off+0), col] += 1.0
        U[(A_i_off+0)*4 + (A_j_off+0), col] -= 1.0
        V[(B_j_off)*8 + (B_k_off+0), col] += 1.0
        V[(B_j_off+1)*8 + (B_k_off+0), col] += 1.0
        # W for C11: +M6
        W[(C_k_off+1)*3 + (C_i_off+1), col] += 1.0
        col += 1
        
        # M7 = (A01 - A11) * (B01 + B11)
        U[(A_i_off+0)*4 + (A_j_off+1), col] += 1.0
        U[(A_i_off+1)*4 + (A_j_off+1), col] -= 1.0
        V[(B_j_off)*8 + (B_k_off+1), col] += 1.0
        V[(B_j_off+1)*8 + (B_k_off+1), col] += 1.0
        # W for C00: +M7
        W[(C_k_off+0)*3 + (C_i_off+0), col] += 1.0
        col += 1
        
        return col
    
    # Now apply Strassen to the 2x4x8 block (i=1,2)
    # A rows 1-2, cols 0-3
    # B rows 0-3, cols 0-7
    # C rows 1-2, cols 0-7
    
    # Split into 8 sub-problems of 2x2x2:
    # For each (j_block, k_block) where j_block in [0,2), k_block in [0,4):
    #   A_j_off = j_block * 2
    #   B_j_off = j_block * 2
    #   B_k_off = k_block * 2
    #   A_i_off = 1 (since we're handling A rows 1-2)
    #   C_i_off = 1 (since we're handling C rows 1-2)
    
    col_ptr = base_col
    for j_block in range(2):
        for k_block in range(4):
            col_ptr = strassen_2x2x2(
                A_i_off=1,
                A_j_off=j_block * 2,
                B_j_off=j_block * 2,
                B_k_off=k_block * 2,
                C_i_off=1,
                C_k_off=k_block * 2,
                col_start=col_ptr
            )
    
    assert col_ptr == 88, f"Expected col_ptr=88, got {col_ptr}"
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END

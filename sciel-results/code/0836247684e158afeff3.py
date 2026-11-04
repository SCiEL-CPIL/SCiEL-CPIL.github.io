"""Program for 4x4x6 matrix multiplication tensor decomposition."""
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = 4, 4, 6
    
    # Strassen rank-7 decomposition for 2x2x2
    # A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]]
    # C = A*B, C[i,j] = sum_k A[i,k]B[k,j]
    
    # The 7 multiplications:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U vectors for A (flattened as [a00,a01,a10,a11]):
    U_2x2 = [
        [1, 0, 0, 1],  # M1: A00+A11
        [0, 0, 1, 1],  # M2: A10+A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00+A01
        [1, 0, -1, 0], # M6: A10-A00
        [0, 1, 0, -1], # M7: A01-A11
    ]
    
    # V vectors for B (flattened as [b00,b01,b10,b11]):
    V_2x2 = [
        [1, 0, 0, 1],  # M1: B00+B11
        [1, 0, 0, 0],  # M2: B00
        [0, 0, 1, -1], # M3: B10-B00
        [0, 1, 0, -1], # M4: B01-B11
        [0, 0, 0, 1],  # M5: B11
        [1, 1, 0, 0],  # M6: B00+B01
        [0, 0, 1, 1],  # M7: B10+B11
    ]
    
    # W vectors for C (flattened as [c00,c01,c10,c11]):
    # C00 = M1 - M5 + M4 + M7
    # C10 = M2 + M6
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    W_2x2 = [
        [1, 0, 0, 1],  # M1: C00+=1, C11+=1
        [0, 0, 1, -1], # M2: C10+=1, C11-=1
        [0, 1, 0, 1],  # M3: C01+=1, C11+=1
        [1, 0, 0, 0],  # M4: C00+=1
        [-1, 1, 0, 0], # M5: C00-=1, C01+=1
        [0, 0, 1, 1],  # M6: C10+=1, C11+=1
        [1, 0, 0, 0],  # M7: C00+=1
    ]
    
    # Now construct the 4x4x6 decomposition
    # A is 4x4, B is 4x6, C is 4x6
    # We can view this as block multiplication where:
    # A has 2x2 blocks (2 block rows, 2 block cols of 2x2 each)
    # B has 2x3 blocks (2 block rows, 3 block cols of 2x2 each)
    # C has 2x3 blocks (2 block rows, 3 block cols of 2x2 each)
    
    # For each output block C[i_block, k_block], we compute:
    # C[i_block, k_block] = sum_{j_block=0}^{1} A[i_block, j_block] * B[j_block, k_block]
    
    # This is a 2x2 matrix multiplication over the ring of 2x2 matrices
    # We can decompose this using Strassen's algorithm on the 2x2 block structure
    
    # The block-level multiplication is 2x2x2 (2 block rows, 2 block cols for A, 
    # 2 block rows, 3 block cols for B, 2 block rows, 3 block cols for C)
    # For each (i_block, k_block), we have a 2x2x2 multiplication
    
    # Total number of independent 2x2x2 multiplications: 2 * 3 = 6
    # Each has rank 7, so total rank = 6 * 7 = 42
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 16 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 24 x 42
    
    col = 0
    for i_block in range(2):
        for k_block in range(3):
            for s in range(7):
                # For Strassen term s of the (i_block, k_block) output block
                # We need to combine A[i_block, j_block] and B[j_block, k_block] for j_block in {0,1}
                
                # The Strassen decomposition at the block level:
                # Let A_blocks = [[A00, A01], [A10, A11]] where Aij is 2x2
                # Let B_blocks = [[B00, B01], [B10, B11]] where Bij is 2x2
                # But B has 3 column blocks, so for output (i_block, k_block), 
                # we only use B[:, k_block] which is 2x1 block column
                
                # Actually, let's think of it differently.
                # For fixed i_block and k_block, we compute:
                # C[i_block, k_block] = A[i_block, 0] * B[0, k_block] + A[i_block, 1] * B[1, k_block]
                
                # This is a 1x2 * 2x1 matrix multiplication at the block level
                # Where each "scalar" is a 2x2 matrix
                
                # We can use Strassen on this 2x2x2 structure if we pad appropriately
                # Or we can just use the schoolbook for the block-level 1x2*2x1, which has rank 2
                # Each of those 2 terms is a 2x2*2x2 multiplication, rank 7 each
                # Total rank per (i_block, k_block) = 2 * 7 = 14
                # Total rank = 6 * 14 = 84, which is worse than 96? No, 84 < 96.
                
                # Wait, let me reconsider. The block-level multiplication is:
                # C_block = A_block_row * B_block_col
                # where A_block_row is 1x2 (two 2x2 blocks) and B_block_col is 2x1 (two 2x2 blocks)
                # This is a 2x2 matrix multiplication where the "scalars" are 2x2 matrices
                # The rank of this tensor is at most 2 * 7 = 14 (using schoolbook for the 2x2 outer product)
                
                # But we can do better by using Strassen on the entire structure.
                # Let's just use the simple approach: for each (i_block, k_block), 
                # compute the two terms separately using Strassen.
                
                for j_block in range(2):
                    # Term: A[i_block, j_block] * B[j_block, k_block]
                    # A[i_block, j_block] is at rows [i_block*2, i_block*2+1], cols [j_block*2, j_block*2+1]
                    # B[j_block, k_block] is at rows [j_block*2, j_block*2+1], cols [k_block*2, k_block*2+1]
                    # C[i_block, k_block] is at rows [i_block*2, i_block*2+1], cols [k_block*2, k_block*2+1]
                    
                    for local_s in range(7):
                        # U: extract from A[i_block, j_block]
                        for a_local in range(4):
                            # a_local: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
                            a_row = i_block * 2 + (a_local // 2)
                            a_col = j_block * 2 + (a_local % 2)
                            U[a_row * m + a_col][col] += U_2x2[local_s][a_local]
                        
                        # V: extract from B[j_block, k_block]
                        for b_local in range(4):
                            b_row = j_block * 2 + (b_local // 2)
                            b_col = k_block * 2 + (b_local % 2)
                            V[b_row * p + b_col][col] += V_2x2[local_s][b_local]
                        
                        # W: add to C[i_block, k_block]
                        for c_local in range(4):
                            c_row = i_block * 2 + (c_local // 2)
                            c_col = k_block * 2 + (c_local % 2)
                            # Trace convention: W[k*n+i] where k is col, i is row
                            W[c_col * n + c_row][col] += W_2x2[local_s][c_local]
                        
                        col += 1
    
    return {"U": U, "V": V, "W": W}

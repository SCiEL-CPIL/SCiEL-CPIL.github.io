import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Total Rank: 49 (Core 4x4x4 Strassen) + 32 (Core 4x4x2 Schoolbook) + 24 (Strip 4x1x6 Schoolbook) = 105
    total_rank = 105
    
    U = [[0] * total_rank for _ in range(n * m)] # 20 x 105
    V = [[0] * total_rank for _ in range(m * p)] # 30 x 105
    W = [[0] * total_rank for _ in range(p * n)] # 24 x 105
    
    col_idx = 0
    
    # --- Part 1: Core 4x4x4 Strassen ---
    # A_part: A[:, 0:4] (4x4)
    # B_part: B[0:4, 0:4] (4x4)
    # C_part: C[:, 0:4] (4x4)
    
    # Indices in flattened U (A is 4x5)
    # A[i, j] -> i*5 + j
    # For j in 0..3
    def a_idx(i, j):
        return i * 5 + j
        
    # Indices in flattened V (B is 5x6)
    # B[j, k] -> j*6 + k
    # For j in 0..3, k in 0..3
    def b_idx(j, k):
        return j * 6 + k
        
    # Indices in flattened W (C is 4x6)
    # C[i, k] -> i*6 + k
    # For i in 0..3, k in 0..3
    def c_idx(i, k):
        return i * 6 + k

    # Strassen 4x4x4 decomposition
    # We use the 7 Strassen products for 2x2 blocks, each computed with 7 scalar multiplications (2x2 Strassen)
    # Or simply use the known rank 49 decomposition.
    # Here we construct it via block-wise Strassen.
    
    # 2x2 Strassen factors
    U_2 = [
        [1, 0, 1, 0, 1, -1, 0], 
        [0, 0, 0, 0, 1, 0, 1],  
        [1, 1, 0, 0, 0, 1, 0],  
        [1, 1, 0, 1, 0, 0, -1]  
    ]
    V_2 = [
        [1, 1, 0, 0, 0, 1, 0], 
        [1, 0, 1, 0, 0, 1, 0], 
        [1, 0, 0, 1, 0, 0, 1], 
        [1, 0, -1, -1, 1, 0, 1]
    ]
    W_2 = [
        [1, 0, 0, 1, -1, 0, 1], 
        [0, 0, 1, 0, 1, 0, 0],  
        [0, 1, 0, 1, 0, 0, 0],  
        [1, -1, 1, 0, 0, 1, 0]  
    ]
    
    # Super products for 4x4 (using 2x2 blocks)
    # Blocks: 0(TL), 1(TR), 2(BL), 3(BR)
    super_products = [
        # P1: (A00+A11)(B00+B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # P2: (A10+A11)B00
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # P3: A00(B01-B11)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # P4: A11(B10-B00)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # P5: (A00+A01)B11
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # P6: (A10-A00)(B00+B01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # P7: (A01-A11)(B10+B11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    for sp_idx, (A_blocks, B_blocks, C_blocks) in enumerate(super_products):
        for k in range(7):
            # For A
            for i in range(4):
                for j in range(4):
                    # Determine block and local index
                    ba = (i // 2) * 2 + (j // 2)
                    la = (i % 2) * 2 + (j % 2)
                    if ba in A_blocks:
                        U[a_idx(i, j)][col_idx] += A_blocks[ba] * U_2[la][k]
                        
            # For B
            for j in range(4):
                for k_b in range(4):
                    bb = (j // 2) * 2 + (k_b // 2)
                    lb = (j % 2) * 2 + (k_b % 2)
                    if bb in B_blocks:
                        V[b_idx(j, k_b)][col_idx] += B_blocks[bb] * V_2[lb][k]
                        
            # For C
            for i in range(4):
                for k_c in range(4):
                    bc = (i // 2) * 2 + (k_c // 2)
                    lc = (i % 2) * 2 + (k_c % 2)
                    if bc in C_blocks:
                        W[c_idx(i, k_c)][col_idx] += C_blocks[bc] * W_2[lc][k]
                        
            col_idx += 1
            
    # --- Part 2: Core 4x4x2 Schoolbook ---
    # A_part: A[:, 0:4]
    # B_part: B[0:4, 4:6]
    # C_part: C[:, 4:6]
    
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                U[a_idx(i, j)][col_idx] = 1
                V[b_idx(j, k)][col_idx] = 1
                W[c_idx(i, k)][col_idx] = 1
                col_idx += 1
                
    # --- Part 3: Strip 4x1x6 Schoolbook ---
    # A_part: A[:, 4]
    # B_part: B[4, :]
    # C_part: C[:, :]
    
    for i in range(4):
        for k in range(6):
            U[a_idx(i, 4)][col_idx] = 1
            V[b_idx(4, k)][col_idx] = 1
            W[c_idx(i, k)][col_idx] = 1
            col_idx += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

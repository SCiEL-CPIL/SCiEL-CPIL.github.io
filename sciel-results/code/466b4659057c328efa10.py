import numpy as np
import time

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 8
    
    # Target rank: 98 (2 * 49)
    # We use recursive Strassen for 4x4x4, applied twice for the two 4x4 halves of B.
    
    # Base case: Rank-7 decomposition for 2x2x2
    # U, V, W are 4x7 matrices
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Strassen terms for 2x2
    # P0 = (A00+A11)(B00+B11) -> C00+=1, C11+=1
    U2[0,0]=1; U2[3,0]=1
    V2[0,0]=1; V2[3,0]=1
    W2[0,0]=1; W2[3,0]=1
    
    # P1 = (A10+A11)B00 -> C10+=1, C11-=1
    U2[2,1]=1; U2[3,1]=1
    V2[0,1]=1
    W2[2,1]=1; W2[3,1]=-1
    
    # P2 = A00(B01-B11) -> C01+=1, C11+=1
    U2[0,2]=1
    V2[1,2]=1; V2[3,2]=-1
    W2[1,2]=1; W2[3,2]=1
    
    # P3 = A11(B10-B00) -> C00+=1, C10+=1
    U2[3,3]=1
    V2[2,3]=1; V2[0,3]=-1
    W2[0,3]=1; W2[2,3]=1
    
    # P4 = (A00+A01)B11 -> C00-=1, C01+=1
    U2[0,4]=1; U2[1,4]=1
    V2[3,4]=1
    W2[0,4]=-1; W2[1,4]=1
    
    # P5 = (A10-A00)(B00+B01) -> C11+=1
    U2[2,5]=1; U2[0,5]=-1
    V2[0,5]=1; V2[1,5]=1
    W2[3,5]=1
    
    # P6 = (A01-A11)(B10+B11) -> C00+=1
    U2[1,6]=1; U2[3,6]=-1
    V2[2,6]=1; V2[3,6]=1
    W2[0,6]=1

    # Function to build 4x4x4 decomposition from 2x2x2 blocks
    # Returns U, V, W of shape (16, 49), (16, 49), (16, 49)
    # Note: Indices for 4x4 matrices:
    # A[i,j] -> index i*4+j
    # B[i,j] -> index i*4+j
    # C[i,j] -> index j*4+i (W convention: k*n+i, here k=j, n=4, i=i -> j*4+i)
    
    def build_4x4x4():
        U4 = np.zeros((16, 49))
        V4 = np.zeros((16, 49))
        W4 = np.zeros((16, 49))
        
        # The 7 Strassen steps for 4x4
        # Each step involves linear combinations of 2x2 blocks.
        # We iterate over the 7 steps. For each step, we have a linear combination of A-blocks,
        # a linear combination of B-blocks, and the result contributes to C-blocks.
        # However, the standard Strassen recursion for 4x4 uses 7 multiplications of 2x2 matrices.
        # Each 2x2 multiplication is decomposed into 7 rank-1 terms.
        # So we have 7 * 7 = 49 terms.
        
        # Define the 7 Strassen combinations for 4x4 in terms of 2x2 blocks.
        # A blocks: A00, A01, A10, A11 (each 2x2)
        # B blocks: B00, B01, B10, B11 (each 2x2)
        # C blocks: C00, C01, C10, C11 (each 2x2)
        
        # Step 1: (A00+A11)(B00+B11) -> contributes to C00, C11
        # Step 2: (A10+A11)B00 -> contributes to C10, C11
        # Step 3: A00(B01-B11) -> contributes to C01, C11
        # Step 4: A11(B10-B00) -> contributes to C00, C10
        # Step 5: (A00+A01)B11 -> contributes to C00, C01
        # Step 6: (A10-A00)(B00+B01) -> contributes to C11
        # Step 7: (A01-A11)(B10+B11) -> contributes to C00
        
        # We need to map the local 2x2 indices of the sub-multiplications to the global 4x4 indices.
        # Let's define the mapping for each of the 7 steps.
        
        # Helper to add a 2x2x2 decomposition scaled and shifted into the 4x4x4 arrays
        def add_block(A_blocks, B_blocks, C_blocks, scale_A, scale_B, scale_C, offset_col):
            # A_blocks: list of (block_row, block_col) for A
            # B_blocks: list of (block_row, block_col) for B
            # C_blocks: list of (block_row, block_col) for C
            # scale_A, scale_B, scale_C: coefficients for each block in the list
            
            # This is getting complicated to generalize. Let's just hardcode the 7 steps.
            pass
            
        # Let's do it explicitly for each of the 7 Strassen steps.
        # For each step s in 0..6:
        #   We have a linear combination of A-blocks: sum_{ba} coeff_A[ba] * A_ba
        #   We have a linear combination of B-blocks: sum_{bb} coeff_B[bb] * B_bb
        #   The product is a 2x2 matrix P.
        #   P contributes to C-blocks: sum_{bc} coeff_C[bc] * C_bc += P
        
        # We decompose the 2x2 product P = A_sub * B_sub into 7 rank-1 terms using U2, V2, W2.
        # A_sub is a 2x2 matrix formed by linear combination of A-blocks.
        # B_sub is a 2x2 matrix formed by linear combination of B-blocks.
        
        # For a specific Strassen step, e.g., Step 1:
        # A_sub = A00 + A11
        # B_sub = B00 + B11
        # C00 += A_sub * B_sub
        # C11 += A_sub * B_sub
        
        # We need to expand "A_sub * B_sub" into rank-1 terms.
        # A_sub is not a single 2x2 matrix in memory, it's a sum.
        # But the tensor decomposition is linear.
        # (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
        # This expansion leads to 4 terms, not 1. This is NOT how Strassen recursion works in tensor decomposition.
        
        # Correction: In tensor decomposition, we cannot just expand the product of sums.
        # The Strassen algorithm computes 7 products of 2x2 matrices.
        # Each of these 7 products is a full 2x2x2 multiplication tensor.
        # So, for Step 1, we are computing the tensor for (A00+A11)(B00+B11).
        # This is NOT a rank-1 tensor. It is the sum of 4 rank-1 tensors if expanded naively.
        # BUT, the Strassen algorithm replaces the 8 naive products with 7 "mixed" products.
        # The "mixed" product (A00+A11)(B00+B11) is treated as a single 2x2x2 multiplication instance in the recursion.
        # So, we decompose the 2x2x2 tensor $T_{i,j,k} = \delta_{...}$ for the sub-problem.
        # The inputs to this sub-problem are $A' = A00+A11$ and $B' = B00+B11$.
        # The output is $C' = A'B'$.
        # The decomposition of this sub-problem uses the base rank-7 decomposition U2, V2, W2.
        # The vectors U2, V2, W2 operate on the flattened 2x2 matrices.
        
        # So, for Step 1:
        # We need to create U, V, W vectors for the 4x4 case.
        # The "A-part" of the sub-problem is $A' = A00 + A11$.
        # In the global 4x4 A matrix, $A00$ is at rows 0-1, cols 0-1. $A11$ is at rows 2-3, cols 2-3.
        # The sub-problem's A' is a 2x2 matrix.
        # The base decomposition U2 has 4 entries corresponding to the 4 elements of the 2x2 A'.
        # U2[0] corresponds to A'[0,0]. In the global A, A'[0,0] = A[0,0] + A[2,2].
        # So, the global U vector for this term will have 1 at index 0*4+0 and 1 at 2*4+2.
        
        # Let's implement a helper to map the base 2x2 indices to global 4x4 indices for a given Strassen step.
        
        # Mapping definitions for the 7 steps:
        # Each step defines:
        # - A_map: dict mapping local 2x2 index (0..3) to list of (global_idx, coeff)
        # - B_map: dict mapping local 2x2 index (0..3) to list of (global_idx, coeff)
        # - C_map: dict mapping local 2x2 index (0..3) to list of (global_idx, coeff)
        
        # Step 1: (A00+A11)(B00+B11) -> C00 += P, C11 += P
        # A' = A00 + A11
        # A'[0,0] = A[0,0] + A[2,2] -> idx 0, 10
        # A'[0,1] = A[0,1] + A[2,3] -> idx 1, 11
        # A'[1,0] = A[1,0] + A[3,2] -> idx 4, 14
        # A'[1,1] = A[1,1] + A[3,3] -> idx 5, 15
        
        # B' = B00 + B11
        # B'[0,0] = B[0,0] + B[2,2] -> idx 0, 10
        # B'[0,1] = B[0,1] + B[2,3] -> idx 1, 11
        # B'[1,0] = B[1,0] + B[3,2] -> idx 4, 14
        # B'[1,1] = B[1,1] + B[3,3] -> idx 5, 15
        
        # C' contributes to C00 and C11.
        # C'[0,0] -> C[0,0] (idx 0) and C[2,2] (idx 10)
        # C'[0,1] -> C[0,1] (idx 1) and C[2,3] (idx 11)
        # C'[1,0] -> C[1,0] (idx 4) and C[3,2] (idx 14)
        # C'[1,1] -> C[1,1] (idx 5) and C[3,3] (idx 15)
        # Note: W index is j*4+i.
        # C[0,0] -> W[0*4+0] = 0
        # C[0,1] -> W[1*4+0] = 4 ? No, C[i,j] -> W[j*4+i].
        # C[0,0] (i=0,j=0) -> W[0]
        # C[0,1] (i=0,j=1) -> W[4]
        # C[1,0] (i=1,j=0) -> W[1]
        # C[1,1] (i=1,j=1) -> W[5]
        # C[2,2] (i=2,j=2) -> W[2*4+2] = 10
        # C[2,3] (i=2,j=3) -> W[3*4+2] = 14
        # C[3,2] (i=3,j=2) -> W[2*4+3] = 11
        # C[3,3] (i=3,j=3) -> W[3*4+3] = 15
        
        # Let's build the maps for all 7 steps.
        
        steps = [
            # Step 1
            {
                'A': {0: [(0,1),(10,1)], 1: [(1,1),(11,1)], 2: [(4,1),(14,1)], 3: [(5,1),(15,1)]},
                'B': {0: [(0,1),(10,1)], 1: [(1,1),(11,1)], 2: [(4,1),(14,1)], 3: [(5,1),(15,1)]},
                'C': {0: [(0,1),(10,1)], 1: [(4,1),(14,1)], 2: [(1,1),(11,1)], 3: [(5,1),(15,1)]}
                # Wait, C mapping:
                # C'[0,0] (local 0) -> C[0,0] (W 0) and C[2,2] (W 10)
                # C'[0,1] (local 1) -> C[0,1] (W 4) and C[2,3] (W 14)
                # C'[1,0] (local 2) -> C[1,0] (W 1) and C[3,2] (W 11)
                # C'[1,1] (local 3) -> C[1,1] (W 5) and C[3,3] (W 15)
            },
            # Step 2: (A10+A11)B00 -> C10 += P, C11 -= P
            # A' = A10 + A11
            # A'[0,0] = A[2,0] + A[2,2] -> idx 8, 10
            # A'[0,1] = A[2,1] + A[2,3] -> idx 9, 11
            # A'[1,0] = A[3,0] + A[3,2] -> idx 12, 14
            # A'[1,1] = A[3,1] + A[3,3] -> idx 13, 15
            # B' = B00
            # B'[0,0] = B[0,0] -> 0
            # B'[0,1] = B[0,1] -> 1
            # B'[1,0] = B[1,0] -> 4
            # B'[1,1] = B[1,1] -> 5
            # C' -> C10 += P, C11 -= P
            # C'[0,0] (loc 0) -> C[2,0] (W 8) +1, C[2,2] (W 10) -1
            # C'[0,1] (loc 1) -> C[2,1] (W 9) +1, C[2,3] (W 14) -1
            # C'[1,0] (loc 2) -> C[3,0] (W 12) +1, C[3,2] (W 11) -1
            # C'[1,1] (loc 3) -> C[3,1] (W 13) +1, C[3,3] (W 15) -1
            {
                'A': {0: [(8,1),(10,1)], 1: [(9,1),(11,1)], 2: [(12,1),(14,1)], 3: [(13,1),(15,1)]},
                'B': {0: [(0,1)], 1: [(1,1)], 2: [(4,1)], 3: [(5,1)]},
                'C': {0: [(8,1),(10,-1)], 1: [(9,1),(14,-1)], 2: [(12,1),(11,-1)], 3: [(13,1),(15,-1)]}
            },
            # Step 3: A00(B01-B11) -> C01 += P, C11 += P
            # A' = A00
            # B' = B01 - B11
            # C' -> C01 += P, C11 += P
            # C'[0,0] (loc 0) -> C[0,2] (W 8) +1, C[2,2] (W 10) +1
            # C'[0,1] (loc 1) -> C[0,3] (W 12) +1, C[2,3] (W 14) +1
            # C'[1,0] (loc 2) -> C[1,2] (W 9) +1, C[3,2] (W 11) +1
            # C'[1,1] (loc 3) -> C[1,3] (W 13) +1, C[3,3] (W 15) +1
            {
                'A': {0: [(0,1)], 1: [(1,1)], 2: [(4,1)], 3: [(5,1)]},
                'B': {0: [(8,1),(10,-1)], 1: [(9,1),(11,-1)], 2: [(12,1),(14,-1)], 3: [(13,1),(15,-1)]},
                'C': {0: [(8,1),(10,1)], 1: [(12,1),(14,1)], 2: [(9,1),(11,1)], 3: [(13,1),(15,1)]}
            },
            # Step 4: A11(B10-B00) -> C00 += P, C10 += P
            # A' = A11
            # B' = B10 - B00
            # C' -> C00 += P, C10 += P
            # C'[0,0] (loc 0) -> C[0,0] (W 0) +1, C[2,0] (W 8) +1
            # C'[0,1] (loc 1) -> C[0,1] (W 4) +1, C[2,1] (W 9) +1
            # C'[1,0] (loc 2) -> C[1,0] (W 1) +1, C[3,0] (W 12) +1
            # C'[1,1] (loc 3) -> C[1,1] (W 5) +1, C[3,1] (W 13) +1
            {
                'A': {0: [(10,1)], 1: [(11,1)], 2: [(14,1)], 3: [(15,1)]},
                'B': {0: [(8,1),(0,-1)], 1: [(9,1),(1,-1)], 2: [(12,1),(4,-1)], 3: [(13,1),(5,-1)]},
                'C': {0: [(0,1),(8,1)], 1: [(4,1),(9,1)], 2: [(1,1),(12,1)], 3: [(5,1),(13,1)]}
            },
            # Step 5: (A00+A01)B11 -> C00 -= P, C01 += P
            # A' = A00 + A01
            # B' = B11
            # C' -> C00 -= P, C01 += P
            # C'[0,0] (loc 0) -> C[0,0] (W 0) -1, C[0,2] (W 8) +1
            # C'[0,1] (loc 1) -> C[0,1] (W 4) -1, C[0,3] (W 12) +1
            # C'[1,0] (loc 2) -> C[1,0] (W 1) -1, C[1,2] (W 9) +1
            # C'[1,1] (loc 3) -> C[1,1] (W 5) -1, C[1,3] (W 13) +1
            {
                'A': {0: [(0,1),(8,1)], 1: [(1,1),(9,1)], 2: [(4,1),(12,1)], 3: [(5,1),(13,1)]},
                'B': {0: [(10,1)], 1: [(11,1)], 2: [(14,1)], 3: [(15,1)]},
                'C': {0: [(0,-1),(8,1)], 1: [(4,-1),(12,1)], 2: [(1,-1),(9,1)], 3: [(5,-1),(13,1)]}
            },
            # Step 6: (A10-A00)(B00+B01) -> C11 += P
            # A' = A10 - A00
            # B' = B00 + B01
            # C' -> C11 += P
            # C'[0,0] (loc 0) -> C[2,2] (W 10) +1
            # C'[0,1] (loc 1) -> C[2,3] (W 14) +1
            # C'[1,0] (loc 2) -> C[3,2] (W 11) +1
            # C'[1,1] (loc 3) -> C[3,3] (W 15) +1
            {
                'A': {0: [(8,1),(0,-1)], 1: [(9,1),(1,-1)], 2: [(12,1),(4,-1)], 3: [(13,1),(5,-1)]},
                'B': {0: [(0,1),(8,1)], 1: [(1,1),(9,1)], 2: [(4,1),(12,1)], 3: [(5,1),(13,1)]},
                'C': {0: [(10,1)], 1: [(14,1)], 2: [(11,1)], 3: [(15,1)]}
            },
            # Step 7: (A01-A11)(B10+B11) -> C00 += P
            # A' = A01 - A11
            # B' = B10 + B11
            # C' -> C00 += P
            # C'[0,0] (loc 0) -> C[0,0] (W 0) +1
            # C'[0,1] (loc 1) -> C[0,1] (W 4) +1
            # C'[1,0] (loc 2) -> C[1,0] (W 1) +1
            # C'[1,1] (loc 3) -> C[1,1] (W 5) +1
            {
                'A': {0: [(8,1),(10,-1)], 1: [(9,1),(11,-1)], 2: [(12,1),(14,-1)], 3: [(13,1),(15,-1)]},
                'B': {0: [(8,1),(10,1)], 1: [(9,1),(11,1)], 2: [(12,1),(14,1)], 3: [(13,1),(15,1)]},
                'C': {0: [(0,1)], 1: [(4,1)], 2: [(1,1)], 3: [(5,1)]}
            }
        ]
        
        col = 0
        for step in steps:
            # For each of the 7 base Strassen terms
            for base_term in range(7):
                # U vector
                for local_idx, coeff in [(0, U2[0,base_term]), (1, U2[1,base_term]), (2, U2[2,base_term]), (3, U2[3,base_term])]:
                    if coeff != 0:
                        for global_idx, g_coeff in step['A'][local_idx]:
                            U4[global_idx, col] += coeff * g_coeff
                            
                # V vector
                for local_idx, coeff in [(0, V2[0,base_term]), (1, V2[1,base_term]), (2, V2[2,base_term]), (3, V2[3,base_term])]:
                    if coeff != 0:
                        for global_idx, g_coeff in step['B'][local_idx]:
                            V4[global_idx, col] += coeff * g_coeff
                            
                # W vector
                for local_idx, coeff in [(0, W2[0,base_term]), (1, W2[1,base_term]), (2, W2[2,base_term]), (3, W2[3,base_term])]:
                    if coeff != 0:
                        for global_idx, g_coeff in step['C'][local_idx]:
                            W4[global_idx, col] += coeff * g_coeff
                            
                col += 1
        return U4, V4, W4

    U4, V4, W4 = build_4x4x4()
    
    # Now construct the full 4x4x8 decomposition.
    # Rank = 49 + 49 = 98
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # Left half: B cols 0-3, C cols 0-3
    # U is same for both (A is 4x4)
    # V: left half is B[0:4, 0:4] -> indices 0-15 in V (since V is m*p = 4*8 = 32, B[i,j] -> i*8+j)
    #    Wait, V index is k*p + j. k is 0..3, j is 0..7.
    #    Left half j=0..3. Indices: k*8 + j.
    # Right half: B cols 4-7, C cols 4-7
    #    V: k*8 + (j+4).
    
    # W: left half C[0:4, 0:4]. W index is j*4 + i.
    #    Left half j=0..3. Indices: j*4 + i.
    # Right half: C[0:4, 4:7]. W index is (j+4)*4 + i? No.
    # C[i, j] with j=4..7. W index = j*4 + i.
    
    # Copy U
    U[:, 0:49] = U4
    U[:, 49:98] = U4
    
    # Copy V
    # V4 has indices 0..15 corresponding to B[i,j] with i,j in 0..3.
    # Global V index for left half: i*8 + j.
    # Global V index for right half: i*8 + (j+4) = i*8 + j + 4.
    
    for i in range(4):
        for j in range(4):
            v_idx_local = i*4 + j
            v_idx_global_left = i*8 + j
            v_idx_global_right = i*8 + j + 4
            
            V[v_idx_global_left, 0:49] = V4[v_idx_local, 0:49]
            V[v_idx_global_right, 49:98] = V4[v_idx_local, 0:49]
            
    # Copy W
    # W4 has indices 0..15 corresponding to C[i,j] with i,j in 0..3.
    # Global W index for left half: j*4 + i.
    # Global W index for right half: (j+4)*4 + i = j*4 + 16 + i.
    
    for i in range(4):
        for j in range(4):
            w_idx_local = j*4 + i
            w_idx_global_left = j*4 + i
            w_idx_global_right = (j+4)*4 + i
            
            W[w_idx_global_left, 0:49] = W4[w_idx_local, 0:49]
            W[w_idx_global_right, 49:98] = W4[w_idx_local, 0:49]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

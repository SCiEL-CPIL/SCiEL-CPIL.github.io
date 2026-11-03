import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 7 decomposition for 2x2x2 using Strassen
    # U is 4x7, V is 4x7, W is 4x7
    # Indices: U[i,j], V[j,k], W[k,i]
    
    # Strassen 7 products for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Output:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))
    
    # Col 0: M1
    # A00 (0,0), A11 (1,1) -> U[0,0]=1, U[3,0]=1
    # B00 (0,0), B11 (1,1) -> V[0,0]=1, V[3,0]=1
    # W: C00 (0,0), C11 (1,1)
    # C00 gets +1, C11 gets +1
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1  # C00
    W_2[3, 0] = 1  # C11
    
    # Col 1: M2
    # A10 (1,0), A11 (1,1) -> U[2,1]=1, U[3,1]=1
    # B00 (0,0) -> V[0,1]=1
    # W: C10 (0,1? no C10 is i=1, k=0 -> idx 1*2+0=2), C11 (1,1 -> idx 3)
    # C10 gets +1
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1 # C10
    
    # Col 2: M3
    # A00 (0,0) -> U[0,2]=1
    # B01 (0,1), B11 (1,1) -> V[1,2]=1, V[3,2]=-1
    # W: C01 (0,1 -> idx 1)
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1 # C01
    
    # Col 3: M4
    # A11 (1,1) -> U[3,3]=1
    # B10 (1,0), B00 (0,0) -> V[2,3]=1, V[0,3]=-1
    # W: C00 (0,0 -> idx 0), C10 (1,0 -> idx 2)
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1 # C00
    W_2[2, 3] = 1 # C10
    
    # Col 4: M5
    # A00 (0,0), A01 (0,1) -> U[0,4]=1, U[1,4]=1
    # B11 (1,1) -> V[3,4]=1
    # W: C01 (0,1 -> idx 1), C11 (1,1 -> idx 3) -- Wait, C11 = M1 - M2 + M3 + M6. M5 is in C00 and C01?
    # Strassen: C00 = M1 + M4 - M5 + M7. C01 = M3 + M5.
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1 # C00
    W_2[1, 4] = 1 # C01
    
    # Col 5: M6
    # A10 (1,0), A00 (0,0) -> U[2,5]=1, U[0,5]=-1
    # B00 (0,0), B01 (0,1) -> V[0,5]=1, V[1,5]=1
    # W: C11 (1,1 -> idx 3)
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1 # C11
    
    # Col 6: M7
    # A01 (0,1), A11 (1,1) -> U[1,6]=1, U[3,6]=-1
    # B10 (1,0), B11 (1,1) -> V[2,6]=1, V[3,6]=1
    # W: C00 (0,0 -> idx 0), C11 (1,1 -> idx 3)?
    # C00 = ... + M7. C11 = ... (no M7).
    # Wait, standard Strassen C11 = M1 - M2 + M3 + M6.
    # Let's re-verify C00, C01, C10, C11 formulas.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1 # C00
    
    # Now construct 4x4x4 decomposition
    # We have 7 "virtual" 2x2 multiplications.
    # Each virtual multiplication uses a specific mapping of the 4x4 blocks.
    
    # Block maps for A (4x4 -> 2x2 blocks)
    # A00: rows 0,1 cols 0,1
    # A01: rows 0,1 cols 2,3
    # A10: rows 2,3 cols 0,1
    # A11: rows 2,3 cols 2,3
    
    # We need to create U_4, V_4, W_4 of size 16x49, 16x49, 16x49
    U_4 = np.zeros((16, 49))
    V_4 = np.zeros((16, 49))
    W_4 = np.zeros((16, 49))
    
    # Strassen block combinations
    # Each step t (0..6) involves specific A and B blocks.
    # For each t, we use the U_2, V_2, W_2 templates, mapping 2x2 indices to 4x4 indices.
    
    # Mapping of 2x2 local index (0,1,2,3) to global 4x4 index (0..15) for a specific block
    # A blocks:
    # Block 0 (Top-Left): 0,1,4,5 (idx = 2*r+c)
    # Block 1 (Top-Right): 2,3,6,7 (idx = 2*r + c + 2) -- wait, cols 2,3.
    # Block 2 (Bottom-Left): 8,9,12,13
    # Block 3 (Bottom-Right): 10,11,14,15
    
    # Let's define a helper to map 2x2 indices (u,v) in a block to 4x4 linear index
    def map_idx_4(r, c, block):
        # r,c in 0,1
        # block: 0 (TL), 1 (TR), 2 (BL), 3 (BR)
        if block == 0:
            return 2*r + c
        elif block == 1:
            return 2*r + (c + 2)
        elif block == 2:
            return 2*(r+2) + c
        else:
            return 2*(r+2) + (c + 2)
        
    # For A, the input is flattened 4x4.
    # For B (4x4 part), input is flattened 4x4.
    
    # Strassen steps for 4x4 using 2x2 blocks:
    # M1: A00+A11, B00+B11
    # M2: A10+A11, B00
    # M3: A00, B01-B11
    # M4: A11, B10-B00
    # M5: A00+A01, B11
    # M6: A10-A00, B00+B01
    # M7: A01-A11, B10+B11
    
    steps = [
        {'A_blocks': [(0,1), (3,1)], 'B_blocks': [(0,1), (3,1)]}, # M1
        {'A_blocks': [(2,1), (3,1)], 'B_blocks': [(0,1)]}, # M2
        {'A_blocks': [(0,1)], 'B_blocks': [(1,1), (3,-1)]}, # M3
        {'A_blocks': [(3,1)], 'B_blocks': [(2,1), (0,-1)]}, # M4
        {'A_blocks': [(0,1), (1,1)], 'B_blocks': [(3,1)]}, # M5
        {'A_blocks': [(2,1), (0,-1)], 'B_blocks': [(0,1), (1,1)]}, # M6
        {'A_blocks': [(1,1), (3,-1)], 'B_blocks': [(2,1), (3,1)]}, # M7
    ]
    
    # Wait, the "blocks" in Strassen are matrices.
    # The tensor decomposition of the *block* multiplication is U_2, V_2, W_2.
    # So for each step, we need to combine the U_2, V_2, W_2 with the block indices.
    
    # This means for a specific step, say M1:
    # We add 7 columns (from U_2) to U_4.
    # For each of the 7 sub-columns, we determine the A and B indices.
    
    # Example: Step 0 (M1).
    # Sub-step 0 (M1 of 2x2x2): Uses A00, A11 and B00, B11.
    # The U_2[0,0]=1 (A00 in 2x2 context). But here "A00" is a 2x2 block.
    # In the 4x4 context, this "A00" of the 2x2 block structure corresponds to Block 0 of the 4x4 matrix?
    # No.
    # The "2x2 matrix" in Strassen step M1 is $(A_{00} + A_{11})$.
    # This is a 2x2 matrix.
    # So we are performing a 2x2x2 multiplication on this 2x2 matrix.
    # The "elements" of this 2x2 matrix are the elements of the blocks $A_{00}, A_{11}$.
    # Wait, $A_{00}$ is 2x2.
    # So $(A_{00} + A_{11})$ is a 4x4 matrix? No.
    # $A$ is 4x4. $A_{00}$ is 2x2.
    # So $A_{00} + A_{11}$ is a 2x2 matrix of "blocks"? No, it's a 2x2 matrix of scalars if we were doing scalar Strassen.
    # Here, we are doing Matrix Strassen.
    # $A = \begin{pmatrix} A_{00} & A_{01} \\ A_{10} & A_{11} \end{pmatrix}$ with 2x2 blocks.
    # $M_1 = (A_{00} + A_{11}) (B_{00} + B_{11})$.
    # This is a product of two 2x2 matrices.
    # The result $M_1$ is a 2x2 matrix.
    # This 2x2 matrix multiplication has rank 7.
    # So we decompose THIS 2x2x2 multiplication into 7 rank-1 tensors.
    
    # So, for Step M1, we use the 7 columns of U_2, V_2, W_2.
    # The "elements" of the 2x2 matrix $A_{sum} = A_{00}+A_{11}$ are the 2x2 blocks.
    # But in our tensor decomposition, U is a vector of length 16 (4x4).
    # So U must pick the elements of $A_{00}$ and $A_{11}$ appropriately to form the "2x2 matrix" $A_{sum}$?
    # No, $A_{sum}$ is already a 2x2 matrix (in the sense of blocks).
    # But the tensor decomposition U, V, W operate on *scalars*.
    # So we must expand the "block multiplication" into scalar multiplications.
    
    # This means each "block multiplication" (rank 7) must be expanded into $7 \times 7 = 49$ scalar multiplications?
    # No, the rank of the block multiplication tensor is 7.
    # But the block multiplication tensor is $M_{2,2,2}$.
    # So we just map the indices.
    
    # Mapping:
    # The 2x2 matrix in Step M1 is formed from $A$'s Block 0 and Block 3.
    # Element (0,0) of the 2x2 matrix $A_{sum}$ corresponds to $A$'s Block 0.
    # Element (1,1) of the 2x2 matrix $A_{sum}$ corresponds to $A$'s Block 3.
    # So, when U_2 says "take element (0,0)", we take ALL elements of Block 0?
    # No. U_2[0, col] = 1 means the *scalar* at (0,0) of the 2x2 matrix.
    # But in the 4x4 case, that "scalar" is actually a 2x2 block of the original A.
    # This implies the decomposition is NOT just a simple index mapping.
    
    # Actually, the rank of $M_{4,4,4}$ is NOT 49.
    # If we use Strassen on 4x4 blocks of size 2x2, we perform 7 block multiplications of 2x2.
    # Each block multiplication is $M_{2,2,2}$ which has rank 7.
    # So the total rank is $7 \times 7 = 49$.
    
    # To implement this:
    # For each of the 7 Strassen steps (say step s):
    #   We determine the mapping from 2x2 local indices (a,b) to 4x4 global indices.
    #   For each of the 7 rank-1 terms in U_2/V_2/W_2 (say term t):
    #     We create a new column in U_4/V_4/W_4.
    #     The values are determined by the product of the local coefficient (from U_2[t]) and the block mapping.
    
    # Let's do this.
    
    col = 0
    for step_idx, step in enumerate(steps):
        # Define mapping of 2x2 local index to 4x4 block index
        # Local (0,0) -> Block ?
        # For M1: A_sum = A00 + A11.
        # Local (0,0) of A_sum is A00.
        # Local (1,1) of A_sum is A11.
        # Local (0,1) and (1,0) are 0.
        
        # We need a mapping for A and B separately for each step.
        # A_map: local_idx (0..3) -> global_block (0..3) or None
        # B_map: local_idx (0..3) -> global_block (0..3) or None
        
        if step_idx == 0: # M1
            A_map = {0: 0, 3: 3} # 0->BLK0, 3->BLK3
            B_map = {0: 0, 3: 3}
        elif step_idx == 1: # M2
            A_map = {2: 2, 3: 3}
            B_map = {0: 0}
        elif step_idx == 2: # M3
            A_map = {0: 0}
            B_map = {1: 1, 3: 3}
        elif step_idx == 3: # M4
            A_map = {3: 3}
            B_map = {2: 2, 0: 0}
        elif step_idx == 4: # M5
            A_map = {0: 0, 1: 1}
            B_map = {3: 3}
        elif step_idx == 5: # M6
            A_map = {2: 2, 0: 0}
            B_map = {0: 0, 1: 1}
        else: # M7
            A_map = {1: 1, 3: 3}
            B_map = {2: 2, 3: 3}
            
        for t in range(7):
            # Add 49 columns? No, U_2 has 7 columns.
            # We are at the "block" level.
            # The "block multiplication" is itself decomposed by U_2, V_2, W_2.
            # So we iterate t (0..6) for the inner 2x2x2 decomposition.
            
            # However, the "elements" of the block matrix are 2x2 blocks.
            # So U_2[local_idx, t] * Block_value.
            # This means we need to multiply the U_2 pattern by the block expansion.
            
            # Actually, U_4 at this step will have 7 entries per block?
            # No.
            # The rank of the block multiplication is 7.
            # So we add 7 columns to U_4.
            # For each of these 7 columns (corresponding to t), we determine the values.
            
            # Wait, U_2[local_idx, t] is the coefficient for the scalar at local_idx.
            # In the 4x4 case, that "scalar" is a 2x2 block.
            # So U_4[global_idx_in_block, col] = U_2[local_idx, t] * 1?
            # No, U_4 is a vector.
            # If U_2[0, t] = 1, and Local 0 maps to Block 0, then U_4[all indices in Block 0, col] = 1?
            # That would be a sum, not a specific element.
            
            # The tensor decomposition MUST be linear.
            # $C = \sum_r U_r(A) V_r(B) W_r$.
            # If $U_r$ picks a block, it must pick the *entire* block as a vector?
            # No, U is a vector.
            
            # This implies we cannot simply nest the decompositions without expanding the blocks into scalars.
            # The rank of $M_{4,4,4}$ via Strassen is indeed 49.
            # We just need to construct the 49 rank-1 tensors.
            
            # For each of the 7 Strassen steps, and each of the 7 inner Strassen terms:
            # We define U, V, W vectors of length 16.
            
            # Let's refine the loop.
            for t in range(7):
                # We are constructing column 'col' of U_4, V_4, W_4
                # This column corresponds to the t-th term of the inner Strassen algorithm
                # applied to the blocks defined by 'step'.
                
                # The inner Strassen term t uses local indices.
                # For each local index that is non-zero in U_2[:,t], we add the corresponding block to U_4.
                
                # U_2 is 4x7.
                for local_a in range(4):
                    coeff_u = U_2[local_a, t]
                    if coeff_u != 0:
                        block_a = A_map.get(local_a, None)
                        if block_a is not None:
                            # Add to U_4 for all indices in block_a
                            # Indices in block_a are map_idx_4(r,c,block_a)
                            for r in range(2):
                                for c in range(2):
                                    idx = map_idx_4(r, c, block_a)
                                    U_4[idx, col] += coeff_u
                
                for local_b in range(4):
                    coeff_v = V_2[local_b, t]
                    if coeff_v != 0:
                        block_b = B_map.get(local_b, None)
                        if block_b is not None:
                            for r in range(2):
                                for c in range(2):
                                    idx = map_idx_4(r, c, block_b)
                                    V_4[idx, col] += coeff_v
                                    
                for local_c in range(4):
                    coeff_w = W_2[local_c, t]
                    if coeff_w != 0:
                        # W map is same as A map for output blocks?
                        # C is 4x4.
                        # Local C indices map to C blocks.
                        block_c = A_map.get(local_c, None) # A_map structure for C?
                        # C blocks are in same positions as A.
                        if block_c is not None:
                            for r in range(2):
                                for c in range(2):
                                    idx = map_idx_4(r, c, block_c)
                                    W_4[idx, col] += coeff_w
                                    
                col += 1
                
    # Now we have U_4, V_4, W_4 for 4x4x4 (Rank 49).
    # We need 4x4x8.
    # We can just duplicate this for the second half of B.
    
    U_8 = np.zeros((16, 98))
    V_8 = np.zeros((16, 98)) # Wait, B is 4x8. V should be (4*8, Rank)?
    # Task: V is (m*p, rank). m=4, p=8. So V is 32xRank.
    # My U_4, V_4, W_4 are for 4x4x4.
    # V_4 is 16x49.
    # We need V to be 32x98.
    
    # Copy first 49 columns to V_8[0:16, 0:49] (using B's first 4 cols)
    # Copy next 49 columns to V_8[16:32, 49:98] (using B's last 4 cols)
    
    U_8[:, 0:49] = U_4
    V_8[0:16, 0:49] = V_4
    
    U_8[:, 49:98] = U_4
    V_8[16:32, 49:98] = V_4
    
    W_8 = np.zeros((16, 98))
    W_8[:, 0:49] = W_4
    W_8[:, 49:98] = W_4
    
    # Convert to lists
    U = [list(map(float, row)) for row in U_8]
    V = [list(map(float, row)) for row in V_8]
    W = [list(map(float, row)) for row in W_8]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

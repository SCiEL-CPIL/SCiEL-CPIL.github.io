"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # The current seed program returns rank 216 (schoolbook).
    # A known improvement is Block-Strassen with 2x2 blocks of size 3x3.
    # Rank = 7 (Strassen steps) * 27 (Schoolbook rank for 3x3) = 189.
    
    # To be safe and ensure a valid result quickly, we will construct the rank 189 decomposition.
    # If we have time, we could try to optimize further, but 189 is a solid structural improvement.
    
    rank = 189
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen formulas for 2x2 block matrices:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # We represent the tensor decomposition.
    # Each "multiplication" in Strassen is a 3x3 matrix multiplication.
    # We decompose each 3x3 multiplication using the schoolbook algorithm (rank 27).
    # Total rank = 7 * 27 = 189.
    
    # Indices:
    # A is 6x6. Blocks: A00 (0:3, 0:3), A01 (0:3, 3:6), A10 (3:6, 0:3), A11 (3:6, 3:6)
    # B is 6x6. Blocks: B00 (0:3, 0:3), B01 (0:3, 3:6), B10 (3:6, 0:3), B11 (3:6, 3:6)
    # C is 6x6. Blocks: C00 (0:3, 0:3), C01 (0:3, 3:6), C10 (3:6, 0:3), C11 (3:6, 3:6)
    
    # U index: i*m + j (row i, col j of A)
    # V index: j*p + k (row j, col k of B)
    # W index: k*n + i (col k, row i of C) -> Note: Task says W[k*n+i, r] corresponds to C[i,k]
    # Wait, let's verify the W convention.
    # "Output factor follows the official trace convention W[k*n+i,r]"
    # Standard tensor: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # So W[k,i,r] is the coefficient for C[i,k].
    # My W array is indexed [k*n + i, r]. This matches.
    
    def add_block_to_U(block_idx, coeff, col_offset):
        """
        block_idx: 0=A00, 1=A01, 2=A10, 3=A11
        Adds coeff * (identity mapping for that block) to U at columns col_offset..col_offset+26
        """
        # Determine row and col ranges for the block in A
        # A00: rows 0-2, cols 0-2
        # A01: rows 0-2, cols 3-5
        # A10: rows 3-5, cols 0-2
        # A11: rows 3-5, cols 3-5
        
        r_start = 0 if block_idx < 2 else 3
        c_start = 0 if block_idx % 2 == 0 else 3
        
        # For each element in the 3x3 block
        for dr in range(3):
            for dc in range(3):
                i = r_start + dr
                j = c_start + dc
                u_idx = i * m + j
                # The "schoolbook" part of the Strassen step.
                # We are effectively doing:
                # Sum over r (0..26) of U[u_idx, base+r] * V[v_idx, base+r] * W[w_idx, base+r]
                # For the schoolbook 3x3, the "r" index corresponds to the intermediate index t in 0..2?
                # No, the rank 27 comes from 3 (rows of left) * 3 (cols of left / rows of right) * 3 (cols of right).
                # Let's map the 27 sub-columns to (i_local, t_local, k_local) where:
                # i_local is row in left matrix (0..2)
                # t_local is col in left / row in right (0..2)
                # k_local is col in right matrix (0..2)
                
                # Wait, the U matrix has entries for A.
                # In the term A[i,t] * B[t,k], U contributes A[i,t].
                # So for a fixed Strassen step, and a fixed (i_local, t_local, k_local):
                # U has a 1 (or coeff) at A[r_start+i_local, c_start+t_local]
                # V has a 1 (or coeff) at B[r_start+t_local, c_start+k_local]  <-- Note: B block row is determined by t_local?
                # No, B is multiplied from the right.
                # Let's look at the blocks.
                # M = A_block * B_block
                # A_block is 3x3, B_block is 3x3.
                # (A_block * B_block)[i,k] = sum_t A_block[i,t] B_block[t,k]
                #
                # So for a specific term in the sum (fixed i_local, t_local, k_local):
                # U index: A's global index for (block_row_start + i_local, block_col_start + t_local)
                # V index: B's global index for (block_row_start + t_local, block_col_start + k_local)
                # W index: C's global index for (block_row_start + i_local, block_col_start + k_local)
                
                # BUT: The Strassen step uses LINEAR COMBINATIONS of blocks.
                # E.g. M1 = (A00 + A11) * (B00 + B11)
                # This expands to:
                # A00*B00 + A00*B11 + A11*B00 + A11*B11
                #
                # Each of these is a 3x3 * 3x3 multiplication.
                # So for M1, we have 4 separate 3x3 multiplications.
                # Each 3x3 multiplication has rank 27.
                # Total rank for M1 = 4 * 27 = 108?
                # No, that's not how Strassen rank counting works in this context.
                #
                # Let's re-evaluate the rank.
                # If we treat (A00+A11) as a single 3x3 matrix X and (B00+B11) as a single 3x3 matrix Y,
                # then M1 = X * Y.
                # The rank of the 3x3 matrix multiplication tensor is 27 (schoolbook).
                # So we need 27 columns for M1.
                #
                # How do we populate U, V, W for these 27 columns?
                # For column c (0..26) within the M1 block:
                # Let c = i_l * 9 + t_l * 3 + k_l
                # U contribution:
                #   From A00: 1 at A[0..2, 0..2] corresponding to (i_l, t_l)
                #   From A11: 1 at A[3..5, 3..5] corresponding to (i_l, t_l)
                # V contribution:
                #   From B00: 1 at B[0..2, 0..2] corresponding to (t_l, k_l)
                #   From B11: 1 at B[3..5, 3..5] corresponding to (t_l, k_l)
                # W contribution:
                #   The result M1 is added to C00, C01, C10, C11?
                #   No, M1 is a 3x3 matrix.
                #   C00 = M1 - M4 + M6 - M7
                #   C01 = M3 + M5
                #   C10 = M2 + M4
                #   C11 = M1 - M2 + M3 + M7
                #
                # So M1 contributes to C00 (top-left 3x3) and C11 (bottom-right 3x3)?
                # Wait.
                # C is 6x6.
                # C00 is 3x3. C01 is 3x3. C10 is 3x3. C11 is 3x3.
                # M1 is 3x3.
                # C00 = M1 - M4 + M6 - M7. So M1 contributes to C00.
                # C11 = M1 - M2 + M3 + M7. So M1 contributes to C11.
                # Does M1 contribute to C01 or C10? No.
                #
                # So for M1, W should have 1s in the C00 block and C11 block?
                # Let's check the dimensions.
                # M1 = (A00+A11)(B00+B11).
                # Result is 3x3.
                # C00 is top-left 3x3.
                # C11 is bottom-right 3x3.
                # So yes, W has entries in C00 and C11.
                
                # Let's refine the loop structure.
                # Total rank 189 = 7 steps * 27 sub-columns.
                
                pass # Placeholder, logic below

    # Correct Logic:
    # Steps:
    # 0: M1 = (A00+A11)*(B00+B11) -> C00 += M1, C11 += M1
    # 1: M2 = (A10+A11)*B00 -> C10 += M2, C11 -= M2
    # 2: M3 = A00*(B01-B11) -> C01 += M3, C11 += M3
    # 3: M4 = A11*(B10-B00) -> C00 -= M4, C10 += M4
    # 4: M5 = (A00+A01)*B11 -> C01 += M5
    # 5: M6 = (A10-A00)*(B00+B01) -> C00 += M6
    # 6: M7 = (A01-A11)*(B10+B11) -> C00 -= M7, C11 += M7
    
    # For each step s (0..6):
    #   Define A_blocks: list of (block_idx, sign)
    #   Define B_blocks: list of (block_idx, sign)
    #   Define C_blocks: list of (block_idx, sign)
    
    steps = [
        # s, A_blocks, B_blocks, C_blocks
        (0, [(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]), # M1
        (1, [(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]), # M2: C10+=M2, C11-=M2
        (2, [(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]), # M3: C01+=M3, C11+=M3
        (3, [(3, 1)], [(2, 1), (0, -1)], [(0, -1), (2, 1)]), # M4: C00-=M4, C10+=M4
        (4, [(0, 1), (1, 1)], [(3, 1)], [(1, 1)]), # M5: C01+=M5
        (5, [(2, 1), (0, -1)], [(0, 1), (1, 1)], [(0, 1)]), # M6: C00+=M6
        (6, [(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, -1), (3, 1)]), # M7: C00-=M7, C11+=M7
    ]
    
    for s, a_blocks, b_blocks, c_blocks in steps:
        base_col = s * 27
        for t in range(27):
            col = base_col + t
            # t maps to (i_l, t_l, k_l)
            i_l = t // 9
            rem = t % 9
            t_l = rem // 3
            k_l = rem % 3
            
            # Update U
            for b_idx, sign in a_blocks:
                # Block b_idx in A
                r_start = 0 if b_idx < 2 else 3
                c_start = 0 if b_idx % 2 == 0 else 3
                i = r_start + i_l
                j = c_start + t_l
                U[i * m + j, col] += sign
            
            # Update V
            for b_idx, sign in b_blocks:
                # Block b_idx in B
                r_start = 0 if b_idx < 2 else 3
                c_start = 0 if b_idx % 2 == 0 else 3
                i = r_start + t_l
                j = c_start + k_l
                V[i * p + j, col] += sign
            
            # Update W
            for b_idx, sign in c_blocks:
                # Block b_idx in C
                r_start = 0 if b_idx < 2 else 3
                c_start = 0 if b_idx % 2 == 0 else 3
                i = r_start + i_l
                k = c_start + k_l
                # W index is k*n + i
                W[k * n + i, col] += sign

    # Convert to lists as required by interface?
    # "factor matrices of shapes ... with integer/half-integer entries"
    # The interface expects dict with 'U', 'V', 'W'.
    # Usually these can be lists of lists or numpy arrays. The seed used lists.
    # I will return lists to be safe, or numpy arrays if accepted.
    # Seed returned lists.
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END

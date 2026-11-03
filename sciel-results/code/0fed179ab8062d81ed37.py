"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # 2x2 Strassen decomposition (rank 7)
    # U_2x2: shape (4, 7), rows indexed by (i,j)
    U_2x2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    
    V_2x2 = np.array([
        [1, 1, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 1, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    
    W_2x2 = np.array([
        [1, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, -1, 0, 0, 1, 0, -1]
    ], dtype=float)
    
    # Verify 2x2 Strassen
    # Check: for each (i,k), sum_j sum_t U[i,j,t] * V[j,k,t] * W[k,i,t] should give C[i,k] = sum_j A[i,j]*B[j,k]
    # This is equivalent to: the tensor T[i,j,k] = sum_t U[i,j,t]*V[j,k,t]*W[k,i,t] should equal the matrix mult tensor
    
    # Block Strassen for 4x4 x 4x4 (rank 49)
    # The block-level Strassen has 7 block multiplications, each of rank 7
    # U_block: shape (4, 7) for the 4x4 x 4x4 case
    # Rows of U_block correspond to (i_block, j_block) where i_block, j_block in {0,1}
    # But we need to map to the full 16-dimensional space
    
    # Actually, let me construct the 49-rank decomposition directly
    # The 7 block multiplications in Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # For each block multiplication, we use the 2x2 Strassen (rank 7)
    # Total rank = 7 * 7 = 49
    
    # U_4x4: shape (16, 49)
    # Row index: i*4 + j where i in 0..3, j in 0..3
    # Column index: t_block * 7 + t_2x2 where t_block in 0..6, t_2x2 in 0..6
    
    U_4x4 = np.zeros((16, 49))
    V_4x4 = np.zeros((16, 49))
    W_4x4 = np.zeros((16, 49))
    
    # Define the block Strassen coefficients
    # For each of the 7 block multiplications, we need:
    # - Which A blocks are involved and with what signs
    # - Which B blocks are involved and with what signs
    # - How the result contributes to each C block
    
    # Block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # A blocks: A00=0, A01=1, A10=2, A11=3
    # B blocks: B00=0, B01=1, B10=2, B11=3
    
    # M1 = (A00+A11)*(B00+B11): A blocks {0,3} with +1, B blocks {0,3} with +1
    # M2 = (A10+A11)*B00: A blocks {2,3} with +1, B blocks {0} with +1
    # M3 = A00*(B01-B11): A blocks {0} with +1, B blocks {1,+1}, {3,-1}
    # M4 = A11*(B10-B00): A blocks {3} with +1, B blocks {2,+1}, {0,-1}
    # M5 = (A00+A01)*B11: A blocks {0,1} with +1, B blocks {3} with +1
    # M6 = (A10-A00)*(B00+B01): A blocks {2,+1}, {0,-1}, B blocks {0,+1}, {1,+1}
    # M7 = (A01-A11)*(B10+B11): A blocks {1,+1}, {3,-1}, B blocks {2,+1}, {3,+1}
    
    # For each block multiplication t_block, and for each 2x2 Strassen index t_2x2:
    # U_4x4[i*4+j, t_block*7+t_2x2] = sum over A block contributions
    # V_4x4[j*4+k, t_block*7+t_2x2] = sum over B block contributions
    # W_4x4[k*4+i, t_block*7+t_2x2] = how it contributes to C
    
    # Let me define the contributions more carefully
    # For block multiplication t_block, the A part is a linear combination of A blocks
    # The B part is a linear combination of B blocks
    # The result contributes to C blocks with certain coefficients
    
    # A_block_coeffs[t_block][a_block] = coefficient
    # B_block_coeffs[t_block][b_block] = coefficient
    # C_block_coeffs[t_block][c_block] = coefficient (for each output C block)
    
    # Actually, for the tensor decomposition, we need:
    # For each (i,j,k) in the 4x4x4 tensor, T[i,j,k] = sum_t U[i,j,t]*V[j,k,t]*W[k,i,t]
    # should equal 1 if (i,j,k) corresponds to a valid matrix mult term, else 0
    # More precisely, for matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all valid (i,j,k), and the decomposition should reproduce this
    
    # Let me just construct it using the Kronecker product structure
    # The 4x4 x 4x4 tensor can be seen as a block structure where each block is 2x2 x 2x2
    
    # For the block Strassen, the 7 block multiplications each involve specific A and B blocks
    # Let me define for each block mult t_block:
    #   a_coeffs: dict mapping A block index to coefficient
    #   b_coeffs: dict mapping B block index to coefficient
    #   c_coeffs: dict mapping C block index to coefficient (this is implicit in how we reconstruct C)
    
    # Actually, for the tensor decomposition, the W matrix encodes how each rank-1 term contributes to the output
    # Let me think of it differently
    
    # For a rank-1 term with index t:
    #   U[i,j,t] * V[j,k,t] * W[k,i,t]
    # The sum over t should give the correct C[i,k] = sum_j A[i,j]*B[j,k]
    
    # For the block Strassen approach:
    # Each of the 7 block multiplications is itself a 2x2 x 2x2 multiplication with rank 7
    # So for block mult t_block and 2x2 Strassen index t_2x2:
    #   The A part: linear combination of A blocks, each multiplied by U_2x2 values
    #   The B part: linear combination of B blocks, each multiplied by V_2x2 values
    #   The W part: how it contributes to C blocks, multiplied by W_2x2 values
    
    # Let me define the block Strassen structure:
    # For t_block in 0..6:
    #   a_blocks[t_block]: list of (block_idx, coeff)
    #   b_blocks[t_block]: list of (block_idx, coeff)
    
    a_blocks = [
        [(0, 1), (3, 1)],       # M1: A00 + A11
        [(2, 1), (3, 1)],       # M2: A10 + A11
        [(0, 1)],               # M3: A00
        [(3, 1)],               # M4: A11
        [(0, 1), (1, 1)],       # M5: A00 + A01
        [(2, 1), (0, -1)],      # M6: A10 - A00
        [(1, 1), (3, -1)]       # M7: A01 - A11
    ]
    
    b_blocks = [
        [(0, 1), (3, 1)],       # M1: B00 + B11
        [(0, 1)],               # M2: B00
        [(1, 1), (3, -1)],      # M3: B01 - B11
        [(2, 1), (0, -1)],      # M4: B10 - B00
        [(3, 1)],               # M5: B11
        [(0, 1), (1, 1)],       # M6: B00 + B01
        [(2, 1), (3, 1)]        # M7: B10 + B11
    ]
    
    # For the W part, we need to know how each block multiplication contributes to C blocks
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for each t_block, c_coeffs maps C block index to coefficient
    c_coeffs = [
        {0: 1, 3: 1},           # M1 contributes to C00 and C11
        {2: 1, 3: -1},          # M2 contributes to C10 and C11
        {1: 1, 3: 1},           # M3 contributes to C01 and C11
        {0: 1, 2: 1},           # M4 contributes to C00 and C10
        {0: -1, 1: 1},          # M5 contributes to C00 and C01
        {3: 1},                 # M6 contributes to C11
        {0: 1, 1: 1}            # M7 contributes to C00 and C01
    ]
    
    # Now construct U_4x4, V_4x4, W_4x4
    for t_block in range(7):
        for t_2x2 in range(7):
            col = t_block * 7 + t_2x2
            
            # U_4x4: for each (i,j) in 0..3 x 0..3
            for i in range(4):
                for j in range(4):
                    i_block = i // 2  # 0 or 1
                    j_block = j // 2  # 0 or 1
                    i_local = i % 2   # 0 or 1
                    j_local = j % 2   # 0 or 1
                    
                    # A block index: i_block * 2 + j_block
                    a_block_idx = i_block * 2 + j_block
                    
                    # Sum over A blocks in this block multiplication
                    val = 0.0
                    for (ab, coeff) in a_blocks[t_block]:
                        if ab == a_block_idx:
                            val += coeff * U_2x2[i_local * 2 + j_local, t_2x2]
                    U_4x4[i * 4 + j, col] = val
            
            # V_4x4: for each (j,k) in 0..3 x 0..3
            for j in range(4):
                for k in range(4):
                    j_block = j // 2
                    k_block = k // 2
                    j_local = j % 2
                    k_local = k % 2
                    
                    b_block_idx = j_block * 2 + k_block
                    
                    val = 0.0
                    for (bb, coeff) in b_blocks[t_block]:
                        if bb == b_block_idx:
                            val += coeff * V_2x2[j_local * 2 + k_local, t_2x2]
                    V_4x4[j * 4 + k, col] = val
            
            # W_4x4: for each (k,i) in 0..3 x 0..3
            for k in range(4):
                for i in range(4):
                    k_block = k // 2
                    i_block = i // 2
                    k_local = k % 2
                    i_local = i % 2
                    
                    c_block_idx = k_block * 2 + i_block
                    
                    val = 0.0
                    if c_block_idx in c_coeffs[t_block]:
                        coeff = c_coeffs[t_block][c_block_idx]
                        val = coeff * W_2x2[k_local * 2 + i_local, t_2x2]
                    W_4x4[k * 4 + i, col] = val
    
    # Now extend to 4x4 x 4x9
    # Columns 0-3 of B: use the 49-rank decomposition
    # Columns 4-8 of B: use schoolbook (rank 4*4*5 = 80)
    
    total_rank = 49 + 80
    
    U_full = np.zeros((16, total_rank))
    V_full = np.zeros((36, total_rank))
    W_full = np.zeros((36, total_rank))
    
    # Copy the 4x4 x 4x4 part (columns 0-3 of B)
    # U_full[:, 0:49] = U_4x4
    U_full[:, 0:49] = U_4x4
    
    # For V_full, the first 16 rows correspond to B columns 0-3 (j in 0..3, k in 0..3)
    # V_full[j*9+k, col] for j in 0..3, k in 0..3
    for j in range(4):
        for k in range(4):
            V_full[j * 9 + k, 0:49] = V_4x4[j * 4 + k, 0:49]
    
    # W_full: for the 4x4 x 4x4 part, W_full[k*4+i, col] for k in 0..3, i in 0..3
    # But in the full problem, W_full has shape (36, total_rank) where row = k*4+i for k in 0..8, i in 0..3
    for k in range(4):
        for i in range(4):
            W_full[k * 4 + i, 0:49] = W_4x4[k * 4 + i, 0:49]
    
    # Schoolbook for columns 4-8 of B (5 columns)
    # For each (i, j, k) where i in 0..3, j in 0..3, k in 4..8:
    # U[i*4+j, col] = 1
    # V[j*9+k, col] = 1
    # W[k*4+i, col] = 1
    
    col = 49
    for i in range(4):
        for j in range(4):
            for k in range(4, 9):
                U_full[i * 4 + j, col] = 1
                V_full[j * 9 + k, col] = 1
                W_full[k * 4 + i, col] = 1
                col += 1
    
    # Convert to lists for the required format
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END


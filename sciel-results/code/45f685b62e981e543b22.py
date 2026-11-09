import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x5 matrix.
    Strategy: Split the 4x5 output into a 4x4 block (using recursive Strassen, rank 49) 
    and a 4x1 block (using schoolbook, rank 16). Total rank 65.
    """
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # We split the output dimension p=5 into p1=4 and p2=1.
    # The first 4 columns of the result come from A * B[:, :4]
    # The last 1 column of the result comes from A * B[:, 4:5]
    
    # --- Block 1: 4x4 * 4x4 using Recursive Strassen ---
    # Strassen decomposition for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M4 + M6
    
    # We apply this recursively for 4x4.
    # A 4x4 matrix is divided into four 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each is 2x2.
    # B = [[B00, B01], [B10, B11]] where each is 2x2.
    
    # Each 2x2 multiplication is done via Strassen (rank 7).
    # So the 4x4 multiplication uses 7 multiplications of 2x2 matrices.
    # Each 2x2 multiplication has rank 7.
    # Total rank = 7 * 7 = 49.
    
    # Let's define the Strassen coefficients for 2x2 matrices.
    # Indices for 2x2 matrix elements: 00->0, 01->1, 10->2, 11->3.
    # U coefficients for A's 2x2 blocks.
    # V coefficients for B's 2x2 blocks.
    # W coefficients for C's 2x2 blocks.
    
    # Strassen 2x2 terms (u, v, w)
    # u: 4 coeffs for A block, v: 4 coeffs for B block, w: 4 coeffs for C block
    S2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),  # M3
        ([0, 0, 0, 1], [0, -1, 1, 0], [1, 0, 1, 0]),  # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1])   # M7
    ]
    
    # Now we build the 4x4x4 decomposition from these 7 terms.
    # The 4x4 matrix A has 4 blocks: A00, A01, A10, A11.
    # Each block is 2x2.
    # In the global 4x4 index space (row-major 0..15):
    # A00 is rows 0-1, cols 0-1. Indices: 0,1,4,5.
    # A01 is rows 0-1, cols 2-3. Indices: 2,3,6,7.
    # A10 is rows 2-3, cols 0-1. Indices: 8,9,12,13.
    # A11 is rows 2-3, cols 2-3. Indices: 10,11,14,15.
    
    # Let's map local 2x2 indices to global 4x4 indices for A.
    # Local idx: 0->00, 1->01, 2->10, 3->11
    # Block A00 (local 0): global 0,1,4,5
    # Block A01 (local 1): global 2,3,6,7
    # Block A10 (local 2): global 8,9,12,13
    # Block A11 (local 3): global 10,11,14,15
    
    def get_A_global_indices(block_local_idx):
        # block_local_idx: 0=A00, 1=A01, 2=A10, 3=A11
        if block_local_idx == 0: return [0, 1, 4, 5]
        if block_local_idx == 1: return [2, 3, 6, 7]
        if block_local_idx == 2: return [8, 9, 12, 13]
        if block_local_idx == 3: return [10, 11, 14, 15]
        
    # Similarly for B (4x4).
    def get_B_global_indices(block_local_idx):
        if block_local_idx == 0: return [0, 1, 4, 5]
        if block_local_idx == 1: return [2, 3, 6, 7]
        if block_local_idx == 2: return [8, 9, 12, 13]
        if block_local_idx == 3: return [10, 11, 14, 15]
        
    # And for C (4x4).
    # W output convention: W[k*n+i, r]. Here n=4.
    # C is 4x4. Element C[i,k] corresponds to W[k*4+i].
    # Block C00: i in 0,1; k in 0,1. W indices: 0,1,4,5.
    # Block C01: i in 0,1; k in 2,3. W indices: 8,9,12,13.
    # Block C10: i in 2,3; k in 0,1. W indices: 2,3,6,7.
    # Block C11: i in 2,3; k in 2,3. W indices: 10,11,14,15.
    
    def get_C_global_indices(block_local_idx):
        if block_local_idx == 0: return [0, 1, 4, 5]
        if block_local_idx == 1: return [8, 9, 12, 13]
        if block_local_idx == 2: return [2, 3, 6, 7]
        if block_local_idx == 3: return [10, 11, 14, 15]

    # The 4x4 Strassen decomposition consists of 7 top-level terms.
    # Each top-level term multiplies a 4x4 "A-block" by a 4x4 "B-block".
    # Wait, no. Strassen multiplies 2x2 blocks.
    # So we have 7 combinations of A-blocks and B-blocks.
    # Each combination is a 2x2 matrix multiplication.
    # We expand each 2x2 multiplication using the S2 coefficients.
    
    # Total terms for 4x4x4: 7 (top-level) * 7 (inner 2x2) = 49 terms.
    
    U_block1 = []
    V_block1 = []
    W_block1 = []
    
    # Top-level Strassen structure for 4x4:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each "A-block" in the top-level formula is a 2x2 matrix.
    # We need to combine the local 2x2 coefficients for the 2x2 multiplication
    # with the selection of which 2x2 block of the 4x4 matrix is used.
    
    # Let's iterate through the 7 top-level terms.
    # For each top-level term, we have a sum of A-blocks and a sum of B-blocks.
    # Actually, it's simpler to think of it as 7 pairs of (A_2x2, B_2x2) products.
    # But A_2x2 and B_2x2 themselves are linear combinations of the 4 sub-blocks of the 4x4 matrix.
    
    # Let's define the top-level coefficients for A and B blocks.
    # A blocks: 0,1,2,3. B blocks: 0,1,2,3.
    top_A_coeffs = [
        {0: 1, 3: 1}, # M1: A00 + A11
        {2: 1, 3: 1}, # M2: A10 + A11
        {0: 1},       # M3: A00
        {3: 1},       # M4: A11
        {0: 1, 1: 1}, # M5: A00 + A01
        {2: 1, 0: -1},# M6: A10 - A00
        {1: 1, 3: -1} # M7: A01 - A11
    ]
    top_B_coeffs = [
        {0: 1, 3: 1}, # M1: B00 + B11
        {0: 1},       # M2: B00
        {1: 1, 3: -1},# M3: B01 - B11
        {2: 1, 0: -1},# M4: B10 - B00
        {3: 1},       # M5: B11
        {0: 1, 1: 1}, # M6: B00 + B01
        {2: 1, 3: 1}  # M7: B10 + B11
    ]
    # Output C blocks for top-level terms:
    # M1 contributes to C00 (coeff 1), C11 (coeff 1)
    # M2 contributes to C10 (coeff 1), C11 (coeff -1)
    # M3 contributes to C01 (coeff 1), C11 (coeff 1)
    # M4 contributes to C00 (coeff 1), C10 (coeff 1)
    # M5 contributes to C00 (coeff -1), C01 (coeff 1)
    # M6 contributes to C11 (coeff 1)
    # M7 contributes to C00 (coeff 1), C11 (coeff 1)
    
    top_W_coeffs = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {0: -1, 1: 1},
        {3: 1},
        {0: 1, 3: 1}
    ]
    
    # Now, for each of the 7 top-level terms, we expand the 2x2 multiplication
    # using the S2 coefficients.
    # Each top-level term t has:
    # A_t = sum_{ab} top_A_coeffs[t][ab] * A_block[ab]
    # B_t = sum_{bb} top_B_coeffs[t][bb] * B_block[bb]
    # C_t = sum_{cb} top_W_coeffs[t][cb] * C_block[cb]
    
    # The 2x2 multiplication A_t * B_t = C_t is decomposed into 7 inner terms.
    # For inner term i (0..6):
    # A_t * B_t += u_i[local_a] * v_i[local_b] * w_i[local_c]
    
    for t in range(7):
        for i in range(7):
            u_inner, v_inner, w_inner = S2[i]
            
            U_row = [0] * 16
            V_row = [0] * 16
            W_row = [0] * 16
            
            # Expand A part
            for ab in top_A_coeffs[t]:
                coeff_A = top_A_coeffs[t][ab]
                # Map local 2x2 indices to global 4x4 indices for this block
                a_indices = get_A_global_indices(ab)
                for la in range(4):
                    U_row[a_indices[la]] += coeff_A * u_inner[la]
                    
            # Expand B part
            for bb in top_B_coeffs[t]:
                coeff_B = top_B_coeffs[t][bb]
                b_indices = get_B_global_indices(bb)
                for lb in range(4):
                    V_row[b_indices[lb]] += coeff_B * v_inner[lb]
                    
            # Expand C part
            for cb in top_W_coeffs[t]:
                coeff_C = top_W_coeffs[t][cb]
                c_indices = get_C_global_indices(cb)
                for lc in range(4):
                    W_row[c_indices[lc]] += coeff_C * w_inner[lc]
                    
            U_block1.append(U_row)
            V_block1.append(V_row)
            W_block1.append(W_row)

    # --- Block 2: 4x4 * 4x1 using Schoolbook ---
    # Rank 16.
    # A is 4x4. B is 4x1 (column vector). C is 4x1.
    # C[i] = sum_j A[i,j] B[j]
    # Tensor: sum_{i,j} A[i,j] B[j] C[i]
    # U index: i*4+j (A is 4x4, row-major)
    # V index: j (B is 4x1, so just j) -> Wait, V shape is (m*p, rank). m=4, p=1. So V has 4 rows.
    # W index: k*n+i. Here output is 4x1. p=1, n=4.
    # The output matrix C is 4x1. Element C[i,0].
    # W index convention: W[k*n+i]. k is the column index of C. k=0.
    # So W index is 0*4+i = i.
    
    U_block2 = []
    V_block2 = []
    W_block2 = []
    
    for i in range(4):
        for j in range(4):
            U_row = [0] * 16
            V_row = [0] * 4  # m*p = 4*1 = 4
            W_row = [0] * 4  # p*n = 1*4 = 4
            
            U_row[i*4 + j] = 1
            V_row[j] = 1
            W_row[i] = 1 # k=0, so index is i
            
            U_block2.append(U_row)
            V_block2.append(V_row)
            W_block2.append(W_row)

    # Combine blocks
    # U: 16 rows for block 1, 16 rows for block 2? No.
    # U represents A. A is 4x4.
    # Block 1 uses A (4x4). U rows = 16.
    # Block 2 uses A (4x4). U rows = 16.
    # So U has 16 rows. Each row is a vector of length (rank_block1 + rank_block2).
    
    rank1 = len(U_block1)
    rank2 = len(U_block2)
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(20)] # m*p = 4*5 = 20
    W = [[0] * total_rank for _ in range(20)] # p*n = 5*4 = 20
    
    # Fill U
    for r in range(16):
        for c in range(rank1):
            U[r][c] = U_block1[c][r]
        for c in range(rank2):
            U[r][rank1 + c] = U_block2[c][r]
            
    # Fill V
    # Block 1 V rows: 16 (4x4)
    # Block 2 V rows: 4 (4x1)
    # Total V rows: 20.
    # B is 4x5.
    # Columns 0-3 correspond to Block 1 (4x4).
    # Column 4 corresponds to Block 2 (4x1).
    
    for r in range(16):
        for c in range(rank1):
            V[r][c] = V_block1[c][r]
            
    for r in range(4):
        for c in range(rank2):
            V[16 + r][rank1 + c] = V_block2[c][r]
            
    # Fill W
    # Block 1 W rows: 16 (4x4 output, indices 0..15)
    # Block 2 W rows: 4 (4x1 output, indices 0..3 for the single column)
    # Total W rows: 20.
    # Output is 4x5.
    # W index: k*4+i.
    # k=0,1,2,3 correspond to Block 1 columns 0..3.
    # k=4 corresponds to Block 2 column 4.
    
    for r in range(16):
        for c in range(rank1):
            W[r][c] = W_block1[c][r]
            
    # For Block 2, W_block2 has 4 rows (indices 0..3 corresponding to i=0..3).
    # The output column is k=4.
    # W index for C[i,4] is 4*4 + i = 16 + i.
    for r in range(4):
        for c in range(rank2):
            W[16 + r][rank1 + c] = W_block2[c][r]
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
import numpy as np
import time

DIMENSIONS = [5, 5, 5]

def _get_strassen_2x2():
    """
    Returns U, V, W for 2x2 matrix multiplication using Strassen's algorithm (Rank 7).
    Indices:
    A: [[A00, A01], [A10, A11]] -> flat [A00, A01, A10, A11]
    B: [[B00, B01], [B10, B11]] -> flat [B00, B01, B10, B11]
    C: [[C00, C01], [C10, C11]] -> flat [C00, C01, C10, C11]
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1 # C21 += M2, C22 -= M2? 
    # Wait, Strassen formulas:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # Let's map M's to W correctly.
    
    # M1
    W[0, 0] = 1
    W[3, 0] = 1
    
    # M2
    W[2, 1] = 1
    W[3, 1] = -1
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    W[3, 2] = 1
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1
    W[2, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1
    W[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _get_rank49_4x4():
    """
    Construct rank 49 decomposition for 4x4 matrix multiplication using tensor product of Strassen.
    """
    U2, V2, W2 = _get_strassen_2x2()
    # Tensor product of two rank 7 decompositions gives rank 49.
    # We need to map the 4x4 indices to the 2x2 block indices.
    # A44 indices: 0..15.
    # Block 00: A00, A01, A10, A11 (indices 0,1,4,5 in 4x4? No.)
    # 4x4 matrix:
    # [[A00, A01, A02, A03],
    #  [A10, A11, A12, A13],
    #  [A20, A21, A22, A23],
    #  [A30, A31, A32, A33]]
    # Top-left 2x2 block: A00, A01, A10, A11. Indices 0,1,4,5.
    # Top-right 2x2 block: A02, A03, A12, A13. Indices 2,3,6,7.
    # Bottom-left 2x2 block: A20, A21, A30, A31. Indices 8,9,12,13.
    # Bottom-right 2x2 block: A22, A23, A32, A33. Indices 10,11,14,15.
    
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    # Mapping of 4x4 indices to 2x2 local indices
    # Local 2x2 indices: 0:00, 1:01, 2:10, 3:11
    # Block 00 (TL): 4x4 idx 0,1,4,5 -> local 0,1,2,3
    # Block 01 (TR): 4x4 idx 2,3,6,7 -> local 0,1,2,3
    # Block 10 (BL): 4x4 idx 8,9,12,13 -> local 0,1,2,3
    # Block 11 (BR): 4x4 idx 10,11,14,15 -> local 0,1,2,3
    
    # The tensor product decomposition:
    # C = A B
    # We view A, B, C as 2x2 blocks of 2x2 matrices.
    # C_{ij} = sum_{k,l} A_{ik} B_{kl}
    # This is not a simple tensor product of the scalar multiplications.
    # The rank 49 decomposition comes from applying Strassen to the block level?
    # No, Strassen is for 2x2 scalar matrices.
    # If we apply Strassen to the 2x2 block matrix, we get 7 block multiplications.
    # Each block multiplication is a 2x2 x 2x2 -> 2x2 multiplication.
    # Each of those has rank 7 (Strassen).
    # Total rank 7 * 7 = 49.
    
    # So we need to construct the block-level Strassen steps, and for each step,
    # insert the scalar-level Strassen steps for the 2x2 blocks.
    
    # Block-level Strassen:
    # M1_B = (A00+A11)(B00+B11)
    # M2_B = (A10+A11)B00
    # M3_B = A00(B01-B11)
    # M4_B = A11(B10-B00)
    # M5_B = (A00+A01)B11
    # M6_B = (A10-A00)(B00+B01)
    # M7_B = (A01-A11)(B10+B11)
    
    # C00 = M1_B + M4_B - M5_B + M7_B
    # C01 = M3_B + M5_B
    # C10 = M2_B + M4_B
    # C11 = M1_B - M2_B + M3_B + M6_B
    
    # For each block-level term, e.g., M1_B = (A00+A11)(B00+B11),
    # This is a product of two 2x2 matrices.
    # We use the scalar Strassen decomposition for this product.
    # Let U_s, V_s, W_s be the scalar Strassen matrices (4x7).
    # The contribution to the global U, V, W is:
    # For each scalar step k in 0..6:
    #   U_global[block_A_indices, k_global] = U_s[local_A_indices, k] * coeff_A
    #   V_global[block_B_indices, k_global] = V_s[local_B_indices, k] * coeff_B
    #   W_global[block_C_indices, k_global] = W_s[local_C_indices, k] * coeff_C
    
    # We need to map global indices to local indices for each block.
    # Block 00 (TL) of A: Global 0,1,4,5. Local 0,1,2,3.
    # Block 11 (BR) of A: Global 10,11,14,15. Local 0,1,2,3.
    
    # Helper to get global indices for a block
    def get_block_indices(block_r, block_c):
        # block_r, block_c in {0,1}
        # Returns list of 4 global indices
        # Row offset: block_r * 2
        # Col offset: block_c * 2
        # Indices in 4x4:
        # r0 = block_r*2, r1 = block_r*2+1
        # c0 = block_c*2, c1 = block_c*2+1
        # idx0 = r0*4 + c0
        # idx1 = r0*4 + c1
        # idx2 = r1*4 + c0
        # idx3 = r1*4 + c1
        r0 = block_r * 2
        c0 = block_c * 2
        return [r0*4 + c0, r0*4 + c1, (r0+1)*4 + c0, (r0+1)*4 + c1]
    
    # Block-level terms definition:
    # (U_blocks, V_blocks, W_blocks, coeffs)
    # U_blocks: list of (block_r, block_c, coeff)
    # V_blocks: list of (block_r, block_c, coeff)
    # W_blocks: list of (block_r, block_c, coeff)
    
    terms = [
        # M1_B
        {
            'U': [((0,0), 1), ((1,1), 1)],
            'V': [((0,0), 1), ((1,1), 1)],
            'W': [((0,0), 1), ((1,1), 1)]
        },
        # M2_B
        {
            'U': [((1,0), 1), ((1,1), 1)],
            'V': [((0,0), 1)],
            'W': [((1,0), 1), ((1,1), -1)]
        },
        # M3_B
        {
            'U': [((0,0), 1)],
            'V': [((0,1), 1), ((1,1), -1)],
            'W': [((0,1), 1), ((1,1), 1)]
        },
        # M4_B
        {
            'U': [((1,1), 1)],
            'V': [((1,0), 1), ((0,0), -1)],
            'W': [((0,0), 1), ((1,0), 1)]
        },
        # M5_B
        {
            'U': [((0,0), 1), ((0,1), 1)],
            'V': [((1,1), 1)],
            'W': [((0,0), -1), ((0,1), 1)]
        },
        # M6_B
        {
            'U': [((1,0), 1), ((0,0), -1)],
            'V': [((0,0), 1), ((0,1), 1)],
            'W': [((1,1), 1)]
        },
        # M7_B
        {
            'U': [((0,1), 1), ((1,1), -1)],
            'V': [((1,0), 1), ((1,1), 1)],
            'W': [((0,0), 1)]
        }
    ]
    
    k_global = 0
    for term in terms:
        for k_local in range(7):
            # For each scalar step in the 2x2 multiplication
            for u_block, u_coeff in term['U']:
                u_idx = get_block_indices(u_block[0], u_block[1])
                for local_idx, global_idx in zip([0,1,2,3], u_idx):
                    U[global_idx, k_global] += u_coeff * U2[local_idx, k_local]
            
            for v_block, v_coeff in term['V']:
                v_idx = get_block_indices(v_block[0], v_block[1])
                for local_idx, global_idx in zip([0,1,2,3], v_idx):
                    V[global_idx, k_global] += v_coeff * V2[local_idx, k_local]
            
            for w_block, w_coeff in term['W']:
                w_idx = get_block_indices(w_block[0], w_block[1])
                for local_idx, global_idx in zip([0,1,2,3], w_idx):
                    W[global_idx, k_global] += w_coeff * W2[local_idx, k_local]
            
            k_global += 1
            
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Rank 103 decomposition for 5x5
    # 1. 4x4 core (Rank 49)
    # 2. Fringe terms (Rank 54)
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # --- Part 1: 4x4 Core ---
    U44, V44, W44 = _get_rank49_4x4()
    
    # Map 4x4 indices to 5x5 indices
    # 4x4 indices 0..15 correspond to 5x5 indices:
    # r in 0..3, c in 0..3 -> 5x5 idx r*5 + c
    U5 = np.zeros((25, 49))
    V5 = np.zeros((25, 49))
    W5 = np.zeros((25, 49))
    
    for i in range(4):
        for j in range(4):
            idx44 = i * 4 + j
            idx55 = i * 5 + j
            U5[idx55, :] = U44[idx44, :]
            V5[idx55, :] = V44[idx44, :]
            W5[idx55, :] = W44[idx44, :]
            
    U_parts.append(U5)
    V_parts.append(V5)
    W_parts.append(W5)
    
    # --- Part 2: Fringe Terms ---
    # We need to add terms for:
    # A44 x (Rank 16)
    # vT B44 (Rank 16)
    # u yT (Rank 1)
    # u beta (Rank 1)
    # vT beta (Rank 1)
    # vT x (Rank 1)
    # alpha yT (Rank 1)
    # alpha beta (Rank 1)
    
    # Indices in 5x5:
    # A44: rows 0-3, cols 0-3. Indices 0..12, 17..19, 22..24?
    # i*5+j for i,j in 0..3.
    # u: col 4, rows 0-3. Indices 4, 9, 14, 19.
    # v: row 4, cols 0-3. Indices 20, 21, 22, 23.
    # alpha: index 24.
    
    # B44: rows 0-3, cols 0-3. Indices 0..12, 17..19, 22..24.
    # x: col 4, rows 0-3. Indices 4, 9, 14, 19.
    # y: row 4, cols 0-3. Indices 20, 21, 22, 23.
    # beta: index 24.
    
    # C44: rows 0-3, cols 0-3. Indices 0..12, 17..19, 22..24.
    # C41: rows 0-3, col 4. Indices 4, 9, 14, 19.
    # C14: row 4, cols 0-3. Indices 20, 21, 22, 23.
    # C44_scalar: index 24.
    
    fringe_rank = 16 + 16 + 1 + 1 + 1 + 1 + 1 + 1
    Uf = np.zeros((25, fringe_rank))
    Vf = np.zeros((25, fringe_rank))
    Wf = np.zeros((25, fringe_rank))
    
    col = 0
    
    # 1. A44 x (Rank 16)
    # C_i = sum_k A_ik x_k for i in 0..3
    for i in range(4):
        for k in range(4):
            # Term A_ik * x_k
            # U: A_ik index i*5+k
            # V: x_k index k*5+4
            # W: C_i index i*5+4
            Uf[i*5+k, col] = 1
            Vf[k*5+4, col] = 1
            Wf[i*5+4, col] = 1
            col += 1
            
    # 2. vT B44 (Rank 16)
    # C_j = sum_k v_k B_kj for j in 0..3
    for j in range(4):
        for k in range(4):
            # Term v_k * B_kj
            # U: v_k index 4*5+k
            # V: B_kj index k*5+j
            # W: C_j index 4*5+j
            Uf[4*5+k, col] = 1
            Vf[k*5+j, col] = 1
            Wf[4*5+j, col] = 1
            col += 1
            
    # 3. u yT (Rank 1)
    # C_ij = u_i y_j for i,j in 0..3
    # This is a rank 1 term?
    # u_i y_j is outer product.
    # In tensor decomposition, we need to represent the map (u, y) -> u y^T.
    # This is rank 1?
    # T_{ijk} = delta_{ik} delta_{jl}?
    # No, the term is u_i y_j.
    # We can use 1 scalar multiplication?
    # No, u_i and y_j are different variables.
    # The rank of the outer product tensor is 1?
    # Let's check: C_{ij} = u_i y_j.
    # If we set k=1, U_{i1}=u_i, V_{j1}=y_j, W_{11}=1?
    # Then sum_k U_{ik} V_{jk} W_{kj} = u_i y_j * 1.
    # Yes, rank 1.
    for i in range(4):
        Uf[i*5+4, col] = 1 # u_i
        for j in range(4):
            Vf[4*5+j, col] = 1 # y_j
            Wf[i*5+j, col] = 1 # C_ij
    col += 1
    
    # 4. u beta (Rank 1)
    # C_i = u_i beta for i in 0..3
    for i in range(4):
        Uf[i*5+4, col] = 1 # u_i
        Vf[24, col] = 1 # beta
        Wf[i*5+4, col] = 1 # C_i
    col += 1
    
    # 5. vT beta (Rank 1)
    # C_j = v_j beta for j in 0..3
    for j in range(4):
        Uf[4*5+j, col] = 1 # v_j
        Vf[24, col] = 1 # beta
        Wf[4*5+j, col] = 1 # C_j
    col += 1
    
    # 6. vT x (Rank 1)
    # C_44 = sum_k v_k x_k
    # This is a dot product. Rank 4?
    # C = v . x = sum v_k x_k.
    # This requires 4 scalar multiplications.
    # So rank 4, not 1.
    # Let's correct the rank count.
    # vT x is a scalar.
    # Term: v_k x_k.
    # We need 4 terms.
    # So we need to add 4 columns.
    # But we allocated rank 1 for this.
    # We need to adjust fringe_rank.
    
    # Let's restart fringe allocation with correct ranks.
    # 1. A44 x: 16
    # 2. vT B44: 16
    # 3. u yT: 1
    # 4. u beta: 1
    # 5. vT beta: 1
    # 6. vT x: 4 (dot product)
    # 7. alpha yT: 1 (vector scale)
    # 8. alpha beta: 1 (scalar)
    
    fringe_rank = 16 + 16 + 1 + 1 + 1 + 4 + 1 + 1
    Uf = np.zeros((25, fringe_rank))
    Vf = np.zeros((25, fringe_rank))
    Wf = np.zeros((25, fringe_rank))
    
    col = 0
    
    # 1. A44 x
    for i in range(4):
        for k in range(4):
            Uf[i*5+k, col] = 1
            Vf[k*5+4, col] = 1
            Wf[i*5+4, col] = 1
            col += 1
            
    # 2. vT B44
    for j in range(4):
        for k in range(4):
            Uf[4*5+k, col] = 1
            Vf[k*5+j, col] = 1
            Wf[4*5+j, col] = 1
            col += 1
            
    # 3. u yT
    for i in range(4):
        Uf[i*5+4, col] = 1
        for j in range(4):
            Vf[4*5+j, col] = 1
            Wf[i*5+j, col] = 1
    col += 1
    
    # 4. u beta
    for i in range(4):
        Uf[i*5+4, col] = 1
        Vf[24, col] = 1
        Wf[i*5+4, col] = 1
    col += 1
    
    # 5. vT beta
    for j in range(4):
        Uf[4*5+j, col] = 1
        Vf[24, col] = 1
        Wf[4*5+j, col] = 1
    col += 1
    
    # 6. vT x (Rank 4)
    for k in range(4):
        Uf[4*5+k, col] = 1 # v_k
        Vf[k*5+4, col] = 1 # x_k
        Wf[24, col] = 1 # C_44
        col += 1
        
    # 7. alpha yT (Rank 1)
    for j in range(4):
        Uf[24, col] = 1 # alpha
        Vf[4*5+j, col] = 1 # y_j
        Wf[4*5+j, col] = 1 # C_j
    col += 1
    
    # 8. alpha beta (Rank 1)
    Uf[24, col] = 1
    Vf[24, col] = 1
    Wf[24, col] = 1
    col += 1
    
    U_parts.append(Uf)
    V_parts.append(Vf)
    W_parts.append(Wf)
    
    U_total = np.hstack(U_parts)
    V_total = np.hstack(V_parts)
    W_total = np.hstack(W_parts)
    
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
# EVOLVE-BLOCK-END

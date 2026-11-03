import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a dictionary with 'U', 'V', 'W' matrices representing the tensor decomposition.
    Uses a mixed Strassen/Schoolbook decomposition for rank 81 (49 + 32).
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Part 1: 4x4x4 using Strassen (Rank 49)
    # Part 2: 4x4x2 using Schoolbook (Rank 32)
    
    # --- Helper to get Strassen 2x2x2 decomposition ---
    def get_strassen_2x2x2():
        """
        Returns U, V, W for 2x2x2 Strassen.
        U: (4, 7), V: (4, 7), W: (4, 7)
        Indices:
        U: i*2 + j (i,j in 0..1)
        V: j*2 + k (j,k in 0..1)
        W: k*2 + i (k,i in 0..1)
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A00+A11)(B00+B11)
        # A00=U[0], A11=U[3], B00=V[0], B11=V[3]
        U[0][0] += 1; U[3][0] += 1
        V[0][0] += 1; V[3][0] += 1
        W[0][0] += 1; W[3][0] += 1 # C00 += M1, C11 += M1? 
        # Wait, W maps to C.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's define the 7 terms M1..M7 explicitly in U, V, W
        # Term 0: M1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        # M1 contributes to C00 and C11
        W[0][0] = 1; W[3][0] = 1 
        
        # Term 1: M2 = (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        # M2 contributes to C10 and C11 (-M2)
        W[2][1] = 1; W[3][1] = -1
        
        # Term 2: M3 = A00(B01-B11)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        # M3 contributes to C01 and C11
        W[1][2] = 1; W[3][2] = 1
        
        # Term 3: M4 = A11(B10-B00)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        # M4 contributes to C00 and C10
        W[0][3] = 1; W[2][3] = 1
        
        # Term 4: M5 = (A00+A01)B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        # M5 contributes to C00 (-M5) and C01
        W[0][4] = -1; W[1][4] = 1
        
        # Term 5: M6 = (A10-A00)(B00+B01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        # M6 contributes to C11
        W[3][5] = 1
        
        # Term 6: M7 = (A01-A11)(B10+B11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        # M7 contributes to C00
        W[0][6] = 1
        
        return U, V, W

    # --- Construct 4x4x4 Strassen Decomposition ---
    def get_strassen_4x4x4():
        """
        Returns U, V, W for 4x4x4 Strassen.
        U: (16, 49), V: (16, 49), W: (16, 49)
        """
        U = [[0]*49 for _ in range(16)]
        V = [[0]*49 for _ in range(16)]
        W = [[0]*49 for _ in range(16)]
        
        u2, v2, w2 = get_strassen_2x2x2()
        
        # The 4x4x4 tensor is composed of 7 "blocks" of 2x2x2 tensors.
        # Each block corresponds to one of the M1..M7 terms in the 2x2 block Strassen.
        # We need to map the local 2x2 indices to the global 4x4 indices.
        
        # Global A indices: i*4 + j.
        # Block (bi, bj) of A corresponds to i in {2*bi, 2*bi+1}, j in {2*bj, 2*bj+1}.
        # Local index in block: li*2 + lj.
        # Global index: (2*bi + li)*4 + (2*bj + lj).
        
        # The 7 block products P1..P7 are:
        # P1 = (A00+A11)(B00+B11)
        # P2 = (A10+A11)B00
        # P3 = A00(B01-B11)
        # P4 = A11(B10-B00)
        # P5 = (A00+A01)B11
        # P6 = (A10-A00)(B00+B01)
        # P7 = (A01-A11)(B10+B11)
        
        # For each block product, we use the 2x2x2 Strassen decomposition.
        # We iterate over the 7 block products (bp) and the 7 scalar terms within them (st).
        
        # Coefficients for A blocks in each block product
        # A blocks: 00, 01, 10, 11
        # P1: A00(1), A11(1)
        # P2: A10(1), A11(1)
        # P3: A00(1)
        # P4: A11(1)
        # P5: A00(1), A01(1)
        # P6: A10(1), A00(-1)
        # P7: A01(1), A11(-1)
        
        a_coeffs = [
            {0: 1, 3: 1}, # P1: A00, A11
            {2: 1, 3: 1}, # P2: A10, A11
            {0: 1},       # P3: A00
            {3: 1},       # P4: A11
            {0: 1, 1: 1}, # P5: A00, A01
            {2: 1, 0: -1},# P6: A10, A00
            {1: 1, 3: -1} # P7: A01, A11
        ]
        
        # Coefficients for B blocks in each block product
        # B blocks: 00, 01, 10, 11
        # P1: B00(1), B11(1)
        # P2: B00(1)
        # P3: B01(1), B11(-1)
        # P4: B10(1), B00(-1)
        # P5: B11(1)
        # P6: B00(1), B01(1)
        # P7: B10(1), B11(1)
        
        b_coeffs = [
            {0: 1, 3: 1}, # P1: B00, B11
            {0: 1},       # P2: B00
            {1: 1, 3: -1},# P3: B01, B11
            {2: 1, 0: -1},# P4: B10, B00
            {3: 1},       # P5: B11
            {0: 1, 1: 1}, # P6: B00, B01
            {2: 1, 3: 1}  # P7: B10, B11
        ]
        
        # Coefficients for C blocks in each block product
        # C blocks: 00, 01, 10, 11
        # P1: C00(1), C11(1)
        # P2: C10(1), C11(-1)
        # P3: C01(1), C11(1)
        # P4: C00(1), C10(1)
        # P5: C00(-1), C01(1)
        # P6: C11(1)
        # P7: C00(1)
        
        c_coeffs = [
            {0: 1, 3: 1}, # P1: C00, C11
            {2: 1, 3: -1},# P2: C10, C11
            {1: 1, 3: 1}, # P3: C01, C11
            {0: 1, 2: 1}, # P4: C00, C10
            {0: -1, 1: 1},# P5: C00, C01
            {3: 1},       # P6: C11
            {0: 1}        # P7: C00
        ]
        
        for bp in range(7):
            base_col = bp * 7
            for st in range(7):
                col = base_col + st
                
                # Get local 2x2 indices for this scalar term
                # u2, v2, w2 are 4x7 matrices.
                # We need to find which local indices are non-zero for this st.
                
                for i_local in range(4):
                    if u2[i_local][st] != 0:
                        coeff_a = u2[i_local][st]
                        # Map local i to block and local index
                        li = i_local // 2
                        lj = i_local % 2
                        # This local index corresponds to A block (li, lj)?
                        # No, u2 is for 2x2 A. i_local is index in 2x2 A.
                        # So A block is (li, lj) is incorrect.
                        # In 2x2 A, index i_local is just the element.
                        # But here we are decomposing the 4x4 A into 2x2 blocks.
                        # The "local" 2x2 A in the Strassen formula for 4x4 is actually a block of the 4x4 A.
                        # Wait. The 4x4 Strassen uses 2x2 blocks.
                        # The "M" terms are products of these 2x2 blocks.
                        # So u2[i_local] refers to the element of the 2x2 block.
                        # Which 2x2 block? It depends on the block product P_bp.
                        
                        # This approach is getting complicated to map manually.
                        # Let's use a simpler method: 
                        # The 4x4x4 Strassen decomposition is the tensor product of the 2x2x2 Strassen structure with itself?
                        # No, it's recursive.
                        
                        pass # Skip complex mapping, use direct construction below

        # Alternative: Direct construction of 49 terms for 4x4x4
        # We can generate the 49 terms by iterating over the 7 block products and 7 scalar terms.
        # For each (bp, st), we determine the global indices.
        
        # Let's refine the mapping.
        # The 2x2x2 Strassen decomposition (u2, v2, w2) gives the coefficients for a single 2x2x2 multiplication.
        # In the 4x4 case, we have 7 such multiplications (P1..P7).
        # Each Pk is a product of two 2x2 matrices (blocks of A and B).
        # So for each bp in 0..6:
        #   We have a 2x2 matrix A_bp and B_bp.
        #   We decompose A_bp * B_bp using u2, v2, w2.
        #   This gives 7 scalar terms.
        #   Total 49 terms.
        
        # We need to know which 2x2 block of A and B is used in P_bp.
        # From a_coeffs and b_coeffs above:
        # P1 uses A00, A11. So A_bp is not a single block, it's a sum.
        # This means we cannot just reuse u2 directly for P1 because u2 assumes specific A elements.
        # We must expand the linear combinations.
        
        # Let's do it properly.
        # For each bp, we have linear combinations of A blocks and B blocks.
        # A_bp = sum_{ab} a_coeff[ab] * A_block[ab]
        # B_bp = sum_{bb} b_coeff[bb] * B_block[bb]
        # P_bp = A_bp * B_bp = sum_{ab, bb} a_coeff[ab] b_coeff[bb] (A_block[ab] * B_block[bb])
        
        # Each A_block[ab] * B_block[bb] is a 2x2x2 tensor.
        # We can decompose it using u2, v2, w2.
        
        # So for each bp:
        #   For each ab in a_coeffs[bp]:
        #     For each bb in b_coeffs[bp]:
        #       coeff = a_coeffs[bp][ab] * b_coeffs[bp][bb]
        #       We add coeff * (decomposition of A_block[ab] * B_block[bb]) to the global U, V, W.
        
        # But wait, if we do this, we are summing 2x2x2 decompositions.
        # The number of terms will be sum_{bp} (|a_coeffs[bp]| * |b_coeffs[bp]| * 7).
        # P1: 2 * 2 * 7 = 28
        # P2: 2 * 1 * 7 = 14
        # P3: 1 * 2 * 7 = 14
        # P4: 1 * 2 * 7 = 14
        # P5: 2 * 1 * 7 = 14
        # P6: 2 * 2 * 7 = 28
        # P7: 2 * 2 * 7 = 28
        # Total = 140 terms. This is worse than 49!
        
        # The standard Strassen 4x4x4 rank 49 relies on the fact that the 7 products M1..M7 are computed *once* and then combined.
        # My expansion above computes A00*B00, A00*B11, A11*B00, A11*B11 separately for P1.
        # But Strassen computes (A00+A11)(B00+B11) as a single 2x2 multiplication.
        # So we should treat A_bp and B_bp as the *inputs* to a 2x2 multiplication.
        # The 2x2 multiplication has rank 7.
        # So for each bp, we just need to map the 7 terms of the 2x2 Strassen decomposition to the global indices,
        # with the coefficients from a_coeffs and b_coeffs applied to the *blocks*.
        
        # How to map?
        # The 2x2 Strassen decomposition u2[i][st] gives the coefficient of A_element_i in the st-th term.
        # In the 4x4 case, A_element_i of the "virtual" 2x2 matrix A_bp is actually a linear combination of global A blocks.
        # Specifically, A_bp[li, lj] = sum_{global_i, global_j in block} ...
        # No, A_bp is the 2x2 matrix formed by the blocks.
        # A_bp[0,0] is A_block_00.
        # A_bp[0,1] is A_block_01.
        # etc.
        
        # So, for a given bp and st:
        # The term is sum_{li, lj} u2[li*2+lj][st] * A_bp[li, lj] * B_bp[km, ln] ...
        
        # We can just iterate over the 4 elements of the 2x2 A_bp and 4 elements of 2x2 B_bp.
        
        for bp in range(7):
            base_col = bp * 7
            for st in range(7):
                col = base_col + st
                
                # Accumulate contributions to U, V, W
                # U contributions come from A_bp elements
                # V contributions come from B_bp elements
                # W contributions come from C_bp elements
                
                # A_bp elements: 00, 01, 10, 11
                # For each local element (li, lj) in A_bp:
                #   coeff_a = u2[li*2+lj][st]
                #   This local element corresponds to global block (li, lj)
                #   We add coeff_a * a_coeffs[bp].get(block_id, 0) to U?
                #   No, a_coeffs[bp] gives the coefficient of the *block* in the sum A_bp.
                #   Wait, A_bp is defined by the Strassen formula.
                #   For P1, A_bp = A00 + A11.
                #   So A_bp[0,0] = A00, A_bp[0,1] = 0, A_bp[1,0] = 0, A_bp[1,1] = A11.
                #   So the local element (0,0) of A_bp is global block 00.
                #   The local element (1,1) of A_bp is global block 11.
                
                # Let's define the mapping from local 2x2 index to global block id.
                # For each bp, we have a sparse representation of A_bp and B_bp in terms of global blocks.
                
                # A_bp structure for each bp:
                # P1: A00 at (0,0), A11 at (1,1)
                # P2: A10 at (1,0), A11 at (1,1)
                # P3: A00 at (0,0)
                # P4: A11 at (1,1)
                # P5: A00 at (0,0), A01 at (0,1)
                # P6: A10 at (1,0), A00 at (0,0) with -1
                # P7: A01 at (0,1), A11 at (1,1) with -1
                
                a_struct = [
                    {(0,0): 0, (1,1): 3}, # P1
                    {(1,0): 2, (1,1): 3}, # P2
                    {(0,0): 0},           # P3
                    {(1,1): 3},           # P4
                    {(0,0): 0, (0,1): 1}, # P5
                    {(1,0): 2, (0,0): 0}, # P6 (coeffs applied later)
                    {(0,1): 1, (1,1): 3}  # P7
                ]
                # Coefficients for these blocks
                a_coeff_vals = [
                    {0: 1, 3: 1},
                    {2: 1, 3: 1},
                    {0: 1},
                    {3: 1},
                    {0: 1, 1: 1},
                    {2: 1, 0: -1},
                    {1: 1, 3: -1}
                ]
                
                b_struct = [
                    {(0,0): 0, (1,1): 3}, # P1
                    {(0,0): 0},           # P2
                    {(0,1): 1, (1,1): 3}, # P3
                    {(1,0): 2, (0,0): 0}, # P4
                    {(1,1): 3},           # P5
                    {(0,0): 0, (0,1): 1}, # P6
                    {(1,0): 2, (1,1): 3}  # P7
                ]
                b_coeff_vals = [
                    {0: 1, 3: 1},
                    {0: 1},
                    {1: 1, 3: -1},
                    {2: 1, 0: -1},
                    {3: 1},
                    {0: 1, 1: 1},
                    {2: 1, 3: 1}
                ]
                
                # Now, for this (bp, st), we iterate over the local 2x2 indices
                for li in range(2):
                    for lj in range(2):
                        local_idx = li * 2 + lj
                        coeff_u = u2[local_idx][st]
                        if coeff_u == 0: continue
                        
                        # Find which global block this local index maps to
                        # a_struct[bp] maps (li, lj) -> block_id
                        if (li, lj) in a_struct[bp]:
                            block_id = a_struct[bp][(li, lj)]
                            # The coefficient of this block in A_bp is a_coeff_vals[bp][block_id]
                            total_coeff = coeff_u * a_coeff_vals[bp][block_id]
                            
                            # Add to U
                            # Global index for block (bi, bj) element (ei, ej)
                            # Block (li, lj) is the block at position (li, lj) in the 2x2 block grid.
                            # So global i = 2*li + ei, global j = 2*lj + ej.
                            for ei in range(2):
                                for ej in range(2):
                                    gi = 2*li + ei
                                    gj = 2*lj + ej
                                    U[gi*4 + gj][col] += total_coeff
                                    
                for ki in range(2):
                    for kj in range(2):
                        local_idx = ki * 2 + kj
                        coeff_v = v2[local_idx][st]
                        if coeff_v == 0: continue
                        
                        if (ki, kj) in b_struct[bp]:
                            block_id = b_struct[bp][(ki, kj)]
                            total_coeff = coeff_v * b_coeff_vals[bp][block_id]
                            
                            for ei in range(2):
                                for ej in range(2):
                                    gi = 2*ki + ei
                                    gj = 2*kj + ej
                                    V[gi*4 + gj][col] += total_coeff
                                    
                for ci in range(2):
                    for cj in range(2):
                        local_idx = ci * 2 + cj
                        coeff_w = w2[local_idx][st]
                        if coeff_w == 0: continue
                        
                        # C struct is same as A struct for Strassen?
                        # C = A B. The block structure of C is determined by the same positions.
                        # C_bp[ci, cj] is the block at (ci, cj).
                        # The coefficients are in c_coeffs.
                        if (ci, cj) in a_struct[bp]: # Assuming C structure matches A in terms of which blocks are involved?
                            # Actually, C structure depends on the formula.
                            # C00 = M1+M4-M5+M7.
                            # M1 is P1. P1 contributes to C00 and C11.
                            # So for P1, C blocks involved are 00 and 11.
                            # This matches a_struct[0] which has 00 and 11.
                            # Let's verify for P2.
                            # P2 contributes to C10 and C11.
                            # a_struct[1] has 10 and 11.
                            # So yes, the set of blocks involved in C_bp is the same as A_bp?
                            # Not necessarily.
                            # Let's define c_struct explicitly.
                            
                            pass # We need c_struct
                            
                # Let's define c_struct and c_coeff_vals
                c_struct = [
                    {(0,0): 0, (1,1): 3}, # P1 -> C00, C11
                    {(1,0): 2, (1,1): 3}, # P2 -> C10, C11
                    {(0,1): 1, (1,1): 3}, # P3 -> C01, C11
                    {(0,0): 0, (1,0): 2}, # P4 -> C00, C10
                    {(0,0): 0, (0,1): 1}, # P5 -> C00, C01
                    {(1,1): 3},           # P6 -> C11
                    {(0,0): 0}            # P7 -> C00
                ]
                c_coeff_vals = [
                    {0: 1, 3: 1},
                    {2: 1, 3: -1},
                    {1: 1, 3: 1},
                    {0: 1, 2: 1},
                    {0: -1, 1: 1},
                    {3: 1},
                    {0: 1}
                ]
                
                for ci in range(2):
                    for cj in range(2):
                        local_idx = ci * 2 + cj
                        coeff_w = w2[local_idx][st]
                        if coeff_w == 0: continue
                        
                        if (ci, cj) in c_struct[bp]:
                            block_id = c_struct[bp][(ci, cj)]
                            total_coeff = coeff_w * c_coeff_vals[bp][block_id]
                            
                            for ei in range(2):
                                for ej in range(2):
                                    gi = 2*ci + ei
                                    gj = 2*cj + ej
                                    W[gi*4 + gj][col] += total_coeff

        return U, V, W

    # --- Construct 4x4x2 Schoolbook Decomposition ---
    def get_schoolbook_4x4x2():
        """
        Returns U, V, W for 4x4x2 Schoolbook.
        U: (16, 32), V: (24, 32), W: (24, 32)
        Note: V and W have 24 rows because p=6.
        """
        rank = 32
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(24)]
        W = [[0]*rank for _ in range(24)]
        
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4, 6): # k is 4, 5
                    U[i*4 + j][col] = 1
                    V[j*6 + k][col] = 1
                    W[k*4 + i][col] = 1
                    col += 1
        return U, V, W

    # --- Combine ---
    U1, V1, W1 = get_strassen_4x4x4()
    U2, V2, W2 = get_schoolbook_4x4x2()
    
    rank1 = 49
    rank2 = 32
    total_rank = rank1 + rank2
    
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(24)]
    W = [[0]*total_rank for _ in range(24)]
    
    # Copy Part 1
    for r in range(16):
        for c in range(rank1):
            U[r][c] = U1[r][c]
    for r in range(16):
        for c in range(rank1):
            V[r][c] = V1[r][c]
    for r in range(16):
        for c in range(rank1):
            W[r][c] = W1[r][c]
            
    # Copy Part 2
    for r in range(16):
        for c in range(rank2):
            U[r][rank1 + c] = U2[r][c]
    for r in range(24):
        for c in range(rank2):
            V[r][rank1 + c] = V2[r][c]
    for r in range(24):
        for c in range(rank2):
            W[r][rank1 + c] = W2[r][c]
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

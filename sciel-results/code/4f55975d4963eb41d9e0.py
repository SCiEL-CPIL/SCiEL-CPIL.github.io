"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Block-wise Strassen decomposition of rank 49 for 4x4 matrix multiplication.
    # Partition 4x4 matrices into 2x2 blocks of 2x2 matrices.
    # Apply Strassen at the block level (7 block multiplications), each block multiplication
    # is a 2x2 x 2x2 matrix product computed via schoolbook (rank 4).
    # Total rank = 7 * 4 = 28? No wait, let me reconsider.
    
    # Actually, the directive says "seven Strassen scalar multiplications are replaced by 
    # 2x2 matrix multiplications computed via the rank-4 schoolbook algorithm, yielding 
    # a total rank of 28"
    
    # But Strassen's algorithm for 2x2 blocks uses 7 multiplications of 2x2 matrices.
    # Each 2x2 matrix multiplication has rank 4 (schoolbook).
    # So total rank = 7 * 4 = 28.
    
    # Let me construct this properly.
    # A is 4x4, partitioned into blocks:
    # A = [[A00, A01],
    #      [A10, A11]]
    # where each block is 2x2.
    
    # B is 4x4, partitioned into blocks:
    # B = [[B00, B01],
    #      [B10, B11]]
    
    # C = A * B, partitioned into blocks:
    # C = [[C00, C01],
    #      [C10, C11]]
    
    # Strassen formulas for block multiplication:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Now, each M_k is a 2x2 matrix multiplication.
    # For each M_k = X * Y where X and Y are 2x2, we use schoolbook with rank 4.
    # The 4 scalar multiplications are:
    # x00*y00, x00*y10, x01*y00, x01*y10, x10*y00, x10*y10, x11*y00, x11*y10
    # Wait, no. Schoolbook for 2x2:
    # C00 = a00*b00 + a01*b10
    # C01 = a00*b01 + a01*b11
    # C10 = a10*b00 + a11*b10
    # C11 = a10*b01 + a11*b11
    # That's 8 multiplications, not 4.
    
    # Hmm, the rank of 2x2 matrix multiplication tensor is actually 7 (Strassen),
    # but schoolbook is 8. The minimum known rank is 7.
    
    # Wait, the directive says "rank-4 schoolbook algorithm". But 2x2 matrix multiplication
    # requires at least 7 scalar multiplications (Strassen's bound is 7, and it's known that
    # the rank is at least 7). Schoolbook uses 8.
    
    # Let me re-read the directive: "the seven Strassen scalar multiplications are replaced 
    # by 2x2 matrix multiplications computed via the rank-4 schoolbook algorithm"
    
    # This is confusing. Maybe they mean something different by "rank-4 schoolbook"?
    
    # Actually, I think there might be a misunderstanding. Let me just implement the 
    # straightforward Strassen-of-Strassen approach where each block multiplication 
    # uses the optimal 7-multiplication Strassen algorithm.
    
    # Total rank would be 7 * 7 = 49.
    
    # But the directive says rank 28. Let me just implement what they describe and see.
    
    # I'll implement the 49-rank decomposition since that's the standard Strassen-of-Strassen.
    
    n, m, p = DIMENSIONS  # 4, 4, 4
    
    # Indexing: A[i,j], B[j,k], C[i,k]
    # Block structure:
    # A[i,j] where i,j in {0,1,2,3}
    # Block A00: i in {0,1}, j in {0,1}
    # Block A01: i in {0,1}, j in {2,3}
    # Block A10: i in {2,3}, j in {0,1}
    # Block A11: i in {2,3}, j in {2,3}
    
    # B[j,k] where j,k in {0,1,2,3}
    # Block B00: j in {0,1}, k in {0,1}
    # Block B01: j in {0,1}, k in {2,3}
    # Block B10: j in {2,3}, k in {0,1}
    # Block B11: j in {2,3}, k in {2,3}
    
    # C[i,k] where i,k in {0,1,2,3}
    # Block C00: i in {0,1}, k in {0,1}
    # etc.
    
    # Strassen block products:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # For each block product M_s = X * Y, where X and Y are 2x2,
    # we use Strassen's algorithm with 7 scalar multiplications.
    
    # Let me define the Strassen algorithm for 2x2 matrices:
    # Given X = [[x00, x01], [x10, x11]] and Y = [[y00, y01], [y10, y11]]
    # Z = X * Y
    # 
    # s1 = y10 - y00
    # s2 = x00 + x01
    # s3 = x10 + x11
    # s4 = y00 + y11
    # s5 = x11 - x10
    # s6 = y01 - y11
    # s7 = x00 - x10
    # s8 = y10 + y11
    #
    # m1 = x00 * (y00 + y11)  -- wait, let me use the standard formulation
    #
    # Actually, let me use a clean formulation:
    # m1 = (x00 + x11) * (y00 + y11)
    # m2 = (x10 + x11) * y00
    # m3 = x00 * (y01 - y11)
    # m4 = x11 * (y10 - y00)
    # m5 = (x00 + x01) * y11
    # m6 = (x10 - x00) * (y00 + y01)
    # m7 = (x01 - x11) * (y10 + y11)
    #
    # z00 = m1 + m4 - m5 + m7
    # z01 = m3 + m5
    # z10 = m2 + m4
    # z11 = m1 - m2 + m3 + m6
    
    # So for each block product, we have 7 scalar multiplications.
    # Total: 7 * 7 = 49 scalar multiplications.
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49
    
    # W follows the trace convention W[k*n+i, r]
    
    # For each of the 7 Strassen block products s in 0..6:
    #   For each of the 7 Strassen scalar multiplications t in 0..6:
    #     We add a rank-1 term.
    
    # Let me define helper functions to get block elements.
    # A[i,j] has index i*m + j = i*4 + j in U
    # B[j,k] has index j*p + k = j*4 + k in V
    # C[i,k] has index k*n + i = k*4 + i in W
    
    # Block definitions:
    # A00: i in [0,1], j in [0,1] -> indices [0,1,4,5]
    # A01: i in [0,1], j in [2,3] -> indices [2,3,6,7]
    # A10: i in [2,3], j in [0,1] -> indices [8,9,12,13]
    # A11: i in [2,3], j in [2,3] -> indices [10,11,14,15]
    
    # B00: j in [0,1], k in [0,1] -> indices [0,1,4,5]
    # B01: j in [0,1], k in [2,3] -> indices [2,3,6,7]
    # B10: j in [2,3], k in [0,1] -> indices [8,9,12,13]
    # B11: j in [2,3], k in [2,3] -> indices [10,11,14,15]
    
    # C00: i in [0,1], k in [0,1] -> W[k*4+i] for i in [0,1], k in [0,1] -> [0,1,4,5]
    # C01: i in [0,1], k in [2,3] -> W[k*4+i] for i in [0,1], k in [2,3] -> [8,9,12,13]
    # C10: i in [2,3], k in [0,1] -> W[k*4+i] for i in [2,3], k in [0,1] -> [2,3,6,7]
    # C11: i in [2,3], k in [2,3] -> W[k*4+i] for i in [2,3], k in [2,3] -> [10,11,14,15]
    
    # Let me define the Strassen block products and their contributions.
    
    # For block product s, with left factor X and right factor Y:
    # X is a linear combination of A blocks, Y is a linear combination of B blocks.
    # Then m_t = (linear form in X) * (linear form in Y)
    # And the result z_ij contributes to a C block.
    
    # This is getting complex. Let me just enumerate all 49 rank-1 terms explicitly.
    
    # I'll define the 7 Strassen products at the block level, and for each,
    # the 7 Strassen products at the element level.
    
    # Block-level Strassen products (left, right) where left/right are sums/diffs of blocks:
    # s=0: M1 = (A00+A11) * (B00+B11)
    # s=1: M2 = (A10+A11) * B00
    # s=2: M3 = A00 * (B01-B11)
    # s=3: M4 = A11 * (B10-B00)
    # s=4: M5 = (A00+A01) * B11
    # s=5: M6 = (A10-A00) * (B00+B01)
    # s=6: M7 = (A01-A11) * (B10+B11)
    
    # For each block product, define the 2x2 matrices X and Y as linear combos of A/B blocks.
    
    # Let me define blocks as lists of (index, coefficient) for A and B.
    # A00: [(0,1),(1,1),(4,1),(5,1)]
    # A01: [(2,1),(3,1),(6,1),(7,1)]
    # A10: [(8,1),(9,1),(12,1),(13,1)]
    # A11: [(10,1),(11,1),(14,1),(15,1)]
    
    # B00: [(0,1),(1,1),(4,1),(5,1)]
    # B01: [(2,1),(3,1),(6,1),(7,1)]
    # B10: [(8,1),(9,1),(12,1),(13,1)]
    # B11: [(10,1),(11,1),(14,1),(15,1)]
    
    A00 = [(0,1),(1,1),(4,1),(5,1)]
    A01 = [(2,1),(3,1),(6,1),(7,1)]
    A10 = [(8,1),(9,1),(12,1),(13,1)]
    A11 = [(10,1),(11,1),(14,1),(15,1)]
    
    B00 = [(0,1),(1,1),(4,1),(5,1)]
    B01 = [(2,1),(3,1),(6,1),(7,1)]
    B10 = [(8,1),(9,1),(12,1),(13,1)]
    B11 = [(10,1),(11,1),(14,1),(15,1)]
    
    # C block contributions:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # C00: [(0,1),(1,1),(4,1),(5,1)]
    # C01: [(8,1),(9,1),(12,1),(13,1)]
    # C10: [(2,1),(3,1),(6,1),(7,1)]
    # C11: [(10,1),(11,1),(14,1),(15,1)]
    
    C00 = [(0,1),(1,1),(4,1),(5,1)]
    C01 = [(8,1),(9,1),(12,1),(13,1)]
    C10 = [(2,1),(3,1),(6,1),(7,1)]
    C11 = [(10,1),(11,1),(14,1),(15,1)]
    
    # For each block product s, define:
    # left_blocks: list of (block, coeff)
    # right_blocks: list of (block, coeff)
    # c_contrib: dict mapping block_product_index -> (C_block, coeff)
    
    # Block product definitions:
    # s=0: left=(A00,A11) with coeffs (1,1), right=(B00,B11) with coeffs (1,1)
    #       contributes to C00 with +1, C11 with +1
    # s=1: left=(A10,A11) with coeffs (1,1), right=B00 with coeff 1
    #       contributes to C10 with +1, C11 with -1
    # s=2: left=A00 with coeff 1, right=(B01,B11) with coeffs (1,-1)
    #       contributes to C01 with +1, C11 with +1
    # s=3: left=A11 with coeff 1, right=(B10,B00) with coeffs (1,-1)
    #       contributes to C00 with +1, C10 with +1
    # s=4: left=(A00,A01) with coeffs (1,1), right=B11 with coeff 1
    #       contributes to C00 with -1, C01 with +1
    # s=5: left=(A10,A00) with coeffs (1,-1), right=(B00,B01) with coeffs (1,1)
    #       contributes to C11 with +1
    # s=6: left=(A01,A11) with coeffs (1,-1), right=(B10,B11) with coeffs (1,1)
    #       contributes to C00 with +1
    
    block_products = [
        {  # s=0
            'left': [('A00', 1), ('A11', 1)],
            'right': [('B00', 1), ('B11', 1)],
            'contrib': [('C00', 1), ('C11', 1)]
        },
        {  # s=1
            'left': [('A10', 1), ('A11', 1)],
            'right': [('B00', 1)],
            'contrib': [('C10', 1), ('C11', -1)]
        },
        {  # s=2
            'left': [('A00', 1)],
            'right': [('B01', 1), ('B11', -1)],
            'contrib': [('C01', 1), ('C11', 1)]
        },
        {  # s=3
            'left': [('A11', 1)],
            'right': [('B10', 1), ('B00', -1)],
            'contrib': [('C00', 1), ('C10', 1)]
        },
        {  # s=4
            'left': [('A00', 1), ('A01', 1)],
            'right': [('B11', 1)],
            'contrib': [('C00', -1), ('C01', 1)]
        },
        {  # s=5
            'left': [('A10', 1), ('A00', -1)],
            'right': [('B00', 1), ('B01', 1)],
            'contrib': [('C11', 1)]
        },
        {  # s=6
            'left': [('A01', 1), ('A11', -1)],
            'right': [('B10', 1), ('B11', 1)],
            'contrib': [('C00', 1)]
        },
    ]
    
    # Strassen element-level products for 2x2 matrices:
    # For X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # m1 = (x00+x11)*(y00+y11)
    # m2 = (x10+x11)*y00
    # m3 = x00*(y01-y11)
    # m4 = x11*(y10-y00)
    # m5 = (x00+x01)*y11
    # m6 = (x10-x00)*(y00+y01)
    # m7 = (x01-x11)*(y10+y11)
    
    # z00 = m1 + m4 - m5 + m7
    # z01 = m3 + m5
    # z10 = m2 + m4
    # z11 = m1 - m2 + m3 + m6
    
    # For each element-level product t, define:
    # left: (x_idx, coeff) where x_idx in {0:x00, 1:x01, 2:x10, 3:x11}
    # right: (y_idx, coeff) where y_idx in {0:y00, 1:y01, 2:y10, 3:y11}
    # z_contrib: (z_idx, coeff)
    
    # x00 -> index 0, x01 -> index 1, x10 -> index 2, x11 -> index 3
    # y00 -> index 0, y01 -> index 1, y10 -> index 2, y11 -> index 3
    # z00 -> index 0, z01 -> index 1, z10 -> index 2, z11 -> index 3
    
    element_products = [
        {  # t=0: m1 = (x00+x11)*(y00+y11)
            'left': [(0, 1), (3, 1)],
            'right': [(0, 1), (3, 1)],
            'z': [(0, 1), (3, 1)]  # z00 +1, z11 +1
        },
        {  # t=1: m2 = (x10+x11)*y00
            'left': [(2, 1), (3, 1)],
            'right': [(0, 1)],
            'z': [(2, 1), (3, -1)]  # z10 +1, z11 -1
        },
        {  # t=2: m3 = x00*(y01-y11)
            'left': [(0, 1)],
            'right': [(1, 1), (3, -1)],
            'z': [(1, 1), (3, 1)]  # z01 +1, z11 +1
        },
        {  # t=3: m4 = x11*(y10-y00)
            'left': [(3, 1)],
            'right': [(2, 1), (0, -1)],
            'z': [(0, 1), (2, 1)]  # z00 +1, z10 +1
        },
        {  # t=4: m5 = (x00+x01)*y11
            'left': [(0, 1), (1, 1)],
            'right': [(3, 1)],
            'z': [(0, -1), (1, 1)]  # z00 -1, z01 +1
        },
        {  # t=5: m6 = (x10-x00)*(y00+y01)
            'left': [(2, 1), (0, -1)],
            'right': [(0, 1), (1, 1)],
            'z': [(3, 1)]  # z11 +1
        },
        {  # t=6: m7 = (x01-x11)*(y10+y11)
            'left': [(1, 1), (3, -1)],
            'right': [(2, 1), (3, 1)],
            'z': [(0, 1)]  # z00 +1
        },
    ]
    
    # Now, for each block product s and element product t, we have a rank-1 term.
    # The left linear form in A is: sum over block products in 'left' of (block * coeff)
    #   where each block is a set of (A_index, block_coeff)
    #   so the total coefficient for A_index is sum of (block_coeff * element_coeff)
    
    # Wait, I need to be more careful.
    # For block product s, the 2x2 matrix X has entries:
    # X[0,0] = sum of A00[0,0]*coeff + A11[0,0]*coeff + ...
    # Actually, X is a 2x2 matrix where X[i,j] is a linear combination of A entries.
    
    # Let me define it differently.
    # For block product s, X is a 2x2 matrix. X[i,j] for i,j in {0,1} is:
    #   sum over (block, block_coeff) in left of (block[i,j] * block_coeff)
    # where block[i,j] is the entry at position (i,j) in that block.
    
    # Similarly for Y.
    
    # Then for element product t, the scalar multiplication is:
    #   (sum of X[left_indices] * coeffs) * (sum of Y[right_indices] * coeffs)
    
    # And this contributes to Z[z_idx] with coefficient z_coeff.
    
    # And Z[z_idx] is a block in C, so it contributes to C entries.
    
    # Let me just iterate and build U, V, W.
    
    col = 0
    for s in range(7):
        bp = block_products[s]
        for t in range(7):
            ep = element_products[t]
            
            # Build U contribution: for each A_index, coefficient
            # X[i,j] = sum over (block_name, block_coeff) in bp['left'] of block[i,j] * block_coeff
            # Then the left linear form for element product t is:
            #   sum over (x_idx, x_coeff) in ep['left'] of X[x_idx_row, x_idx_col] * x_coeff
            # where x_idx 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
            
            x_map = {0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)}
            y_map = {0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)}
            
            U_coeff = {}
            for x_idx, x_coeff in ep['left']:
                xi, xj = x_map[x_idx]
                for block_name, block_coeff in bp['left']:
                    if block_name == 'A00':
                        a_idx = xi * 4 + xj
                    elif block_name == 'A01':
                        a_idx = xi * 4 + (xj + 2)
                    elif block_name == 'A10':
                        a_idx = (xi + 2) * 4 + xj
                    elif block_name == 'A11':
                        a_idx = (xi + 2) * 4 + (xj + 2)
                    else:
                        continue
                    coeff = x_coeff * block_coeff
                    if a_idx in U_coeff:
                        U_coeff[a_idx] += coeff
                    else:
                        U_coeff[a_idx] = coeff
            
            V_coeff = {}
            for y_idx, y_coeff in ep['right']:
                yi, yj = y_map[y_idx]
                for block_name, block_coeff in bp['right']:
                    if block_name == 'B00':
                        b_idx = yi * 4 + yj
                    elif block_name == 'B01':
                        b_idx = yi * 4 + (yj + 2)
                    elif block_name == 'B10':
                        b_idx = (yi + 2) * 4 + yj
                    elif block_name == 'B11':
                        b_idx = (yi + 2) * 4 + (yj + 2)
                    else:
                        continue
                    coeff = y_coeff * block_coeff
                    if b_idx in V_coeff:
                        V_coeff[b_idx] += coeff
                    else:
                        V_coeff[b_idx] = coeff
            
            # Z contribution:
            for z_idx, z_coeff in ep['z']:
                zi, zj = x_map[z_idx]  # same mapping for z
                for block_name, block_bcoeff in bp['contrib']:
                    if block_name == 'C00':
                        c_i = zi
                        c_k = zj
                    elif block_name == 'C01':
                        c_i = zi
                        c_k = zj + 2
                    elif block_name == 'C10':
                        c_i = zi + 2
                        c_k = zj
                    elif block_name == 'C11':
                        c_i = zi + 2
                        c_k = zj + 2
                    else:
                        continue
                    w_idx = c_k * 4 + c_i  # W[k*n+i]
                    coeff = z_coeff * block_bcoeff
                    # Add to W
                    if w_idx in locals():
                        pass
                    # We'll accumulate W contributions
                    if not hasattr(construct, '_w_accum'):
                        construct._w_accum = {}
                    if w_idx not in construct._w_accum:
                        construct._w_accum[w_idx] = 0
                    construct._w_accum[w_idx] += coeff
            
            # Set U and V
            for a_idx, coeff in U_coeff.items():
                U[a_idx][col] = coeff
            for b_idx, coeff in V_coeff.items():
                V[b_idx][col] = coeff
            
            # Set W
            for w_idx, coeff in construct._w_accum.items():
                W[w_idx][col] = coeff
            construct._w_accum = {}
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

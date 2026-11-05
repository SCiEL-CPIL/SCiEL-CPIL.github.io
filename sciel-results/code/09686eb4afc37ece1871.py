"""Matrix multiplication 4,4,6 tensor decomposition."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # 4, 4, 6

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term_ijk(u_dict, v_dict, w_dict):
        """
        Add a rank-1 term using (i,j) for U, (j,k) for V, (i,k) for W indexing.
        u_dict: {(i,j): coeff}
        v_dict: {(j,k): coeff}
        w_dict: {(i,k): coeff}
        """
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        
        for (i, j), coeff in u_dict.items():
            u_col[i * m + j] += coeff
        for (j, k), coeff in v_dict.items():
            v_col[j * p + k] += coeff
        for (i, k), coeff in w_dict.items():
            w_col[k * n + i] += coeff
        
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)

    def strassen_2x2(oi, oj, ok):
        """
        Add Strassen decomposition for 2x2 * 2x2 block multiplication.
        oi: offset row in A
        oj: offset col in A / row in B
        ok: offset col in B
        """
        # m1 = (x00+x11)(y00+y11)
        add_term_ijk(
            {(oi, oj): 1, (oi+1, oj+1): 1},
            {(oj, ok): 1, (oj+1, ok+1): 1},
            {(oi, ok): 1, (oi+1, ok+1): 1}
        )
        
        # m2 = (x10+x11)y00
        add_term_ijk(
            {(oi+1, oj): 1, (oi+1, oj+1): 1},
            {(oj, ok): 1},
            {(oi+1, ok): 1, (oi+1, ok+1): -1}
        )
        
        # m3 = x00(y01-y11)
        add_term_ijk(
            {(oi, oj): 1},
            {(oj, ok+1): 1, (oj+1, ok+1): -1},
            {(oi, ok+1): 1, (oi+1, ok+1): 1}
        )
        
        # m4 = x11(y10-y00)
        add_term_ijk(
            {(oi+1, oj+1): 1},
            {(oj+1, ok): 1, (oj, ok): -1},
            {(oi, ok): 1, (oi+1, ok): 1}
        )
        
        # m5 = (x00+x01)y11
        add_term_ijk(
            {(oi, oj): 1, (oi, oj+1): 1},
            {(oj+1, ok+1): 1},
            {(oi, ok): -1, (oi, ok+1): 1}
        )
        
        # m6 = (x10-x00)(y00+y01)
        add_term_ijk(
            {(oi+1, oj): 1, (oi, oj): -1},
            {(oj, ok): 1, (oj, ok+1): 1},
            {(oi+1, ok+1): 1}
        )
        
        # m7 = (x01-x11)(y10+y11)
        add_term_ijk(
            {(oi, oj+1): 1, (oi+1, oj+1): -1},
            {(oj+1, ok): 1, (oj+1, ok+1): 1},
            {(oi, ok): 1}
        )

    def strassen_super(oi, oj, ok, u_blocks, v_blocks, w_blocks):
        """
        Add a Strassen term where the sub-blocks are 2x2 matrices.
        This effectively scales a standard 2x2 Strassen term by the block structure.
        u_blocks: {block_id: coeff} where block_id 0=TL, 1=TR, 2=BL, 3=BR
        """
        # Block offsets in local 2x2 grid
        block_offsets = {
            0: (0, 0),
            1: (0, 1),
            2: (1, 0),
            3: (1, 1)
        }
        
        # We need to generate the 7 rank-1 terms for the 2x2 Strassen algorithm,
        # but each "scalar" operation is replaced by a 2x2 matrix operation.
        # However, Strassen's algorithm relies on specific linear combinations.
        # If we just scale the whole 2x2 Strassen, it works if the blocks are identical.
        # But here, the blocks are different parts of the larger matrix.
        
        # Let's construct the U, V, W dictionaries for the super-term.
        # The 7 terms of Strassen 2x2 are:
        # 1. (A00+A11)(B00+B11) -> C00, C11
        # 2. (A10+A11)B00 -> C10, C11(-1)
        # 3. A00(B01-B11) -> C01, C11
        # 4. A11(B10-B00) -> C00, C10
        # 5. (A00+A01)B11 -> C00(-1), C01
        # 6. (A10-A00)(B00+B01) -> C11
        # 7. (A01-A11)(B10+B11) -> C00
        
        # For each of these 7 terms, we have a pattern of U, V, W contributions.
        # We need to map the "scalar" Strassen indices to the actual matrix indices.
        
        # Term 1: U has A00, A11; V has B00, B11; W has C00, C11
        # In our block context:
        # U gets contributions from u_blocks[0] and u_blocks[3]
        # V gets contributions from v_blocks[0] and v_blocks[3]
        # W gets contributions from w_blocks[0] and w_blocks[3]
        
        # This is getting complex to generalize "super" directly from 2x2 Strassen
        # because the linear combinations (like A00+A11) mix different blocks.
        # It is safer to explicitly build the 4x4 Strassen terms using the 2x2 Strassen
        # on the 2x2 sub-blocks.
        
        # Let's implement 4x4 Strassen by calling strassen_2x2 on the appropriate sub-blocks.
        # But wait, 4x4 Strassen uses 7 multiplications of 2x2 matrices.
        # Each 2x2 multiplication has rank 7. Total rank 49.
        # The structure is:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # To implement M1 = (A00+A11)(B00+B11), we cannot just call strassen_2x2 once
        # because the inputs are sums of blocks.
        # We must expand (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
        # This would require 4 separate 2x2 multiplications, which is rank 4*7=28 for just M1.
        # That defeats the purpose.
        
        # The correct way to implement 4x4 Strassen with rank 49 is to use the fact that
        # the Strassen algorithm works on the "super-matrices".
        # We treat A00, A01, A10, A11 as scalars in the Strassen formulas, but they are 2x2 matrices.
        # The rank of a 2x2 matrix multiplication is 7.
        # So each "scalar" multiplication in the Strassen formula (like A00*B00) costs rank 7.
        # The Strassen formula has 7 such multiplications.
        # Therefore, the total rank is 7 * 7 = 49.
        
        # How to construct the terms?
        # We need to generate the 7 terms for each of the 7 Strassen steps.
        # But the Strassen steps involve linear combinations of the sub-matrices.
        # e.g., M1 = (A00+A11)(B00+B11).
        # This term contributes to C00 and C11.
        # The U part is (A00+A11), V part is (B00+B11), W part is (C00+C11).
        # Since A00 and A11 are 2x2 blocks, the U vector for this term will have non-zero entries
        # in the rows corresponding to A00 and A11.
        # Specifically, for the term (A00+A11)(B00+B11), the U vector is the sum of the U vectors
        # for A00 and A11.
        
        # Let's define a helper to add a "block-strassen" term.
        # A term is defined by:
        # - A set of (block, coeff) for U
        # - A set of (block, coeff) for V
        # - A set of (block, coeff) for W
        # And we need to decompose the product of these linear combinations into rank-1 terms.
        # Wait, the product of linear combinations is NOT rank 1.
        # (A00+A11)(B00+B11) is a matrix product of two 2x2 matrices (the sums).
        # Let U_sum = A00+A11, V_sum = B00+B11.
        # We need to decompose U_sum * V_sum.
        # We can use the rank-7 Strassen decomposition for 2x2 * 2x2 on U_sum and V_sum.
        # The U_sum and V_sum are 2x2 matrices.
        # So we can call strassen_2x2 on the "virtual" 2x2 matrices formed by the sums.
        
        # How to map "virtual" 2x2 indices to global indices?
        # The virtual matrix U_sum has elements:
        # [A00[0,0]+A11[0,0], A00[0,1]+A11[0,1]]
        # [A00[1,0]+A11[1,0], A00[1,1]+A11[1,1]]
        
        # In the global matrix A, A00 is at (0:2, 0:2) and A11 is at (2:4, 2:4).
        # So U_sum[i,j] = A[i,j] + A[i+2, j+2].
        
        # We can create a mapping from virtual (i,j) to global (I,J) with coefficients.
        # virtual (0,0) -> global (0,0) coeff 1, global (2,2) coeff 1
        # virtual (0,1) -> global (0,1) coeff 1, global (2,3) coeff 1
        # virtual (1,0) -> global (1,0) coeff 1, global (3,2) coeff 1
        # virtual (1,1) -> global (1,1) coeff 1, global (3,3) coeff 1
        
        # We can generalize this.
        # For a term with U blocks {b1: c1, b2: c2, ...}, V blocks {b3: c3, ...}, W blocks {b4: c4, ...}
        # We form virtual 2x2 matrices U_v, V_v, W_v.
        # Then we apply the 7 rank-1 terms of 2x2 Strassen to U_v, V_v, W_v.
        # Each rank-1 term of the virtual Strassen is a product of linear combinations of U_v elements.
        # We then expand these linear combinations into global indices.
        
        # This is doable but complex. Let's write a function to do this expansion.
        
        def add_virtual_strassen(u_map, v_map, w_map):
            """
            u_map: {(local_i, local_j): [(global_i, global_j, coeff), ...]}
            This maps each element of the virtual 2x2 matrix to its global contributions.
            """
            # The 7 terms of 2x2 Strassen
            terms = [
                # u_terms, v_terms, w_terms
                # Each term is a list of (virtual_index, coeff)
                [
                    [((0,0), 1), ((1,1), 1)], # U: A00+A11
                    [((0,0), 1), ((1,1), 1)], # V: B00+B11
                    [((0,0), 1), ((1,1), 1)]  # W: C00+C11
                ],
                [
                    [((1,0), 1), ((1,1), 1)], # U: A10+A11
                    [((0,0), 1)],             # V: B00
                    [((1,0), 1), ((1,1), -1)] # W: C10-C11
                ],
                [
                    [((0,0), 1)],             # U: A00
                    [((0,1), 1), ((1,1), -1)],# V: B01-B11
                    [((0,1), 1), ((1,1), 1)]  # W: C01+C11
                ],
                [
                    [((1,1), 1)],             # U: A11
                    [((1,0), 1), ((0,0), -1)],# V: B10-B00
                    [((0,0), 1), ((1,0), 1)]  # W: C00+C10
                ],
                [
                    [((0,0), 1), ((0,1), 1)], # U: A00+A01
                    [((1,1), 1)],             # V: B11
                    [((0,0), -1), ((0,1), 1)] # W: -C00+C01
                ],
                [
                    [((1,0), 1), ((0,0), -1)],# U: A10-A00
                    [((0,0), 1), ((0,1), 1)], # V: B00+B01
                    [((1,1), 1)]              # W: C11
                ],
                [
                    [((0,1), 1), ((1,1), -1)],# U: A01-A11
                    [((1,0), 1), ((1,1), 1)], # V: B10+B11
                    [((0,0), 1)]              # W: C00
                ]
            ]
            
            for term in terms:
                u_dict = {}
                v_dict = {}
                w_dict = {}
                
                # Expand U
                for (vi, vj), coeff in term[0]:
                    for (gi, gj, gcoeff) in u_map.get((vi, vj), []):
                        key = (gi, gj)
                        u_dict[key] = u_dict.get(key, 0) + coeff * gcoeff
                
                # Expand V
                for (vi, vj), coeff in term[1]:
                    for (gi, gj, gcoeff) in v_map.get((vi, vj), []):
                        # V indices are (row, col) in the virtual 2x2.
                        # Global V index is (global_row, global_col).
                        # In our context, V is B.
                        key = (gi, gj)
                        v_dict[key] = v_dict.get(key, 0) + coeff * gcoeff
                
                # Expand W
                for (vi, vj), coeff in term[2]:
                    for (gi, gj, gcoeff) in w_map.get((vi, vj), []):
                        # W indices are (row, col) in the virtual 2x2.
                        # Global W index is (global_row, global_col).
                        # In our context, W is C.
                        key = (gi, gj)
                        w_dict[key] = w_dict.get(key, 0) + coeff * gcoeff
                        
                add_term_ijk(u_dict, v_dict, w_dict)
                
        # Now, construct the maps for the 4x4 Strassen terms.
        # Block 0: A[0:2, 0:2], B[0:2, 0:2], C[0:2, 0:2]
        # Block 1: A[0:2, 2:4], B[0:2, 2:4], C[0:2, 2:4]
        # Block 2: A[2:4, 0:2], B[2:4, 0:2], C[2:4, 0:2]
        # Block 3: A[2:4, 2:4], B[2:4, 2:4], C[2:4, 2:4]
        
        # Helper to create a map for a set of blocks
        def create_block_map(blocks, offset_i, offset_j, is_v=False):
            """
            blocks: {block_id: coeff}
            offset_i, offset_j: base offset for the 2x2 block.
            is_v: if True, this is for V (B matrix), else U (A matrix) or W (C matrix).
            For U and W, the virtual (vi, vj) maps to global (oi+vi, oj+vj) + block_offset.
            For V, the virtual (vi, vj) maps to global (oj+vi, ok+vj) + block_offset?
            Wait, V is B. B is m x p.
            In the 4x4 split, B is split into 2x2 blocks of size 2x2.
            B[0:2, 0:2] is block 0. B[0:2, 2:4] is block 1.
            So for V, the row index is the "col" index of A (j), and col index is k.
            In the virtual 2x2 for V, (vi, vj) corresponds to local row vi, local col vj.
            Global row = oj + vi + block_row_offset.
            Global col = ok + vj + block_col_offset.
            
            Let's unify:
            For U (A): global (i, j) = (oi + vi + block_i_off, oj + vj + block_j_off)
            For V (B): global (j, k) = (oj + vi + block_i_off, ok + vj + block_k_off)
            For W (C): global (i, k) = (oi + vi + block_i_off, ok + vj + block_k_off)
            
            Block offsets:
            Block 0: (0, 0)
            Block 1: (0, 1) -> col offset 1 (in 2x2 grid) -> global col offset 2
            Block 2: (1, 0) -> row offset 1 -> global row offset 2
            Block 3: (1, 1) -> row offset 1, col offset 1 -> global row offset 2, col offset 2
            """
            map = {}
            for b_id, coeff in blocks.items():
                bi = b_id // 2
                bj = b_id % 2
                # Global offsets for this block
                gi_off = bi * 2
                gj_off = bj * 2
                
                for vi in range(2):
                    for vj in range(2):
                        if is_v:
                            gi = oj + vi + gi_off
                            gj = ok + vj + gj_off
                        else: # U or W
                            gi = oi + vi + gi_off
                            gj = (oj if is_u else ok) + vj + gj_off # Need to distinguish U and W
                
                # This is getting messy. Let's pass explicit global offsets.
            return map

        # Let's simplify by just hardcoding the maps for each of the 7 terms of 4x4 Strassen.
        # Term 1: M1 = (A00+A11)(B00+B11)
        # U blocks: 0 (+1), 3 (+1)
        # V blocks: 0 (+1), 3 (+1)
        # W blocks: 0 (+1), 3 (+1)
        
        # We need to build u_map, v_map, w_map for this term.
        # u_map: virtual (vi,vj) -> list of (global_i, global_j, coeff)
        
        def build_maps(u_blocks, v_blocks, w_blocks):
            u_map = {}
            v_map = {}
            w_map = {}
            
            for b_id, coeff in u_blocks.items():
                bi = b_id // 2
                bj = b_id % 2
                gi_off = bi * 2
                gj_off = bj * 2
                for vi in range(2):
                    for vj in range(2):
                        key = (vi, vj)
                        if key not in u_map:
                            u_map[key] = []
                        u_map[key].append((oi + vi + gi_off, oj + vj + gj_off, coeff))
            
            for b_id, coeff in v_blocks.items():
                bi = b_id // 2
                bj = b_id % 2
                gi_off = bi * 2
                gj_off = bj * 2
                for vi in range(2):
                    for vj in range(2):
                        key = (vi, vj)
                        if key not in v_map:
                            v_map[key] = []
                        # V is B. Global row is j, global col is k.
                        # In the 4x4 split, B is 4x4.
                        # Block (bi, bj) of B is at rows 2*bi:2*bi+2, cols 2*bj:2*bj+2.
                        # So global row = 2*bi + vi, global col = 2*bj + vj.
                        # Note: oj is the column offset in A, which is the row offset in B.
                        # So oj should be 0 for the top-left 4x4 block.
                        v_map[key].append((oj + vi + gi_off, ok + vj + gj_off, coeff))
            
            for b_id, coeff in w_blocks.items():
                bi = b_id // 2
                bj = b_id % 2
                gi_off = bi * 2
                gj_off = bj * 2
                for vi in range(2):
                    for vj in range(2):
                        key = (vi, vj)
                        if key not in w_map:
                            w_map[key] = []
                        # W is C. Global row is i, global col is k.
                        # Block (bi, bj) of C is at rows 2*bi:2*bi+2, cols 2*bj:2*bj+2.
                        w_map[key].append((oi + vi + gi_off, ok + vj + gj_off, coeff))
            
            return u_map, v_map, w_map

        # Term 1
        u_map, v_map, w_map = build_maps({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1})
        add_virtual_strassen(u_map, v_map, w_map)
        
        # Term 2: M2 = (A10+A11)B00
        u_map, v_map, w_map = build_maps({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1})
        add_virtual_strassen(u_map, v_map, w_map)
        
        # Term 3: M3 = A00(B01-B11)
        u_map, v_map, w_map = build_maps({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1})
        add_virtual_strassen(u_map, v_map, w_map)
        
        # Term 4: M4 = A11(B10-B00)
        u_map, v_map, w_map = build_maps({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1})
        add_virtual_strassen(u_map, v_map, w_map)
        
        # Term 5: M5 = (A00+A01)B11
        u_map, v_map, w_map = build_maps({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1})
        add_virtual_strassen(u_map, v_map, w_map)
        
        # Term 6: M6 = (A10-A00)(B00+B01)
        u_map, v_map, w_map = build_maps({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1})
        add_virtual_strassen(u_map, v_map, w_map)
        
        # Term 7: M7 = (A01-A11)(B10+B11)
        u_map, v_map, w_map = build_maps({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
        add_virtual_strassen(u_map, v_map, w_map)

    # Build 4x4 * 4x4 part (columns 0-3 of B)
    strassen_super(0, 0, 0, {}, {}, {}) # This will call the inner logic. 
    # Wait, strassen_super as defined above doesn't take u_blocks etc in the final version.
    # I defined strassen_super to take them but then ignored them and hardcoded the terms.
    # Let's just call the hardcoded terms for the first 4x4 block.
    
    # Actually, I'll just call the hardcoded logic for the 4x4 * 4x4 part.
    # The function strassen_super above is a bit messy. Let's just execute the 7 terms directly.
    
    # Reset U_cols, V_cols, W_cols to be safe if called multiple times? No, construct is called once.
    
    # 4x4 * 4x4 part
    # Term 1
    u_map, v_map, w_map = build_maps({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1})
    add_virtual_strassen(u_map, v_map, w_map)
    
    # Term 2
    u_map, v_map, w_map = build_maps({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1})
    add_virtual_strassen(u_map, v_map, w_map)
    
    # Term 3
    u_map, v_map, w_map = build_maps({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1})
    add_virtual_strassen(u_map, v_map, w_map)
    
    # Term 4
    u_map, v_map, w_map = build_maps({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1})
    add_virtual_strassen(u_map, v_map, w_map)
    
    # Term 5
    u_map, v_map, w_map = build_maps({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1})
    add_virtual_strassen(u_map, v_map, w_map)
    
    # Term 6
    u_map, v_map, w_map = build_maps({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1})
    add_virtual_strassen(u_map, v_map, w_map)
    
    # Term 7
    u_map, v_map, w_map = build_maps({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    add_virtual_strassen(u_map, v_map, w_map)
    
    # Now add the 4x4 * 4x2 part (columns 4-5 of B)
    # Split into 4 independent 2x2 * 2x2 multiplications?
    # A is 4x4, B is 4x2.
    # C is 4x2.
    # We can split A into 2x2 blocks (4 blocks) and B into 2x2 blocks (2 blocks, vertically stacked? No, B is 4x2, so 2 blocks of 2x2 stacked vertically).
    # B0: rows 0-1, cols 4-5
    # B1: rows 2-3, cols 4-5
    # C0: rows 0-1, cols 4-5
    # C1: rows 2-3, cols 4-5
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    # This is 4 multiplications of 2x2 * 2x2.
    # Each has rank 7. Total rank 28.
    
    # Let's add these 4 Strassen decompositions.
    # Multiplication 1: A00 * B0 -> C0
    # A00 is at (0,0), B0 is at (0,4), C0 is at (0,4)
    # Note: B columns are 4,5. So ok=4.
    strassen_2x2(0, 0, 4)
    
    # Multiplication 2: A01 * B1 -> C0
    # A01 is at (0,2), B1 is at (2,4), C0 is at (0,4)
    # Wait, A01 is rows 0-1, cols 2-3. B1 is rows 2-3, cols 4-5.
    # The product A01 * B1 contributes to C0 (rows 0-1, cols 4-5).
    # strassen_2x2(oi, oj, ok) computes C[oi:oi+2, ok:ok+2] = A[oi:oi+2, oj:oj+2] * B[oj:oj+2, ok:ok+2]
    # Here, A is A01, so oi=0, oj=2.
    # B is B1, so oj=2, ok=4.
    # C is C0, so oi=0, ok=4.
    # So strassen_2x2(0, 2, 4)
    strassen_2x2(0, 2, 4)
    
    # Multiplication 3: A10 * B0 -> C1
    # A10 is at (2,0), B0 is at (0,4), C1 is at (2,4)
    # strassen_2x2(2, 0, 4)
    strassen_2x2(2, 0, 4)
    
    # Multiplication 4: A11 * B1 -> C1
    # A11 is at (2,2), B1 is at (2,4), C1 is at (2,4)
    # strassen_2x2(2, 2, 4)
    strassen_2x2(2, 2, 4)

    # Total rank should be 49 + 28 = 77.
    
    U = np.array(U_cols).T  # Shape (rank, n*m) -> (n*m, rank)
    V = np.array(V_cols).T  # Shape (rank, m*p) -> (m*p, rank)
    W = np.array(W_cols).T  # Shape (rank, p*n) -> (p*n, rank)
    
    # Verify
    def verify(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :])
                    if abs(val - 1.0) > 1e-8:
                        return False
        return True

    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, column] = 1
                    V[j*p+k, column] = 1
                    W[k*n+i, column] = 1
                    column += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

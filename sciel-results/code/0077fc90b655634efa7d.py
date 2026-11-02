import numpy as np
import time

def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication.
    Uses recursive Strassen algorithm.
    
    U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    Convention: 
    U[i*4+j, r] corresponds to A[i,j]
    V[j*4+k, r] corresponds to B[j,k]
    W[k*4+i, r] corresponds to C[k,i] (note: W index is k*4+i)
    
    The tensor identity is:
    sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k in {0,1,2,3}
    """
    start_time = time.time()
    deadline = start_time + 115.0

    n = 4
    rank = 49
    
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((16, rank), dtype=float)
    W = np.zeros((16, rank), dtype=float)

    # Helper to add to U, V, W
    # U index: i*4 + j
    # V index: j*4 + k
    # W index: k*4 + i
    
    def add_U(i, j, r, coeff):
        U[i*4 + j, r] += coeff

    def add_V(j, k, r, coeff):
        V[j*4 + k, r] += coeff

    def add_W(k, i, r, coeff):
        W[k*4 + i, r] += coeff

    # Recursive function to decompose 2x2 block multiplication
    # It assumes we are working on a sub-problem of size 2x2.
    # The 'offset' and 'stride' handle the mapping to the global 4x4 indices.
    # For the top level (4x4), we split into 2x2 blocks.
    # Let A = [[A00, A01], [A10, A11]] where each block is 2x2.
    # A00 corresponds to A[0:2, 0:2].
    # A01 corresponds to A[0:2, 2:4].
    # A10 corresponds to A[2:4, 0:2].
    # A11 corresponds to A[2:4, 2:4].
    
    # We define a function to generate the 7 Strassen products for 2x2 blocks
    # given the global index ranges.
    
    # Strassen 2x2 formula:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We will instantiate 7 such decompositions (one for each Strassen term of the top level)
    # Each top-level term is a 2x2 matrix multiplication.
    # So we call the 2x2 decomposition 7 times, with different index mappings.
    
    # Let's define the 7 top-level Strassen terms for 4x4:
    # A blocks: A00 (0,0), A01 (0,2), A10 (2,0), A11 (2,2) -- using (row_start, col_start)
    # B blocks: B00 (0,0), B01 (0,2), B10 (2,0), B11 (2,2)
    # C blocks: C00 (0,0), C01 (0,2), C10 (2,0), C11 (2,2)
    
    # Term 1: (A00 + A11)(B00 + B11) -> contributes to C00, C11
    # Term 2: (A10 + A11)B00 -> contributes to C10, C11
    # Term 3: A00(B01 - B11) -> contributes to C01, C11
    # Term 4: A11(B10 - B00) -> contributes to C00, C10
    # Term 5: (A00 + A01)B11 -> contributes to C00, C01
    # Term 6: (A10 - A00)(B00 + B01) -> contributes to C11
    # Term 7: (A01 - A11)(B10 + B11) -> contributes to C00
    
    # We need to decompose each 2x2 block multiplication into 7 scalar multiplications.
    # Total 7 * 7 = 49 scalar multiplications.
    
    # Let's define a helper to decompose a 2x2 matrix multiplication
    # given the global indices of the 4 sub-blocks of A, B, and C.
    
    def decompose_2x2(A_blocks, B_blocks, C_blocks, base_rank):
        """
        A_blocks: dict with keys '00', '01', '10', '11' mapping to (row_start, col_start)
        B_blocks: dict with keys '00', '01', '10', '11' mapping to (row_start, col_start)
        C_blocks: dict with keys '00', '01', '10', '11' mapping to (row_start, col_start)
        base_rank: starting rank index
        """
        # 1. M1 = (A00 + A11)(B00 + B11)
        # 2. M2 = (A10 + A11)B00
        # 3. M3 = A00(B01 - B11)
        # 4. M4 = A11(B10 - B00)
        # 5. M5 = (A00 + A01)B11
        # 6. M6 = (A10 - A00)(B00 + B01)
        # 7. M7 = (A01 - A11)(B10 + B11)
        
        # We will process each of the 7 terms.
        # For each term, we have a linear combination of A-blocks and a linear combination of B-blocks.
        # We need to expand these into global indices.
        
        # Helper to get global indices for a block
        def get_block_indices(block_key, blocks_dict):
            r_start, c_start = blocks_dict[block_key]
            # The block is 2x2.
            # Returns a list of (global_row, global_col) for the 4 elements
            indices = []
            for dr in range(2):
                for dc in range(2):
                    indices.append((r_start + dr, c_start + dc))
            return indices

        # Define the 7 terms as lists of (coeff, block_key)
        terms = [
            # M1
            ([ (1, '00'), (1, '11') ], [ (1, '00'), (1, '11') ]),
            # M2
            ([ (1, '10'), (1, '11') ], [ (1, '00') ]),
            # M3
            ([ (1, '00') ], [ (1, '01'), (-1, '11') ]),
            # M4
            ([ (1, '11') ], [ (1, '10'), (-1, '00') ]),
            # M5
            ([ (1, '00'), (1, '01') ], [ (1, '11') ]),
            # M6
            ([ (1, '10'), (-1, '00') ], [ (1, '00'), (1, '01') ]),
            # M7
            ([ (1, '01'), (-1, '11') ], [ (1, '10'), (1, '11') ])
        ]
        
        # C contributions for each M
        # M1 -> C00 (+1), C11 (+1)
        # M2 -> C10 (+1), C11 (-1)
        # M3 -> C01 (+1), C11 (+1)
        # M4 -> C00 (+1), C10 (+1)
        # M5 -> C00 (-1), C01 (+1)
        # M6 -> C11 (+1)
        # M7 -> C00 (+1)
        
        c_contribs = [
            [ ('00', 1), ('11', 1) ],
            [ ('10', 1), ('11', -1) ],
            [ ('01', 1), ('11', 1) ],
            [ ('00', 1), ('10', 1) ],
            [ ('00', -1), ('01', 1) ],
            [ ('11', 1) ],
            [ ('00', 1) ]
        ]
        
        for t_idx in range(7):
            a_terms, b_terms = terms[t_idx]
            r_idx = base_rank + t_idx
            
            # Expand A terms
            # U will have non-zeros at indices corresponding to the A elements involved
            # V will have non-zeros at indices corresponding to the B elements involved
            
            # Note: The rank is for the scalar multiplication.
            # U[i,j,r] * V[j,k,r] * W[k,i,r]
            
            # For a specific term t_idx, we are computing M_t = (sum c_a A_block) (sum c_b B_block)
            # This M_t contributes to C blocks.
            
            # We need to add to U, V, W such that:
            # sum_{i,j,k} U[i,j,r] V[j,k,r] W[k,i,r] = sum_{c_blocks} coeff_c * C_block
            # Actually, the decomposition is:
            # M_t = A'_t B'_t
            # A'_t is a 2x2 matrix (sum of A blocks)
            # B'_t is a 2x2 matrix (sum of B blocks)
            # The product M_t is a 2x2 matrix.
            # We decompose this 2x2 product into 7 scalar products?
            # NO. The recursive step says:
            # We decompose the 4x4 product into 7 2x2 products.
            # Each 2x2 product is decomposed into 7 scalar products.
            # So total 49 scalar products.
            
            # My previous logic was slightly off.
            # The 7 terms above (M1..M7) are the 7 2x2 matrix multiplications.
            # EACH of these 2x2 multiplications must be decomposed using Strassen (7 scalar mults).
            
            # So, for each of the 7 top-level terms, I need to call a function that decomposes a 2x2 matrix multiplication.
            
            pass # Placeholder

    # Let's implement the 2x2 decomposition properly.
    # A 2x2 matrix multiplication C = AB can be decomposed into 7 scalar multiplications.
    # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]].
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # We will apply this to each of the 7 top-level 2x2 block multiplications.
    
    def add_scalar_decomp(A_lin, B_lin, C_lin, base_rank):
        """
        A_lin: list of (coeff, global_row, global_col) for A
        B_lin: list of (coeff, global_row, global_col) for B
        C_lin: list of (coeff, global_row, global_col) for C
        base_rank: starting rank index for the 7 scalar mults
        """
        # We need to form the 7 Strassen terms for this specific 2x2 multiplication.
        # But A_lin and B_lin are already linear combinations of global elements.
        # We can just expand the 7 formulas directly.
        
        # m1 = (a+d)(e+h)
        # We need to identify which global elements correspond to a, b, c, d and e, f, g, h.
        # This is tricky because A_lin is a sum of blocks.
        
        # Better approach:
        # For each of the 7 top-level terms, we have specific A-blocks and B-blocks.
        # We can treat the 2x2 blocks as the "elements" a,b,c,d and e,f,g,h.
        # Then we decompose the 2x2 product.
        
        pass

    # Let's do it explicitly for the 7 top-level terms.
    
    # Top Level Term 1: (A00 + A11)(B00 + B11)
    # A-part: A00 + A11. B-part: B00 + B11.
    # C-part: C00 + C11.
    
    # We need to decompose (A00+A11)(B00+B11) into 7 scalar mults.
    # Let X = A00 + A11, Y = B00 + B11.
    # X is a 2x2 matrix, Y is a 2x2 matrix.
    # We decompose XY.
    
    # To do this, we need to map the 4 elements of X and Y to global indices.
    # X[0,0] = A[0,0] + A[2,2]
    # X[0,1] = A[0,1] + A[2,3]
    # X[1,0] = A[1,0] + A[3,2]
    # X[1,1] = A[1,1] + A[3,3]
    
    # Y[0,0] = B[0,0] + B[2,2]
    # Y[0,1] = B[0,2] + B[2,3]
    # Y[1,0] = B[2,0] + B[3,2]
    # Y[1,1] = B[2,2] + B[3,3]
    
    # We can create a helper that takes a list of (coeff, row, col) for X and Y,
    # and a list of (coeff, row, col) for C (the result block),
    # and adds the 7 Strassen scalar decompositions.
    
    def decompose_matrix_mult(X_terms, Y_terms, C_terms, base_rank):
        """
        X_terms: list of (coeff, i, j) for the left matrix (2x2)
        Y_terms: list of (coeff, j, k) for the right matrix (2x2)
        C_terms: list of (coeff, i, k) for the result matrix (2x2)
        base_rank: starting rank index
        """
        # We need to compute the 7 Strassen terms for X*Y.
        # X = [[x00, x01], [x10, x11]]
        # Y = [[y00, y01], [y10, y11]]
        
        # We need to extract x00, x01, x10, x11 from X_terms.
        # Since X is 2x2, the indices are local (0,0), (0,1), (1,0), (1,1).
        # But X_terms uses global indices?
        # No, it's better to pass local indices and a mapping function.
        
        # Let's change the interface:
        # X_map: dict local (r,c) -> list of (coeff, global_row, global_col)
        # Y_map: dict local (r,c) -> list of (coeff, global_row, global_col)
        # C_map: dict local (r,c) -> list of (coeff, global_row, global_col)
        
        pass

    # This is getting complicated. Let's just hardcode the 49 terms.
    # There are 7 top-level terms. Each has 7 sub-terms.
    
    # Top Term 1: X = A00 + A11, Y = B00 + B11, C = C00 + C11
    # X local (0,0) -> A[0,0], A[2,2]
    # X local (0,1) -> A[0,1], A[2,3]
    # X local (1,0) -> A[1,0], A[3,2]
    # X local (1,1) -> A[1,1], A[3,3]
    
    # Y local (0,0) -> B[0,0], B[2,2]
    # Y local (0,1) -> B[0,2], B[2,3]
    # Y local (1,0) -> B[2,0], B[3,2]
    # Y local (1,1) -> B[2,2], B[3,3]
    
    # C local (0,0) -> C[0,0], C[2,2]
    # C local (0,1) -> C[0,2], C[2,3]
    # C local (1,0) -> C[1,0], C[3,2]
    # C local (1,1) -> C[1,2], C[3,3]
    
    # We will write a function that takes these mappings and generates the 7 ranks.
    
    def generate_ranks(X_map, Y_map, C_map, base_rank):
        # X_map: { (0,0): [(1, 0, 0), (1, 2, 2)], ... }
        # Y_map: { (0,0): [(1, 0, 0), (1, 2, 2)], ... }
        # C_map: { (0,0): [(1, 0, 0), (1, 2, 2)], ... }
        
        # Strassen terms:
        # m1 = (x00+x11)(y00+y11)
        # m2 = (x10+x11)y00
        # m3 = x00(y01-y11)
        # m4 = x11(y10-y00)
        # m5 = (x00+x01)y11
        # m6 = (x10-x00)(y00+y01)
        # m7 = (x01-x11)(y10+y11)
        
        # C00 = m1 + m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6
        
        # For each m_k, we have a linear combination of X elements and a linear combination of Y elements.
        # We expand these to global indices.
        
        terms = [
            # (X_coeffs, Y_coeffs, C_contribs)
            # m1
            ([ (1, (0,0)), (1, (1,1)) ], [ (1, (0,0)), (1, (1,1)) ], [ (1, (0,0)), (1, (1,1)) ]),
            # m2
            ([ (1, (1,0)), (1, (1,1)) ], [ (1, (0,0)) ], [ (1, (1,0)), (-1, (1,1)) ]),
            # m3
            ([ (1, (0,0)) ], [ (1, (0,1)), (-1, (1,1)) ], [ (1, (0,1)), (1, (1,1)) ]),
            # m4
            ([ (1, (1,1)) ], [ (1, (1,0)), (-1, (0,0)) ], [ (1, (0,0)), (1, (1,0)) ]),
            # m5
            ([ (1, (0,0)), (1, (0,1)) ], [ (1, (1,1)) ], [ (-1, (0,0)), (1, (0,1)) ]),
            # m6
            ([ (1, (1,0)), (-1, (0,0)) ], [ (1, (0,0)), (1, (0,1)) ], [ (1, (1,1)) ]),
            # m7
            ([ (1, (0,1)), (-1, (1,1)) ], [ (1, (1,0)), (1, (1,1)) ], [ (1, (0,0)) ]),
        ]
        
        for t_idx in range(7):
            x_coeffs, y_coeffs, c_coeffs = terms[t_idx]
            r_idx = base_rank + t_idx
            
            # Expand X
            for coeff, (lr, lc) in x_coeffs:
                for (g_coeff, gi, gj) in X_map[(lr, lc)]:
                    add_U(gi, gj, r_idx, coeff * g_coeff)
            
            # Expand Y
            for coeff, (lr, lc) in y_coeffs:
                for (g_coeff, gi, gj) in Y_map[(lr, lc)]:
                    add_V(gi, gj, r_idx, coeff * g_coeff)
            
            # Expand C
            for coeff, (lr, lc) in c_coeffs:
                for (g_coeff, gi, gj) in C_map[(lr, lc)]:
                    # W index is k*4 + i.
                    # C is indexed by (i, k) in standard math, but here C_map gives (row, col) of C.
                    # If C_map gives (i, k), then W index is k*4 + i.
                    add_W(gj, gi, r_idx, coeff * g_coeff)
                    
    # Define the mappings for the 7 top-level terms
    
    # Term 1: X = A00+A11, Y = B00+B11, C = C00+C11
    X1 = {
        (0,0): [(1, 0, 0), (1, 2, 2)],
        (0,1): [(1, 0, 1), (1, 2, 3)],
        (1,0): [(1, 1, 0), (1, 3, 2)],
        (1,1): [(1, 1, 1), (1, 3, 3)]
    }
    Y1 = {
        (0,0): [(1, 0, 0), (1, 2, 2)],
        (0,1): [(1, 0, 2), (1, 2, 3)],
        (1,0): [(1, 2, 0), (1, 3, 2)],
        (1,1): [(1, 2, 2), (1, 3, 3)]
    }
    C1 = {
        (0,0): [(1, 0, 0), (1, 2, 2)],
        (0,1): [(1, 0, 2), (1, 2, 3)],
        (1,0): [(1, 1, 0), (1, 3, 2)],
        (1,1): [(1, 1, 2), (1, 3, 3)]
    }
    generate_ranks(X1, Y1, C1, 0)
    
    # Term 2: X = A10+A11, Y = B00, C = C10 - C11
    X2 = {
        (0,0): [(1, 2, 0), (1, 2, 2)],
        (0,1): [(1, 2, 1), (1, 2, 3)],
        (1,0): [(1, 3, 0), (1, 3, 2)],
        (1,1): [(1, 3, 1), (1, 3, 3)]
    }
    Y2 = {
        (0,0): [(1, 0, 0)],
        (0,1): [(1, 0, 2)],
        (1,0): [(1, 2, 0)],
        (1,1): [(1, 2, 2)]
    }
    C2 = {
        (0,0): [(1, 1, 0)], # C10 is at (1,0)
        (0,1): [(1, 1, 2)], # C11 is at (1,2)
        (1,0): [],
        (1,1): []
    }
    # Wait, C10 is the block at row 2-3, col 0-1.
    # C11 is the block at row 2-3, col 2-3.
    # In C_map, (0,0) corresponds to C10?
    # Let's align C_map keys with the local 2x2 structure of the RESULT.
    # The result of Term 2 is a 2x2 matrix.
    # This 2x2 matrix is added to C10 and subtracted from C11?
    # No, Strassen formula: C10 = M2 + M4, C11 = M1 - M2 + M3 + M6.
    # So M2 contributes +1 to C10 and -1 to C11.
    # So C_map should map the local 2x2 result to the global C blocks.
    # Local (0,0) of the result -> Global C10 (0,0) i.e. C[2,0]
    # Local (0,1) of the result -> Global C10 (0,1) i.e. C[2,1]
    # Local (1,0) of the result -> Global C11 (0,0) i.e. C[2,2] ? No.
    # C10 is a 2x2 block. C11 is a 2x2 block.
    # The result of M2 is a 2x2 matrix.
    # This matrix is added to C10.
    # So Local (0,0) -> C[2,0], Local (0,1) -> C[2,1], Local (1,0) -> C[3,0], Local (1,1) -> C[3,1].
    # And it is subtracted from C11.
    # So Local (0,0) -> -C[2,2], Local (0,1) -> -C[2,3], Local (1,0) -> -C[3,2], Local (1,1) -> -C[3,3].
    
    C2 = {
        (0,0): [(1, 2, 0), (-1, 2, 2)],
        (0,1): [(1, 2, 1), (-1, 2, 3)],
        (1,0): [(1, 3, 0), (-1, 3, 2)],
        (1,1): [(1, 3, 1), (-1, 3, 3)]
    }
    generate_ranks(X2, Y2, C2, 7)
    
    # Term 3: X = A00, Y = B01-B11, C = C01 + C11
    X3 = {
        (0,0): [(1, 0, 0)],
        (0,1): [(1, 0, 1)],
        (1,0): [(1, 1, 0)],
        (1,1): [(1, 1, 1)]
    }
    Y3 = {
        (0,0): [(1, 0, 2), (-1, 2, 2)],
        (0,1): [(1, 0, 3), (-1, 2, 3)],
        (1,0): [(1, 2, 0), (-1, 2, 0)], # B01 is (0,2)-(0,3), B11 is (2,2)-(2,3)
        # Wait, B01 is block (0,1). B11 is block (1,1).
        # B01: B[0,2], B[0,3], B[1,2], B[1,3]
        # B11: B[2,2], B[2,3], B[3,2], B[3,3]
        # Y = B01 - B11
        (0,0): [(1, 0, 2), (-1, 2, 2)],
        (0,1): [(1, 0, 3), (-1, 2, 3)],
        (1,0): [(1, 1, 2), (-1, 3, 2)],
        (1,1): [(1, 1, 3), (-1, 3, 3)]
    }
    C3 = {
        (0,0): [(1, 0, 2), (1, 2, 2)], # C01 is (0,2)-(0,3), (1,2)-(1,3). C11 is (2,2)-(2,3), (3,2)-(3,3)
        (0,1): [(1, 0, 3), (1, 2, 3)],
        (1,0): [(1, 1, 2), (1, 3, 2)],
        (1,1): [(1, 1, 3), (1, 3, 3)]
    }
    generate_ranks(X3, Y3, C3, 14)
    
    # Term 4: X = A11, Y = B10-B00, C = C00 + C10
    X4 = {
        (0,0): [(1, 2, 2)],
        (0,1): [(1, 2, 3)],
        (1,0): [(1, 3, 2)],
        (1,1): [(1, 3, 3)]
    }
    Y4 = {
        (0,0): [(1, 2, 0), (-1, 0, 0)],
        (0,1): [(1, 2, 1), (-1, 0, 1)],
        (1,0): [(1, 3, 0), (-1, 1, 0)],
        (1,1): [(1, 3, 1), (-1, 1, 1)]
    }
    C4 = {
        (0,0): [(1, 0, 0), (1, 1, 0)], # C00 (0,0)-(0,1), (1,0)-(1,1). C10 (2,0)-(2,1), (3,0)-(3,1)
        (0,1): [(1, 0, 1), (1, 1, 1)],
        (1,0): [(1, 2, 0), (1, 3, 0)],
        (1,1): [(1, 2, 1), (1, 3, 1)]
    }
    generate_ranks(X4, Y4, C4, 21)
    
    # Term 5: X = A00+A01, Y = B11, C = -C00 + C01
    X5 = {
        (0,0): [(1, 0, 0), (1, 0, 2)],
        (0,1): [(1, 0, 1), (1, 0, 3)],
        (1,0): [(1, 1, 0), (1, 1, 2)],
        (1,1): [(1, 1, 1), (1, 1, 3)]
    }
    Y5 = {
        (0,0): [(1, 2, 2)],
        (0,1): [(1, 2, 3)],
        (1,0): [(1, 3, 2)],
        (1,1): [(1, 3, 3)]
    }
    C5 = {
        (0,0): [(-1, 0, 0), (1, 0, 2)],
        (0,1): [(-1, 0, 1), (1, 0, 3)],
        (1,0): [(-1, 1, 0), (1, 1, 2)],
        (1,1): [(-1, 1, 1), (1, 1, 3)]
    }
    generate_ranks(X5, Y5, C5, 28)
    
    # Term 6: X = A10-A00, Y = B00+B01, C = C11
    X6 = {
        (0,0): [(1, 2, 0), (-1, 0, 0)],
        (0,1): [(1, 2, 1), (-1, 0, 1)],
        (1,0): [(1, 3, 0), (-1, 1, 0)],
        (1,1): [(1, 3, 1), (-1, 1, 1)]
    }
    Y6 = {
        (0,0): [(1, 0, 0), (1, 0, 2)],
        (0,1): [(1, 0, 1), (1, 0, 3)],
        (1,0): [(1, 2, 0), (1, 2, 2)],
        (1,1): [(1, 2, 1), (1, 2, 3)]
    }
    C6 = {
        (0,0): [(1, 2, 2)],
        (0,1): [(1, 2, 3)],
        (1,0): [(1, 3, 2)],
        (1,1): [(1, 3, 3)]
    }
    generate_ranks(X6, Y6, C6, 35)
    
    # Term 7: X = A01-A11, Y = B10+B11, C = C00
    X7 = {
        (0,0): [(1, 0, 2), (-1, 2, 2)],
        (0,1): [(1, 0, 3), (-1, 2, 3)],
        (1,0): [(1, 1, 2), (-1, 3, 2)],
        (1,1): [(1, 1, 3), (-1, 3, 3)]
    }
    Y7 = {
        (0,0): [(1, 2, 0), (1, 2, 2)],
        (0,1): [(1, 2, 1), (1, 2, 3)],
        (1,0): [(1, 3, 0), (1, 3, 2)],
        (1,1): [(1, 3, 1), (1, 3, 3)]
    }
    C7 = {
        (0,0): [(1, 0, 0)],
        (0,1): [(1, 0, 1)],
        (1,0): [(1, 1, 0)],
        (1,1): [(1, 1, 1)]
    }
    generate_ranks(X7, Y7, C7, 42)
    
    # Verify
    if time.time() < deadline - 2:
        U_np = U
        V_np = V
        W_np = W
        max_err = 0.0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    u_idx = i * 4 + j
                    v_idx = j * 4 + k
                    w_idx = k * 4 + i
                    val = np.sum(U_np[u_idx] * V_np[v_idx] * W_np[w_idx])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        if max_err > 1e-10:
            print(f"Verification failed with max error {max_err}")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


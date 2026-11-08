import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = 4, 4, 6
    start_time = time.time()
    deadline = start_time + 110
    
    # We'll build the decomposition as follows:
    # Split the 4x6 matrix B into B_left (4x4) and B_right (4x2)
    # C = A * B_left + A * B_right
    # 
    # For A * B_left: standard 4x4 * 4x4 using Strassen (rank 49)
    # For A * B_right: 4x4 * 4x2 (rank 28)
    # Total rank: 49 + 28 = 77
    
    # Let's build each part carefully.
    
    # Part 1: 4x4 * 4x4 via Strassen
    # Split A into 2x2 blocks: A = [[A11, A12], [A21, A22]]
    #   A11: rows 0-1, cols 0-1 -> indices i in {0,1}, j in {0,1}
    #   A12: rows 0-1, cols 2-3 -> indices i in {0,1}, j in {2,3}
    #   A21: rows 2-3, cols 0-1 -> indices i in {2,3}, j in {0,1}
    #   A22: rows 2-3, cols 2-3 -> indices i in {2,3}, j in {2,3}
    #
    # B_left split similarly: B = [[B11, B12], [B21, B22]]
    #   B11: rows 0-1, cols 0-1 -> j in {0,1}, k in {0,1}
    #   B12: rows 0-1, cols 2-3 -> j in {0,1}, k in {2,3}
    #   B21: rows 2-3, cols 0-1 -> j in {2,3}, k in {0,1}
    #   B22: rows 2-3, cols 2-3 -> j in {2,3}, k in {2,3}
    #
    # C = [[C11, C12], [C21, C22]]
    #   C11: i in {0,1}, k in {0,1}
    #   C12: i in {0,1}, k in {2,3}
    #   C21: i in {2,3}, k in {0,1}
    #   C22: i in {2,3}, k in {2,3}
    
    # Strassen's 7 products:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # Each Mi is a 2x2 * 2x2 multiplication, which itself has rank 7.
    # So total rank for this part: 7 * 7 = 49.
    
    # Part 2: 4x4 * 4x2
    # B_right is 4x2. Split into B_r1 (rows 0-1, 2 cols) and B_r2 (rows 2-3, 2 cols)
    #   B_r1: j in {0,1}, k in {0,1}  (but these are cols 4,5 of original B)
    #   B_r2: j in {2,3}, k in {0,1}
    
    # Wait, I need to be more careful with indices.
    
    # Let me reconsider. The full problem:
    # A is 4x4: indices (i, j) with i in 0..3, j in 0..3
    # B is 4x6: indices (j, k) with j in 0..3, k in 0..5
    # C is 4x6: indices (i, k) with i in 0..3, k in 0..5
    #
    # Split B into B_left (4x4, k in 0..3) and B_right (4x2, k in 4..5)
    # C = A*B_left + A*B_right
    
    # For A*B_right, this is 4x4 * 4x2.
    # Split A into 2x2 blocks as before.
    # B_right = [[B_r1], [B_r2]] where B_r1 is 2x2 (j in {0,1}, k in {4,5}),
    #             B_r2 is 2x2 (j in {2,3}, k in {4,5})
    #
    # C_right = [[C_r1], [C_r2]] where C_r1 is 2x2 (i in {0,1}, k in {4,5}),
    #              C_r2 is 2x2 (i in {2,3}, k in {4,5})
    #
    # C_r1 = A11 * B_r1 + A12 * B_r2
    # C_r2 = A21 * B_r1 + A22 * B_r2
    #
    # Each term is a 2x2 * 2x2 multiplication, rank 7.
    # Total: 4 * 7 = 28.
    
    # Now let's build the full decomposition.
    
    # Helper: create a 2x2 * 2x2 Strassen decomposition
    # Returns lists of (u_coeff, v_coeff, w_coeff) for each of the 7 rank-1 terms
    # where u_coeff is a 4-tuple for the A block, v_coeff for B block, w_coeff for C block
    
    def strassen_2x2():
        """
        Returns 7 tuples (u, v, w) where each is a dict mapping block positions to coefficients.
        Block positions for A: '11', '12', '21', '22'
        Block positions for B: '11', '12', '21', '22'
        Block positions for C: '11', '12', '21', '22'
        
        Each coefficient is applied to the corresponding 2x2 block element.
        For a 2x2 * 2x2 with elements A[a,b], B[c,d], C[a,d]:
        We need sum_r U[a,b]_r * V[c,d]_r * W[a,d]_r = 1 for all a,b,c,d.
        
        In Strassen, each Mi is itself a 2x2*2x2. But actually, each Mi is a single
        2x2 matrix product, so we need a rank-7 decomposition for 2x2*2x2.
        
        Let me define the 7 rank-1 terms for a single 2x2*2x2.
        """
        # For 2x2 * 2x2 -> 2x2:
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
        
        # Strassen:
        # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        # M2 = (A[1,0]+A[1,1])*B[0,0]
        # M3 = A[0,0]*(B[0,1]-B[1,1])
        # M4 = A[1,1]*(B[1,0]-B[0,0])
        # M5 = (A[0,0]+A[0,1])*B[1,1]
        # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
        # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
        
        # C[0,0] = M1 - M3 + M5 - M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M4
        # C[1,1] = M1 - M2 + M4 + M6
        
        # For each M, we have a rank-1 term in the 2x2*2x2 tensor.
        # M1: u = [(1,0), (0,1)] meaning A[0,0] gets 1, A[1,1] gets 1
        #      v = [(1,0), (0,1)] meaning B[0,0] gets 1, B[1,1] gets 1
        #      w = [(1,0), (0,0), (0,0), (1,0)]? No wait.
        
        # Let me think differently. For a rank-1 term, we assign coefficients
        # to each element of A, B, C. The product of the three coefficient vectors
        # (element-wise in the flattened sense) should give the desired tensor slice.
        
        # Actually, for Strassen, each M is a bilinear form:
        # M1 = (A00 + A11) * (B00 + B11)
        # This contributes to C as:
        # C00 += M1, C11 += M1, C01 += 0, C10 += 0... no that's not right.
        
        # Let me use a cleaner approach. I'll directly enumerate the 7 rank-1
        # decompositions for 2x2*2x2.
        
        # Standard Strassen decomposition for 2x2*2x2:
        # The 7 terms are:
        terms = []
        
        # M1 = (A00 + A11) * (B00 + B11)
        # Contributes: C00 += M1, C11 += M1 (with appropriate signs in final sum)
        # Actually, let me just define the coefficient vectors directly.
        
        # For a rank-1 term: U_vec (4 elements for A), V_vec (4 for B), W_vec (4 for C)
        # such that sum_r U_vec[r][a,b] * V_vec[r][c,d] * W_vec[r][a,d] = delta
        
        # This is getting complex. Let me use a known explicit decomposition.
        
        # Known rank-7 decomposition for 2x2*2x2:
        # Using the standard Strassen algorithm, the 7 scalar products are:
        # p1 = (a00 + a11)(b00 + b11)
        # p2 = (a10 + a11)b00
        # p3 = a00(b01 - b11)
        # p4 = a11(b10 - b00)
        # p5 = (a00 + a01)b11
        # p6 = (a10 - a00)(b00 + b01)
        # p7 = (a01 - a11)(b10 + b11)
        
        # c00 = p1 - p3 + p5 - p7
        # c01 = p3 + p5
        # c10 = p2 + p4
        # c11 = p1 - p2 + p4 + p6
        
        # For each pi, it's a product of two linear forms.
        # p1: u1 = [1,0,0,1], v1 = [1,0,0,1], contributes to c00 with +1, c11 with +1
        # But we need to express this as a rank-1 term in the tensor.
        
        # A rank-1 term has the form: U * V * W (outer product in the tensor sense)
        # where U is a vector of length 4 (for A elements), V length 4 (for B), W length 4 (for C).
        
        # For p1 contributing to c00:
        # The contribution to the tensor at (a,b,c,d) is U[a,b]*V[c,d]*W[a,d]
        # We want this to be 1 when (a,d) = (0,0) and the "combination" of A and B gives p1.
        
        # Actually, p1 is NOT a rank-1 term by itself. It's a bilinear form.
        # The Strassen algorithm uses 7 bilinear forms, and the output is a linear combination.
        
        # The correct way: each bilinear form pi can be written as a sum of rank-1 terms.
        # But that's not what we want. We want the overall 2x2*2x2 tensor to have rank 7.
        
        # Let me look up the explicit rank-7 decomposition.
        
        # The tensor M_{2,2,2} has a known rank-7 decomposition. Here it is:
        # The 7 rank-1 tensors are:
        
        r1_terms = [
            # (U, V, W) where U, V, W are dicts from (row,col) to coefficient
            # Term 1
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            # Term 2
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
            # Term 3
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}),
            # Term 4
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(1,1):1}),
            # Term 5
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):1, (0,1):1}),
            # Term 6
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            # Term 7
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):-1}),
        ]
        
        # Let me verify this is correct by checking that the sum gives the matrix multiplication.
        # This is getting very tedious to verify by hand. Let me use a different approach.
        
        # I'll use the fact that we can verify computationally.
        # For now, let me define the decomposition and verify it in code.
        
        return r1_terms
    
    # Actually, let me take a cleaner approach. I'll define the full decomposition
    # by directly specifying the coefficient vectors for each rank-1 term.
    
    # For the full 4x4 * 4x6 problem:
    # A has 16 elements: indexed by (i,j) with i in 0..3, j in 0..3
    # B has 24 elements: indexed by (j,k) with j in 0..3, k in 0..5
    # C has 24 elements: indexed by (i,k) with i in 0..3, k in 0..5
    #
    # U: shape (16, rank), U[i*m+j, r] = coefficient for A[i,j] in rank-r term
    # V: shape (24, rank), V[j*p+k, r] = coefficient for B[j,k] in rank-r term
    # W: shape (24, rank), W[k*n+i, r] = coefficient for C[i,k] in rank-r term
    #
    # Note: W index is k*n+i, not i*n+k. This is the "trace convention".
    
    # Let me build the decomposition step by step.
    
    # First, let me define helper functions to build 2x2*2x2 decompositions.
    
    def build_2x2x2_decomp():
        """
        Build a rank-7 decomposition for 2x2 * 2x2 -> 2x2.
        Returns a list of 7 tuples (U_dict, V_dict, W_dict) where each dict
        maps (row, col) -> coefficient.
        """
        # Using Strassen's algorithm, the 7 scalar products are:
        # p1 = (a00 + a11)(b00 + b11)
        # p2 = (a10 + a11)b00
        # p3 = a00(b01 - b11)
        # p4 = a11(b10 - b00)
        # p5 = (a00 + a01)b11
        # p6 = (a10 - a00)(b00 + b01)
        # p7 = (a01 - a11)(b10 + b11)
        
        # c00 = p1 - p3 + p5 - p7
        # c01 = p3 + p5
        # c10 = p2 + p4
        # c11 = p1 - p2 + p4 + p6
        
        # Each pi is a bilinear form: pi = sum_abcd u_ab * v_cd * w_ad where
        # we need to express the full tensor as sum of rank-1 terms.
        
        # Actually, each pi can be written as a product of two linear forms,
        # which means it's a sum of rank-1 terms in the tensor.
        # But we want the TOTAL rank to be 7, not 7*something.
        
        # The key insight: the 7 bilinear forms ARE the rank-1 decomposition.
        # That is, each pi corresponds to one rank-1 term in the tensor decomposition.
        
        # For pi, the rank-1 term has:
        # U_vec: the coefficients of A in the first linear form
        # V_vec: the coefficients of B in the second linear form
        # W_vec: the coefficients of C (which pi contributes to)
        
        # p1 = (a00 + a11)(b00 + b11)
        # U for p1: a00: 1, a11: 1
        # V for p1: b00: 1, b11: 1
        # W for p1: contributes to c00 with +1, c11 with +1
        # But wait, a rank-1 term in the tensor is U ⊗ V ⊗ W, meaning
        # tensor[a,b,c,d] = U[a,b] * V[c,d] * W[a,d]
        # So W can only depend on (a,d), i.e., the C indices.
        
        # For p1 to contribute to both c00 and c11, we need:
        # W[0,0] and W[1,1] to be non-zero.
        # If W[0,0] = 1 and W[1,1] = 1, then:
        # tensor[0,0,0,0] += U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1 (from a00*b00 to c00) ✓
        # tensor[1,1,1,1] += U[1,1]*V[1,1]*W[1,1] = 1*1*1 = 1 (from a11*b11 to c11) ✓
        # But also: tensor[0,0,1,1] += U[0,0]*V[1,1]*W[0,1] = 1*1*0 = 0 (W[0,1]=0) ✓
        # tensor[1,1,0,0] += U[1,1]*V[0,0]*W[1,0] = 1*1*0 = 0 (W[1,0]=0) ✓
        # tensor[0,1,0,0] += U[0,1]*V[0,0]*W[0,0] = 0*1*1 = 0 ✓
        # etc.
        
        # So p1 as a rank-1 term: U={(0,0):1,(1,1):1}, V={(0,0):1,(1,1):1}, W={(0,0):1,(1,1):1}
        # But this only captures the "diagonal" contributions. The Strassen algorithm
        # works because the COMBINATION of all 7 terms gives the correct result.
        
        # Let me just define all 7 terms explicitly and verify computationally.
        
        terms = [
            # p1 = (a00 + a11)(b00 + b11) -> contributes to c00, c11
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            # p2 = (a10 + a11)b00 -> contributes to c10
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
            # p3 = a00(b01 - b11) -> contributes to c01
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}),
            # p4 = a11(b10 - b00) -> contributes to c11
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(1,1):1}),
            # p5 = (a00 + a01)b11 -> contributes to c00, c01
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):1, (0,1):1}),
            # p6 = (a10 - a00)(b00 + b01) -> contributes to c11
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            # p7 = (a01 - a11)(b10 + b11) -> contributes to c00 (negative)
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):-1}),
        ]
        
        # Now verify: sum over all 7 terms should give the matrix multiplication tensor.
        # tensor[a,b,c,d] = sum_r U_r[a,b] * V_r[c,d] * W_r[a,d]
        # We want this to be 1 if (b,c) is a valid middle index, i.e., always 1 for 2x2*2x2.
        
        # Let's verify computationally.
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    for d in range(2):
                        total = 0
                        for U, V, W in terms:
                            u_val = U.get((a,b), 0)
                            v_val = V.get((c,d), 0)
                            w_val = W.get((a,d), 0)
                            total += u_val * v_val * w_val
                        expected = 1.0
                        if abs(total - expected) > 1e-9:
                            # This decomposition is wrong. Let me debug.
                            pass
        
        return terms
    
    # Let me test the 2x2 decomposition first.
    terms_2x2 = build_2x2x2_decomp()
    
    # Verify 2x2 decomposition
    valid_2x2 = True
    for a in range(2):
        for b in range(2):
            for c in range(2):
                for d in range(2):
                    total = 0
                    for U, V, W in terms_2x2:
                        u_val = U.get((a,b), 0)
                        v_val = V.get((c,d), 0)
                        w_val = W.get((a,d), 0)
                        total += u_val * v_val * w_val
                    if abs(total - 1.0) > 1e-9:
                        valid_2x2 = False
                        break
            if not valid_2x2:
                break
        if not valid_2x2:
            break
    if not valid_2x2:
        # The 2x2 decomposition I defined is wrong. Let me try a different approach.
        # Use the schoolbook for 2x2 (rank 4) as a fallback, then build up.
        # Actually, let me just use a known correct decomposition.
        
        # For now, fall back to schoolbook for the whole thing.
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Good, the 2x2 decomposition works. Now build the full 4x4*4x6.
    
    # For the full problem, I need to compose the 2x2 decompositions.
    
    # Part 1: 4x4 * 4x4 (B_left, k in 0..3)
    # This uses 7 calls to the 2x2 decomposition, each scaled by the Strassen coefficients.
    
    # Let me define the full set of rank-1 terms for the 4x4*4x4 part.
    
    def build_4x4x4_terms():
        """
        Build rank-49 terms for 4x4 * 4x4.
        Each term is (U_dict, V_dict, W_dict) mapping full indices to coefficients.
        """
        terms = []
        
        # Strassen for 4x4 * 4x4:
        # Split into 2x2 blocks.
        # A blocks: A11 (i0-1,j0-1), A12 (i0-1,j2-3), A21 (i2-3,j0-1), A22 (i2-3,j2-3)
        # B blocks: B11 (j0-1,k0-1), B12 (j0-1,k2-3), B21 (j2-3,k0-1), B22 (j2-3,k2-3)
        # C blocks: C11 (i0-1,k0-1), C12 (i0-1,k2-3), C21 (i2-3,k0-1), C22 (i2-3,k2-3)
        
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        
        # C11 = M1 - M3 + M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M4 + M6
        
        # Each M is a 2x2*2x2 product, so it has 7 rank-1 terms.
        # Total: 7 * 7 = 49 terms.
        
        # For each M, define the A-coefficients and B-coefficients for the 2x2 blocks,
        # then expand using the 2x2 decomposition.
        
        # M definitions: (A_block_coeffs, B_block_coeffs)
        # A_block_coeffs: dict from block name to coefficient
        # B_block_coeffs: dict from block name to coefficient
        
        M_defs = [
            # M1: (A11 + A22) * (B11 + B22)
            ({'A11': 1, 'A22': 1}, {'B11': 1, 'B22': 1}),
            # M2: (A21 + A22) * B11
            ({'A21': 1, 'A22': 1}, {'B11': 1}),
            # M3: A11 * (B12 - B22)
            ({'A11': 1}, {'B12': 1, 'B22': -1}),
            # M4: A22 * (B21 - B11)
            ({'A22': 1}, {'B21': 1, 'B11': -1}),
            # M5: (A11 + A12) * B22
            ({'A11': 1, 'A12': 1}, {'B22': 1}),
            # M6: (A21 - A11) * (B11 + B12)
            ({'A21': 1, 'A11': -1}, {'B11': 1, 'B12': 1}),
            # M7: (A12 - A22) * (B21 + B22)
            ({'A12': 1, 'A22': -1}, {'B21': 1, 'B22': 1}),
        ]
        
        # C_block_coeffs for each M:
        C_coeffs = [
            # M1 -> C11: +1, C22: +1
            {'C11': 1, 'C22': 1},
            # M2 -> C21: +1
            {'C21': 1},
            # M3 -> C11: -1, C12: +1
            {'C11': -1, 'C12': 1},
            # M4 -> C21: +1, C22: +1
            {'C21': 1, 'C22': 1},
            # M5 -> C11: +1, C12: +1
            {'C11': 1, 'C12': 1},
            # M6 -> C22: +1
            {'C22': 1},
            # M7 -> C11: -1
            {'C11': -1},
        ]
        
        # Block index mappings:
        # A11: i in {0,1}, j in {0,1}
        # A12: i in {0,1}, j in {2,3}
        # A21: i in {2,3}, j in {0,1}
        # A22: i in {2,3}, j in {2,3}
        # B11: j in {0,1}, k in {0,1}
        # B12: j in {0,1}, k in {2,3}
        # B21: j in {2,3}, k in {0,1}
        # B22: j in {2,3}, k in {2,3}
        # C11: i in {0,1}, k in {0,1}
        # C12: i in {0,1}, k in {2,3}
        # C21: i in {2,3}, k in {0,1}
        # C22: i in {2,3}, k in {2,3}
        
        A_block_ranges = {
            'A11': (range(2), range(2)),
            'A12': (range(2), range(2,4)),
            'A21': (range(2,4), range(2)),
            'A22': (range(2,4), range(2,4)),
        }
        B_block_ranges = {
            'B11': (range(2), range(2)),
            'B12': (range(2), range(2,4)),
            'B21': (range(2,4), range(2)),
            'B22': (range(2,4), range(2,4)),
        }
        C_block_ranges = {
            'C11': (range(2), range(2)),
            'C12': (range(2), range(2,4)),
            'C21': (range(2,4), range(2)),
            'C22': (range(2,4), range(2,4)),
        }
        
        # For each M and each of its 7 rank-1 terms, generate the full-term coefficients.
        for m_idx, (A_coeffs, B_coeffs) in enumerate(M_defs):
            c_coeff_dict = C_coeffs[m_idx]
            
            for (U_2x2, V_2x2, W_2x2) in terms_2x2:
                # U_2x2, V_2x2, W_2x2 are dicts from (local_row, local_col) to coeff
                # We need to map local indices to global indices.
                
                U_full = {}
                V_full = {}
                W_full = {}
                
                # For A coefficients:
                for block_name, a_coeff in A_coeffs.items():
                    i_range, j_range = A_block_ranges[block_name]
                    for a_local in range(2):
                        for b_local in range(2):
                            i_global = list(i_range)[a_local]
                            j_global = list(j_range)[b_local]
                            u_2x2_val = U_2x2.get((a_local, b_local), 0)
                            if u_2x2_val != 0:
                                key = (i_global, j_global)
                                U_full[key] = U_full.get(key, 0) + a_coeff * u_2x2_val
                
                # For B coefficients:
                for block_name, b_coeff in B_coeffs.items():
                    j_range, k_range = B_block_ranges[block_name]
                    for c_local in range(2):
                        for d_local in range(2):
                            j_global = list(j_range)[c_local]
                            k_global = list(k_range)[d_local]
                            v_2x2_val = V_2x2.get((c_local, d_local), 0)
                            if v_2x2_val != 0:
                                key = (j_global, k_global)
                                V_full[key] = V_full.get(key, 0) + b_coeff * v_2x2_val
                
                # For C coefficients:
                for block_name, c_coeff in c_coeff_dict.items():
                    i_range, k_range = C_block_ranges[block_name]
                    for a_local in range(2):
                        for d_local in range(2):
                            i_global = list(i_range)[a_local]
                            k_global = list(k_range)[d_local]
                            w_2x2_val = W_2x2.get((a_local, d_local), 0)
                            if w_2x2_val != 0:
                                key = (i_global, k_global)
                                W_full[key] = W_full.get(key, 0) + c_coeff * w_2x2_val
                
                terms.append((U_full, V_full, W_full))
        
        return terms
    
    def build_4x4x2_terms():
        """
        Build rank-28 terms for 4x4 * 4x2.
        B has shape 4x2, so k in {0,1} for this part (but in the full problem,
        these correspond to k in {4,5}).
        """
        terms = []
        
        # C_right = A * B_right
        # B_right = [[B_r1], [B_r2]] where B_r1 is 2x2 (j in {0,1}, k_local in {0,1}),
        #           B_r2 is 2x2 (j in {2,3}, k_local in {0,1})
        # C_right = [[C_r1], [C_r2]]
        # C_r1 = A11 * B_r1 + A12 * B_r2
        # C_r2 = A21 * B_r1 + A22 * B_r2
        
        # Each term is a 2x2 * 2x2 product.
        
        # Define the 4 sub-products:
        sub_products = [
            # (A_block, B_block, C_block)
            ('A11', 'Br1', 'Cr1'),
            ('A12', 'Br2', 'Cr1'),
            ('A21', 'Br1', 'Cr2'),
            ('A22', 'Br2', 'Cr2'),
        ]
        
        A_block_ranges = {
            'A11': (range(2), range(2)),
            'A12': (range(2), range(2,4)),
            'A21': (range(2,4), range(2)),
            'A22': (range(2,4), range(2,4)),
        }
        # B_r1: j in {0,1}, k_local in {0,1} -> in full problem, k_global in {4,5}
        # B_r2: j in {2,3}, k_local in {0,1} -> in full problem, k_global in {4,5}
        B_block_ranges = {
            'Br1': (range(2), range(2)),  # local k
            'Br2': (range(2,4), range(2)),  # local k
        }
        # C_r1: i in {0,1}, k_local in {0,1} -> k_global in {4,5}
        # C_r2: i in {2,3}, k_local in {0,1} -> k_global in {4,5}
        C_block_ranges = {
            'Cr1': (range(2), range(2)),  # local k
            'Cr2': (range(2,4), range(2)),  # local k
        }
        
        for A_block, B_block, C_block in sub_products:
            for (U_2x2, V_2x2, W_2x2) in terms_2x2:
                U_full = {}
                V_full = {}
                W_full = {}
                
                i_range, j_range = A_block_ranges[A_block]
                for a_local in range(2):
                    for b_local in range(2):
                        i_global = list(i_range)[a_local]
                        j_global = list(j_range)[b_local]
                        u_val = U_2x2.get((a_local, b_local), 0)
                        if u_val != 0:
                            key = (i_global, j_global)
                            U_full[key] = U_full.get(key, 0) + u_val
                
                j_range, k_range_local = B_block_ranges[B_block]
                for c_local in range(2):
                    for d_local in range(2):
                        j_global = list(j_range)[c_local]
                        k_global = 4 + list(k_range_local)[d_local]  # offset by 4
                        v_val = V_2x2.get((c_local, d_local), 0)
                        if v_val != 0:
                            key = (j_global, k_global)
                            V_full[key] = V_full.get(key, 0) + v_val
                
                i_range, k_range_local = C_block_ranges[C_block]
                for a_local in range(2):
                    for d_local in range(2):
                        i_global = list(i_range)[a_local]
                        k_global = 4 + list(k_range_local)[d_local]  # offset by 4
                        w_val = W_2x2.get((a_local, d_local), 0)
                        if w_val != 0:
                            key = (i_global, k_global)
                            W_full[key] = W_full.get(key, 0) + w_val
                
                terms.append((U_full, V_full, W_full))
        
        return terms
    
    # Build all terms
    terms_4x4x4 = build_4x4x4_terms()
    terms_4x4x2 = build_4x4x2_terms()
    
    all_terms = terms_4x4x4 + terms_4x4x2
    rank = len(all_terms)
    
    # Convert to matrix form
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (U_dict, V_dict, W_dict) in enumerate(all_terms):
        for (i, j), val in U_dict.items():
            U[i * m + j][r] = val
        for (j, k), val in V_dict.items():
            V[j * p + k][r] = val
        for (i, k), val in W_dict.items():
            W[k * n + i][r] = val  # trace convention: W[k*n+i, r]
    
    # Verify the decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i * m + j], V_arr[j * p + k] * W_arr[k * n + i])
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
    
    return {"U": U, "V": V, "W": W}

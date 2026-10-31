"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Recursive Strassen decomposition for 4x4 matrix multiplication, rank 49.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    U: (16, 49), V: (16, 49), W: (16, 49)
    """
    n, m, p = DIMENSIONS
    assert (n, m, p) == (4, 4, 4)
    
    rank = 49
    # Initialize matrices with zeros
    # U[i*m+j, r] is coeff of A[i,j] in r-th term
    # V[j*p+k, r] is coeff of B[j,k] in r-th term
    # W[k*n+i, r] is coeff of r-th term in C[i,k]
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's algorithm for 2x2 matrices:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # We apply this recursively.
    # Level 0: 4x4 matrices. Split into 2x2 blocks of 2x2 matrices.
    # Level 1: 2x2 matrices. Split into 2x2 blocks of 1x1 matrices (scalars).
    
    # Helper function to add contributions to U, V, W for a specific term
    # a_coeffs: dict of (i, j) -> coeff for A
    # b_coeffs: dict of (j, k) -> coeff for B
    # c_coeffs: dict of (i, k) -> coeff for C
    # r: current rank index
    def add_term(r, a_coeffs, b_coeffs, c_coeffs):
        for (i, j), val in a_coeffs.items():
            if val != 0:
                U[i * m + j][r] += val
        for (j, k), val in b_coeffs.items():
            if val != 0:
                V[j * p + k][r] += val
        for (i, k), val in c_coeffs.items():
            if val != 0:
                W[k * n + i][r] += val

    def strassen_2x2(a_blocks, b_blocks, c_blocks, r_offset):
        """
        Recursively decompose 2x2 matrix multiplication.
        a_blocks, b_blocks, c_blocks are 2x2 arrays of linear combinations.
        Each element is a dict: {global_index: coeff}
        For level 1 (scalars), global_index is (i, j) for A, (j, k) for B, (i, k) for C.
        For level 0 (blocks), global_index is (block_i, block_j, local_i, local_j) etc.
        
        To simplify, we will define the linear combinations explicitly for the 4x4 case.
        """
        pass

    # Instead of deep recursion with dicts, let's just write out the 49 terms directly
    # based on the recursive structure.
    
    # Let A[i][j], B[j][k], C[i][k] be the scalar entries.
    # Indices: i, j, k in {0, 1, 2, 3}
    
    # Outer Strassen terms (7 terms)
    # Each outer term is a 2x2 matrix product of linear combinations of A and B blocks.
    # Let's define the 7 outer terms X_t, Y_t (2x2 matrices of linear combos of A, B)
    # and their contributions to C blocks Z_t.
    
    # A block (bi, bj) contains A[2*bi + i][2*bj + j] for i,j in {0,1}
    # B block (bi, bj) contains B[2*bi + i][2*bj + j] for i,j in {0,1}
    # C block (bi, bj) contains C[2*bi + i][2*bj + j] for i,j in {0,1}
    
    # Outer M1: X = A00+A11, Y = B00+B11
    # Outer M2: X = A10+A11, Y = B00
    # Outer M3: X = A00, Y = B01-B11
    # Outer M4: X = A11, Y = B10-B00
    # Outer M5: X = A00+A01, Y = B11
    # Outer M6: X = A10-A00, Y = B00+B01
    # Outer M7: X = A01-A11, Y = B10+B11
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each outer term t, X_t and Y_t are 2x2 matrices.
    # We compute Z_t = X_t * Y_t using inner Strassen (7 terms).
    # So total 7 * 7 = 49 terms.
    
    # Let's represent X_t and Y_t as 2x2 arrays of linear combinations.
    # A linear combination is a list of (i, j, coeff) for A, or (j, k, coeff) for B.
    
    def get_a_block(bi, bj):
        """Returns list of (i, j, 1) for the 4 entries in block (bi, bj) of A"""
        return [(2*bi + i, 2*bj + j, 1) for i in range(2) for j in range(2)]
    
    def get_b_block(bi, bj):
        """Returns list of (j, k, 1) for the 4 entries in block (bi, bj) of B"""
        return [(2*bi + i, 2*bj + j, 1) for i in range(2) for j in range(2)]
    
    def add_combos(c1, c2):
        d = {}
        for (i, j, v) in c1:
            d[(i, j)] = d.get((i, j), 0) + v
        for (i, j, v) in c2:
            d[(i, j)] = d.get((i, j), 0) + v
        return [(i, j, v) for (i, j), v in d.items() if v != 0]
        
    def sub_combos(c1, c2):
        return add_combos(c1, [(i, j, -v) for (i, j, v) in c2])
    
    # Define the 7 outer terms
    # X_t[i][j] is a linear combo of A entries
    # Y_t[i][j] is a linear combo of B entries
    
    A00 = get_a_block(0, 0)
    A01 = get_a_block(0, 1)
    A10 = get_a_block(1, 0)
    A11 = get_a_block(1, 1)
    
    B00 = get_b_block(0, 0)
    B01 = get_b_block(0, 1)
    B10 = get_b_block(1, 0)
    B11 = get_b_block(1, 1)
    
    outer_terms = []
    
    # M1: X = A00+A11, Y = B00+B11
    X = [[add_combos(A00, A11) for _ in range(2)] for _ in range(2)]
    Y = [[add_combos(B00, B11) for _ in range(2)] for _ in range(2)]
    outer_terms.append((X, Y, [(0,0,1), (1,1,1)])) # C00 += 1, C11 += 1
    
    # M2: X = A10+A11, Y = B00
    X = [[add_combos(A10, A11) for _ in range(2)] for _ in range(2)]
    Y = [[B00 for _ in range(2)] for _ in range(2)]
    outer_terms.append((X, Y, [(1,0,1), (1,1,-1)])) # C10 += 1, C11 -= 1
    
    # M3: X = A00, Y = B01-B11
    X = [[A00 for _ in range(2)] for _ in range(2)]
    Y = [[sub_combos(B01, B11) for _ in range(2)] for _ in range(2)]
    outer_terms.append((X, Y, [(0,1,1), (1,1,1)])) # C01 += 1, C11 += 1
    
    # M4: X = A11, Y = B10-B00
    X = [[A11 for _ in range(2)] for _ in range(2)]
    Y = [[sub_combos(B10, B00) for _ in range(2)] for _ in range(2)]
    outer_terms.append((X, Y, [(0,0,1), (1,0,1)])) # C00 += 1, C10 += 1
    
    # M5: X = A00+A01, Y = B11
    X = [[add_combos(A00, A01) for _ in range(2)] for _ in range(2)]
    Y = [[B11 for _ in range(2)] for _ in range(2)]
    outer_terms.append((X, Y, [(0,0,-1), (0,1,1)])) # C00 -= 1, C01 += 1
    
    # M6: X = A10-A00, Y = B00+B01
    X = [[sub_combos(A10, A00) for _ in range(2)] for _ in range(2)]
    Y = [[add_combos(B00, B01) for _ in range(2)] for _ in range(2)]
    outer_terms.append((X, Y, [(1,1,1)])) # C11 += 1
    
    # M7: X = A01-A11, Y = B10+B11
    X = [[sub_combos(A01, A11) for _ in range(2)] for _ in range(2)]
    Y = [[add_combos(B10, B11) for _ in range(2)] for _ in range(2)]
    outer_terms.append((X, Y, [(0,0,1)])) # C00 += 1
    
    col = 0
    for t in range(7):
        X, Y, c_targets = outer_terms[t]
        
        # Inner Strassen on X*Y
        # Inner M1: (X00+X11)(Y00+Y11) -> Z00 += 1, Z11 += 1
        X_in = add_combos(X[0][0], X[1][1])
        Y_in = add_combos(Y[0][0], Y[1][1])
        for (br, bc, coeff) in c_targets:
            # Z[br][bc] is a 2x2 block.
            # Z00 local (0,0) -> Global C[2*br+0][2*bc+0]
            # Z00 local (0,1) -> Global C[2*br+0][2*bc+1]
            # Z00 local (1,0) -> Global C[2*br+1][2*bc+0]
            # Z00 local (1,1) -> Global C[2*br+1][2*bc+1]
            # Inner M1 contributes to Z00 and Z11.
            # So for C block (br, bc), we add to local (0,0) and (1,1) of Z00?
            # No, Inner M1 is the top-left element of Z.
            # Wait. Z = X*Y.
            # Z00 = M1 + M4 - M5 + M7 (Inner)
            # Z11 = M1 - M2 + M3 + M6 (Inner)
            # So Inner M1 contributes +1 to Z00 and +1 to Z11.
            # Z00 is the top-left 2x2 of the 4x4 block result?
            # No, X and Y are 2x2 matrices of blocks.
            # Z is a 2x2 matrix of blocks.
            # Z[0][0] is the top-left block of C.
            # Inner M1 is a scalar (linear combo) that forms part of Z[0][0] and Z[1][1].
            # Specifically, Z[0][0] = Inner_M1 + ...
            # So Inner_M1 is added to the entire block Z[0][0]?
            # No. X and Y entries are 2x2 matrices (blocks).
            # X[0][0] is a 2x2 matrix.
            # Inner M1 = (X[0][0] + X[1][1]) * (Y[0][0] + Y[1][1]).
            # This is a 2x2 matrix product.
            # The result is a 2x2 matrix.
            # This 2x2 matrix is added to Z[0][0] and Z[1][1].
            # So we need to decompose this 2x2 matrix product into 7 rank-1 terms.
            # Each rank-1 term corresponds to one global rank-1 term in the 4x4 decomposition.
            
            pass # Placeholder, logic below
            
        # Correct Logic:
        # For each inner term, we have a 2x2 matrix X_in and Y_in.
        # We add 7 rank-1 terms to U, V, W.
        # The W entries depend on which Z block (br, bc) and which local position.
        
        # Inner M1: X_in = X[0][0] + X[1][1], Y_in = Y[0][0] + Y[1][1]
        # Contributes to Z[0][0] and Z[1][1].
        # For each (br, bc, coeff) in c_targets:
        #   Z[br][bc] is a 2x2 block.
        #   Inner M1 contributes to local (0,0) and (1,1) of this block?
        #   No. Inner M1 is the (0,0) element of the 2x2 result Z.
        #   So it contributes to the entire 2x2 block Z[br][bc]?
        #   Wait. X and Y are 2x2 arrays of 2x2 matrices.
        #   Z is a 2x2 array of 2x2 matrices.
        #   Z[i][j] = sum_k X[i][k] Y[k][j].
        #   Inner Strassen computes Z[i][j] for i,j in {0,1}.
        #   Inner M1 = (X[0][0]+X[1][1])(Y[0][0]+Y[1][1]).
        #   Z[0][0] += Inner M1.
        #   Z[1][1] += Inner M1.
        #   So Inner M1 is a 2x2 matrix.
        #   We add this 2x2 matrix to Z[0][0] and Z[1][1].
        #   This means for each global C entry in block Z[0][0] and Z[1][1], we add the corresponding entry from Inner M1.
        #   But Inner M1 itself is decomposed into 7 rank-1 terms.
        #   So for each of the 7 rank-1 terms of Inner M1, we add to U, V.
        #   And for W, we add to the corresponding positions in Z[0][0] and Z[1][1].
        
        # Let's define the 7 inner terms explicitly.
        
        # Inner M1
        X_in = add_combos(X[0][0], X[1][1])
        Y_in = add_combos(Y[0][0], Y[1][1])
        # Z[0][0] += X_in * Y_in, Z[1][1] += X_in * Y_in
        # We need to decompose X_in * Y_in (2x2 product) into 7 terms.
        # But X_in and Y_in are 2x2 matrices of linear combos.
        # This is getting recursive.
        # Let's just iterate the 7 inner Strassen steps for the 2x2 product.
        
        inner_terms = []
        # M1
        inner_terms.append((add_combos(X[0][0], X[1][1]), add_combos(Y[0][0], Y[1][1]), [(0,0,1), (1,1,1)]))
        # M2
        inner_terms.append((add_combos(X[1][0], X[1][1]), Y[0][0], [(1,0,1), (1,1,-1)]))
        # M3
        inner_terms.append((X[0][0], sub_combos(Y[0][1], Y[1][1]), [(0,1,1), (1,1,1)]))
        # M4
        inner_terms.append((X[1][1], sub_combos(Y[1][0], Y[0][0]), [(0,0,1), (1,0,1)]))
        # M5
        inner_terms.append((add_combos(X[0][0], X[0][1]), Y[1][1], [(0,0,-1), (0,1,1)]))
        # M6
        inner_terms.append((sub_combos(X[1][0], X[0][0]), add_combos(Y[0][0], Y[0][1]), [(1,1,1)]))
        # M7
        inner_terms.append((sub_combos(X[0][1], X[1][1]), add_combos(Y[1][0], Y[1][1]), [(0,0,1)]))
        
        for (Xi, Yi, zi_targets) in inner_terms:
            # Xi is a 2x2 matrix of linear combos of A.
            # Yi is a 2x2 matrix of linear combos of B.
            # We need to form the scalar products.
            # Actually, Xi and Yi are already 2x2 matrices.
            # The product Xi * Yi is a 2x2 matrix.
            # Each entry (i, j) of this product is a sum of 2 products of linear combos.
            # This is not a single rank-1 term.
            # Wait. The "rank-1 term" in the tensor decomposition corresponds to a single multiplication of two linear combinations.
            # In Strassen, each "M" is a multiplication of two linear combinations.
            # So M1 = (X00+X11) * (Y00+Y11).
            # Here X00 is a 2x2 matrix (block).
            # So M1 is a 2x2 matrix product.
            # This 2x2 matrix product is NOT a rank-1 term in the 4x4 tensor.
            # The 4x4 tensor rank-1 terms are scalar multiplications.
            # So we must decompose the 2x2 matrix product M1 into 7 scalar multiplications using Strassen again?
            # Yes. That's why it's 7*7=49.
            
            # So for each inner term (which is a 2x2 matrix product), we apply Strassen 2x2.
            # This will generate 7 scalar rank-1 terms.
            
            # Let's define a helper to add a 2x2 Strassen decomposition.
            # X_block: 2x2 array of linear combos of A
            # Y_block: 2x2 array of linear combos of B
            # C_block_targets: list of (block_i, block_j, coeff) for the 4x4 C blocks
            # r: current column index
            
            # We need to map the local 2x2 indices to global 4x4 indices.
            # X_block[i][j] is a linear combo of A entries.
            # Y_block[i][j] is a linear combo of B entries.
            
            # Strassen 2x2 on X_block * Y_block:
            # M1 = (X00+X11)(Y00+Y11)
            # ...
            # Z00 = M1 + M4 - M5 + M7
            # Z01 = M3 + M5
            # Z10 = M2 + M4
            # Z11 = M1 - M2 + M3 + M6
            
            # Each M is a scalar product of two linear combos.
            # So we add 7 terms to U, V.
            # For W, we add to the corresponding global C entries.
            
            # Let's implement this helper.
            
            def add_2x2_strassen(X_block, Y_block, c_block_targets, r):
                # M1
                X_m1 = add_combos(X_block[0][0], X_block[1][1])
                Y_m1 = add_combos(Y_block[0][0], Y_block[1][1])
                # Z00 += M1, Z11 += M1
                # c_block_targets gives which 4x4 blocks get this 2x2 result.
                # For each (bi, bj, coeff) in c_block_targets:
                #   The 2x2 result Z is added to C[2*bi:2*bi+2, 2*bj:2*bj+2].
                #   Z[0][0] -> C[2*bi, 2*bj]
                #   Z[0][1] -> C[2*bi, 2*bj+1]
                #   Z[1][0] -> C[2*bi+1, 2*bj]
                #   Z[1][1] -> C[2*bi+1, 2*bj+1]
                
                # M1 contributes to Z[0][0] and Z[1][1].
                for (bi, bj, coeff) in c_block_targets:
                    # Z[0][0]
                    W[(2*bj) * n + (2*bi)][r] += coeff
                    # Z[1][1]
                    W[(2*bj+1) * n + (2*bi+1)][r] += coeff
                
                for (i, j, val) in X_m1:
                    U[i * m + j][r] += val
                for (j, k, val) in Y_m1:
                    V[j * p + k][r] += val
                
                # M2
                X_m2 = add_combos(X_block[1][0], X_block[1][1])
                Y_m2 = Y_block[0][0]
                # Z10 += M2, Z11 -= M2
                for (bi, bj, coeff) in c_block_targets:
                    # Z[1][0]
                    W[(2*bj) * n + (2*bi+1)][r] += coeff
                    # Z[1][1]
                    W[(2*bj+1) * n + (2*bi+1)][r] -= coeff
                for (i, j, val) in X_m2:
                    U[i * m + j][r] += val
                for (j, k, val) in Y_m2:
                    V[j * p + k][r] += val
                
                # M3
                X_m3 = X_block[0][0]
                Y_m3 = sub_combos(Y_block[0][1], Y_block[1][1])
                # Z01 += M3, Z11 += M3
                for (bi, bj, coeff) in c_block_targets:
                    # Z[0][1]
                    W[(2*bj+1) * n + (2*bi)][r] += coeff
                    # Z[1][1]
                    W[(2*bj+1) * n + (2*bi+1)][r] += coeff
                for (i, j, val) in X_m3:
                    U[i * m + j][r] += val
                for (j, k, val) in Y_m3:
                    V[j * p + k][r] += val
                
                # M4
                X_m4 = X_block[1][1]
                Y_m4 = sub_combos(Y_block[1][0], Y_block[0][0])
                # Z00 += M4, Z10 += M4
                for (bi, bj, coeff) in c_block_targets:
                    # Z[0][0]
                    W[(2*bj) * n + (2*bi)][r] += coeff
                    # Z[1][0]
                    W[(2*bj) * n + (2*bi+1)][r] += coeff
                for (i, j, val) in X_m4:
                    U[i * m + j][r] += val
                for (j, k, val) in Y_m4:
                    V[j * p + k][r] += val
                
                # M5
                X_m5 = add_combos(X_block[0][0], X_block[0][1])
                Y_m5 = Y_block[1][1]
                # Z00 -= M5, Z01 += M5
                for (bi, bj, coeff) in c_block_targets:
                    # Z[0][0]
                    W[(2*bj) * n + (2*bi)][r] -= coeff
                    # Z[0][1]
                    W[(2*bj+1) * n + (2*bi)][r] += coeff
                for (i, j, val) in X_m5:
                    U[i * m + j][r] += val
                for (j, k, val) in Y_m5:
                    V[j * p + k][r] += val
                
                # M6
                X_m6 = sub_combos(X_block[1][0], X_block[0][0])
                Y_m6 = add_combos(Y_block[0][0], Y_block[0][1])
                # Z11 += M6
                for (bi, bj, coeff) in c_block_targets:
                    # Z[1][1]
                    W[(2*bj+1) * n + (2*bi+1)][r] += coeff
                for (i, j, val) in X_m6:
                    U[i * m + j][r] += val
                for (j, k, val) in Y_m6:
                    V[j * p + k][r] += val
                
                # M7
                X_m7 = sub_combos(X_block[0][1], X_block[1][1])
                Y_m7 = add_combos(Y_block[1][0], Y_block[1][1])
                # Z00 += M7
                for (bi, bj, coeff) in c_block_targets:
                    # Z[0][0]
                    W[(2*bj) * n + (2*bi)][r] += coeff
                for (i, j, val) in X_m7:
                    U[i * m + j][r] += val
                for (j, k, val) in Y_m7:
                    V[j * p + k][r] += val

            # Now call add_2x2_strassen for each inner term
            for (Xi, Yi, zi_targets) in inner_terms:
                add_2x2_strassen(Xi, Yi, zi_targets, col)
                col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

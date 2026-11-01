"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Recursive Strassen: 4x4 = 2x2 blocks, each 2x2 uses Strassen (7 products).
    Total rank = 7 * 7 = 49.
    Convention: W[k*n + i, c] for 4x4*4x4 -> 4x4.
    """
    n, m, p = DIMENSIONS
    rank = 49

    # U is (n*m, rank) = (16, 49)
    # V is (m*p, rank) = (16, 49)
    # W is (p*n, rank) = (16, 49)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen for 2x2 matrices:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # For 4x4, split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each block is 2x2
    # B = [[B11, B12], [B21, B22]] where each block is 2x2
    # C11 = A11*B11 + A12*B21 (2x2 each, so 4 scalar products per block product)
    # But we use Strassen on the 2x2 block level AND within each 2x2 sub-product.

    # Outer Strassen on 2x2 blocks (each block is 2x2):
    # M1_outer = (A11 + A22)(B11 + B22)  -> each is a 2x2 matrix sum
    # M2_outer = (A21 + A22) * B11
    # M3_outer = A11 * (B12 - B22)
    # M4_outer = A22 * (B21 - B11)
    # M5_outer = (A11 + A12) * B22
    # M6_outer = (A21 - A11) * (B11 + B12)
    # M7_outer = (A12 - A22) * (B21 + B22)

    # C11 = M1_outer + M4_outer - M5_outer + M7_outer
    # C12 = M3_outer + M5_outer
    # C21 = M2_outer + M4_outer
    # C22 = M1_outer - M2_outer + M3_outer + M6_outer

    # Each "product" like (A11+A22)(B11+B22) is a 2x2 * 2x2 product,
    # which we do with inner Strassen (7 scalar products).

    # Let's define the inner Strassen combinations for 2x2 matrices.
    # For matrices X = [[x1, x2], [x3, x4]] and Y = [[y1, y2], [y3, y4]]:
    # s1 = x1 + x4, s2 = y1 + y4  -> M1 = s1 * s2
    # s3 = x3 + x4, s4 = y1       -> M2 = s3 * s4
    # s5 = x1, s6 = y2 - y4       -> M3 = s5 * s6
    # s7 = x4, s8 = y3 - y1       -> M4 = s7 * s8
    # s9 = x1 + x2, s10 = y4      -> M5 = s9 * s10
    # s11 = x3 - x1, s12 = y1 + y2 -> M6 = s11 * s12
    # s13 = x2 - x4, s14 = y3 + y4 -> M7 = s13 * s14
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # Now for the outer level, each "product" is a 2x2*2x2 which we expand
    # into 7 scalar products using inner Strassen.

    # Let's build the U, V, W matrices column by column (each column = one rank term).

    # For each outer Strassen term t (1..7), and each inner Strassen term s (1..7),
    # we have one rank column.

    # U[row, col]: coefficient for the (i,j) element of A in rank term (t,s)
    # V[row, col]: coefficient for the (j,k) element of B in rank term (t,s)
    # W[row, col]: coefficient for the (k,i) element of C in rank term (t,s)

    # Let's define helper to get 2x2 block indices.
    # For 4x4 matrix indexed 0..15 as row*4+col:
    # Block (br, bc) where br,bc in {0,1} covers rows br*2..br*2+1, cols bc*2..bc*2+1
    # Element (r,c) within block: global row = br*2+r, global col = bc*2+c
    # Global index = (br*2+r)*4 + (bc*2+c)

    # For A: element (i,j) -> U[i*4+j, col]
    # For B: element (j,k) -> V[j*4+k, col]
    # For C: element (i,j) -> W[j*4+i, col]  (since W[k*n+i] with k=j, i=i)

    # Inner Strassen for 2x2*2x2:
    # Given X = sum of alpha * A_block and Y = sum of beta * B_block,
    # where A_block is a 2x2 block of A, B_block is a 2x2 block of B.
    # The 7 products are:
    # P1 = (X[0,0]+X[1,1]) * (Y[0,0]+Y[1,1])
    # P2 = (X[1,0]+X[1,1]) * Y[0,0]
    # P3 = X[0,0] * (Y[0,1]-Y[1,1])
    # P4 = X[1,1] * (Y[1,0]-Y[0,0])
    # P5 = (X[0,0]+X[0,1]) * Y[1,1]
    # P6 = (X[1,0]-X[0,0]) * (Y[0,0]+Y[0,1])
    # P7 = (X[0,1]-X[1,1]) * (Y[1,0]+Y[1,1])

    # For each product, we expand X[r1,c1] and Y[r2,c2] into linear combinations
    # of A and B elements.

    # Let's define the outer Strassen terms:
    # t=1: X = A11+A22, Y = B11+B22
    # t=2: X = A21+A22, Y = B11
    # t=3: X = A11, Y = B12-B22
    # t=4: X = A22, Y = B21-B11
    # t=5: X = A11+A12, Y = B22
    # t=6: X = A21-A11, Y = B11+B12
    # t=7: X = A12-A22, Y = B21+B22

    # For each (t, s), we compute the U, V contributions.

    # Block positions:
    # A11: block (0,0) -> rows 0,1; cols 0,1
    # A12: block (0,1) -> rows 0,1; cols 2,3
    # A21: block (1,0) -> rows 2,3; cols 0,1
    # A22: block (1,1) -> rows 2,3; cols 2,3

    # Same for B.

    # Let's define for each outer term t, the list of (block, coeff) for X and Y.
    # block = (br, bc), coeff = 1 or -1

    outer_terms = {
        1: {'X': [((0,0), 1), ((1,1), 1)], 'Y': [((0,0), 1), ((1,1), 1)]},
        2: {'X': [((1,0), 1), ((1,1), 1)], 'Y': [((0,0), 1)]},
        3: {'X': [((0,0), 1)], 'Y': [((0,1), 1), ((1,1), -1)]},
        4: {'X': [((1,1), 1)], 'Y': [((1,0), 1), ((0,0), -1)]},
        5: {'X': [((0,0), 1), ((0,1), 1)], 'Y': [((1,1), 1)]},
        6: {'X': [((1,0), 1), ((0,0), -1)], 'Y': [((0,0), 1), ((0,1), 1)]},
        7: {'X': [((0,1), 1), ((1,1), -1)], 'Y': [((1,0), 1), ((1,1), 1)]},
    }

    # Inner Strassen products (s=1..7):
    # Each product is of the form: (sum of X elements) * (sum of Y elements)
    # where X elements are from the X 2x2 matrix, Y elements from Y 2x2 matrix.

    # P1: (X00 + X11) * (Y00 + Y11)
    # P2: (X10 + X11) * Y00
    # P3: X00 * (Y01 - Y11)
    # P4: X11 * (Y10 - Y00)
    # P5: (X00 + X01) * Y11
    # P6: (X10 - X00) * (Y00 + Y01)
    # P7: (X01 - X11) * (Y10 + Y11)

    # For each product, we need to expand into contributions to U and V.

    # X[r1, c1] is a linear combination of A elements from the blocks in outer_terms[t]['X']
    # Y[r2, c2] is a linear combination of B elements from the blocks in outer_terms[t]['Y']

    # Let's precompute for each (t, r1, c1) the list of (global_i, global_j, coeff) for A
    # and for each (t, r2, c2) the list of (global_j, global_k, coeff) for B

    # For X: the 2x2 matrix X has elements X[0,0], X[0,1], X[1,0], X[1,1]
    # X[r1, c1] = sum over (block, coeff) in outer_terms[t]['X'] of coeff * A[block_row + r1, block_col + c1]
    # where block_row = br*2, block_col = bc*2

    # Similarly for Y.

    # Let's build U and V column by column.
    col = 0
    for t in range(1, 8):
        X_terms = outer_terms[t]['X']
        Y_terms = outer_terms[t]['Y']

        # For each inner Strassen product s (1..7):
        inner_products = [
            # P1: (X00 + X11) * (Y00 + Y11)
            ([(0,0), (1,1)], [(0,0), (1,1)]),
            # P2: (X10 + X11) * Y00
            ([(1,0), (1,1)], [(0,0)]),
            # P3: X00 * (Y01 - Y11)
            ([(0,0)], [(0,1), (1,1)]),
            # P4: X11 * (Y10 - Y00)
            ([(1,1)], [(1,0), (0,0)]),
            # P5: (X00 + X01) * Y11
            ([(0,0), (0,1)], [(1,1)]),
            # P6: (X10 - X00) * (Y00 + Y01)
            ([(1,0), (0,0)], [(0,0), (0,1)]),
            # P7: (X01 - X11) * (Y10 + Y11)
            ([(0,1), (1,1)], [(1,0), (1,1)]),
        ]

        # Signs for each product:
        # P3: Y part has (Y01 - Y11) so second term is -1
        # P4: Y part has (Y10 - Y00) so second term is -1
        # P6: X part has (X10 - X00) so second term is -1
        # P7: X part has (X01 - X11) so second term is -1

        x_signs = [
            [1, 1],  # P1
            [1, 1],  # P2
            [1],      # P3
            [1],      # P4
            [1, 1],  # P5
            [1, -1],  # P6
            [1, -1],  # P7
        ]

        y_signs = [
            [1, 1],  # P1
            [1],      # P2
            [1, -1],  # P3
            [1, -1],  # P4
            [1],      # P5
            [1, 1],  # P6
            [1, 1],  # P7
        ]

        for s in range(7):
            col = (t - 1) * 7 + s
            x_elems = inner_products[s][0]
            y_elems = inner_products[s][1]
            xs = x_signs[s]
            ys = y_signs[s]

            # For each X element in this product, add to U
            for (r1, c1) in x_elems:
                x_coeff = xs[x_elems.index((r1, c1))]
                for (br, bc), a_coeff in X_terms:
                    # A element at (br*2 + r1, bc*2 + c1)
                    global_i = br * 2 + r1
                    global_j = bc * 2 + c1
                    U[global_i * 4 + global_j][col] += x_coeff * a_coeff

            # For each Y element in this product, add to V
            for (r2, c2) in y_elems:
                y_coeff = ys[y_elems.index((r2, c2))]
                for (br, bc), b_coeff in Y_terms:
                    # B element at (br*2 + r2, bc*2 + c2)
                    global_j = br * 2 + r2
                    global_k = bc * 2 + c2
                    V[global_j * 4 + global_k][col] += y_coeff * b_coeff

    # Now compute W.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # Each Mt is a 2x2 matrix.
    # C11[0,0] = M1[0,0] + M4[0,0] - M5[0,0] + M7[0,0]
    # etc.

    # For each outer term t, and each inner product s, the contribution to C
    # is determined by the inner Strassen reconstruction.

    # Inner Strassen reconstruction for 2x2:
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6

    # So for each (t, s), the contribution to C blocks is:
    # If s=1 (P1): C11 += 1, C22 += 1
    # If s=2 (P2): C21 += 1, C22 -= 1
    # If s=3 (P3): C12 += 1, C22 += 1
    # If s=4 (P4): C11 += 1, C21 += 1
    # If s=5 (P5): C11 -= 1, C12 += 1
    # If s=6 (P6): C22 += 1
    # If s=7 (P7): C11 += 1

    inner_w = {
        1: {0: 1, 3: 1},   # P1: C11 +1, C22 +1
        2: {1: 1, 3: -1},  # P2: C21 +1, C22 -1
        3: {2: 1, 3: 1},   # P3: C12 +1, C22 +1
        4: {0: 1, 1: 1},   # P4: C11 +1, C21 +1
        5: {0: -1, 2: 1},  # P5: C11 -1, C12 +1
        6: {3: 1},         # P6: C22 +1
        7: {0: 1},         # P7: C11 +1
    }

    # Outer Strassen reconstruction for 4x4:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # So for each outer term t:
    # t=1 (M1): C11 +1, C22 +1
    # t=2 (M2): C21 +1, C22 -1
    # t=3 (M3): C12 +1, C22 +1
    # t=4 (M4): C11 +1, C21 +1
    # t=5 (M5): C11 -1, C12 +1
    # t=6 (M6): C22 +1
    # t=7 (M7): C11 +1

    outer_w = {
        1: {0: 1, 3: 1},   # M1: C11 +1, C22 +1
        2: {1: 1, 3: -1},  # M2: C21 +1, C22 -1
        3: {2: 1, 3: 1},   # M3: C12 +1, C22 +1
        4: {0: 1, 1: 1},   # M4: C11 +1, C21 +1
        5: {0: -1, 2: 1},  # M5: C11 -1, C12 +1
        6: {3: 1},         # M6: C22 +1
        7: {0: 1},         # M7: C11 +1
    }

    # For each (t, s), the W contribution is the product of outer_w[t] and inner_w[s].
    # The block position in C is determined by t (which outer term) and the inner position
    # from inner_w[s].

    # Wait, the outer term t determines which 2x2 block of C gets the contribution,
    # and the inner term s determines which element within that 2x2 block.

    # Actually, each Mt is a full 2x2 matrix, so for a given t, the inner Strassen
    # produces a 2x2 matrix Mt. Then the outer combination places Mt into the
    # appropriate block of C.

    # So for (t, s):
    # The 2x2 block of C is determined by t's outer_w contribution.
    # The element within that 2x2 block is determined by s's inner_w contribution.

    # For example, if t=1 and s=1:
    # outer_w[1] = {0: 1, 3: 1} means C11 += 1, C22 += 1
    # inner_w[1] = {0: 1, 3: 1} means within the 2x2, element (0,0) += 1, element (1,1) += 1
    # So C11[0,0] += 1, C11[1,1] += 1, C22[0,0] += 1, C22[1,1] += 1

    # For t=1, s=2:
    # outer_w[1] = {0: 1, 3: 1}
    # inner_w[2] = {1: 1, 3: -1} means within the 2x2, element (0,1) += 1, element (1,1) -= 1
    # So C11[0,1] += 1, C11[1,1] -= 1, C22[0,1] += 1, C22[1,1] -= 1

    for t in range(1, 8):
        for s in range(1, 8):
            col = (t - 1) * 7 + (s - 1)
            for c_block, c_coeff in outer_w[t].items():
                for e_idx, e_coeff in inner_w[s].items():
                    # c_block: 0=C11, 1=C21, 2=C12, 3=C22
                    # e_idx: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
                    # Convert c_block to (br, bc):
                    # C11: (0,0), C12: (0,1), C21: (1,0), C22: (1,1)
                    block_map = {0: (0,0), 1: (1,0), 2: (0,1), 3: (1,1)}
                    br, bc = block_map[c_block]
                    # e_idx to (r, c):
                    elem_map = {0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)}
                    r, c = elem_map[e_idx]
                    # Global C element: (br*2+r, bc*2+c)
                    global_i = br * 2 + r
                    global_j = bc * 2 + c
                    # W index: k*n + i = global_j * 4 + global_i
                    W[global_j * 4 + global_i][col] += c_coeff * e_coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

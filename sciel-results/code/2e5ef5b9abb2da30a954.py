import numpy as np
import time

def construct():
    n, m, p = 4, 4, 7
    deadline = time.time() + 140

    # Matrix multiplication: C = A * B
    # A is 4x4 (n x m), B is 4x7 (m x p), C is 4x7 (n x p)
    # T[i,j,k] = 1 for all valid (i,j,k)
    #
    # Rank-1 term: u ⊗ v ⊗ w where:
    #   u indexed by (i,j) -> size n*m = 16
    #   v indexed by (j,k) -> size m*p = 28
    #   w indexed by (k,i) -> size p*n = 28
    #
    # We need: sum_c U[c, i*m+j] * V[c, j*p+k] * W[c, k*n+i] = 1 for all i,j,k
    #
    # Strategy:
    # - For the 4x4 * 4x4 part (k=0,1,2,3): use recursive Strassen (rank 49)
    # - For the 4x4 * 4x3 part (k=4,5,6): use schoolbook (rank 48)
    # Total: 49 + 48 = 97
    #
    # Recursive Strassen for 4x4*4x4:
    # Split A into 2x2 blocks of 2x2 submatrices:
    #   A = [[A11, A12], [A21, A22]]
    #   B = [[B11, B12], [B21, B22]]
    #
    # Strassen computes 7 products:
    #   M1 = (A11+A22)*(B11+B22)
    #   M2 = (A21+A22)*B11
    #   M3 = A11*(B12-B22)
    #   M4 = A22*(B21-B11)
    #   M5 = (A11+A12)*B22
    #   M6 = (A21-A11)*(B11+B12)
    #   M7 = (A12-A22)*(B21+B22)
    #
    # Each of these is a 2x2*2x2 multiplication, which we decompose with
    # the 7-term Strassen for 2x2*2x2.
    #
    # Total: 7 * 7 = 49 terms for 4x4*4x4.

    terms = []  # list of (u, v, w) triples

    def strassen_2x2(i_off, j_off, k_off):
        """
        Generate 7 Strassen terms for a 2x2*2x2 block multiplication.
        i_off, j_off, k_off are the top-left offsets in the full matrices.
        The 2x2 blocks span:
          A: rows [i_off, i_off+1], cols [j_off, j_off+1]
          B: rows [j_off, j_off+1], cols [k_off, k_off+1]
          C: rows [i_off, i_off+1], cols [k_off, k_off+1]
        """
        result = []

        # M1 = (A11+A22)*(B11+B22)
        # A11 at (i_off, j_off), A22 at (i_off+1, j_off+1)
        # B11 at (j_off, k_off), B22 at (j_off+1, k_off+1)
        # C11 += M1, C22 += M1
        u = [0] * (n * m)
        u[i_off * m + j_off] = 1
        u[(i_off+1) * m + (j_off+1)] = 1
        v = [0] * (m * p)
        v[j_off * p + k_off] = 1
        v[(j_off+1) * p + (k_off+1)] = 1
        w = [0] * (p * n)
        w[k_off * n + i_off] = 1
        w[(k_off+1) * n + (i_off+1)] = 1
        result.append((u[:], v[:], w[:]))

        # M2 = (A21+A22)*B11
        # A21 at (i_off+1, j_off), A22 at (i_off+1, j_off+1)
        # B11 at (j_off, k_off)
        # C21 += M2, C22 -= M2
        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = 1
        u[(i_off+1) * m + (j_off+1)] = 1
        v = [0] * (m * p)
        v[j_off * p + k_off] = 1
        w = [0] * (p * n)
        w[k_off * n + (i_off+1)] = 1
        w[(k_off+1) * n + (i_off+1)] = -1
        result.append((u[:], v[:], w[:]))

        # M3 = A11*(B12-B22)
        # A11 at (i_off, j_off)
        # B12 at (j_off, k_off+1), B22 at (j_off+1, k_off+1)
        # C12 += M3, C22 += M3
        u = [0] * (n * m)
        u[i_off * m + j_off] = 1
        v = [0] * (m * p)
        v[j_off * p + (k_off+1)] = 1
        v[(j_off+1) * p + (k_off+1)] = -1
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = 1
        w[(k_off+1) * n + (i_off+1)] = 1
        result.append((u[:], v[:], w[:]))

        # M4 = A22*(B21-B11)
        # A22 at (i_off+1, j_off+1)
        # B21 at (j_off+1, k_off), B11 at (j_off, k_off)
        # C11 += M4, C21 += M4
        u = [0] * (n * m)
        u[(i_off+1) * m + (j_off+1)] = 1
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = 1
        v[j_off * p + k_off] = -1
        w = [0] * (p * n)
        w[k_off * n + i_off] = 1
        w[k_off * n + (i_off+1)] = 1
        result.append((u[:], v[:], w[:]))

        # M5 = (A11+A12)*B22
        # A11 at (i_off, j_off), A12 at (i_off, j_off+1)
        # B22 at (j_off+1, k_off+1)
        # C12 += M5, C11 -= M5
        u = [0] * (n * m)
        u[i_off * m + j_off] = 1
        u[i_off * m + (j_off+1)] = 1
        v = [0] * (m * p)
        v[(j_off+1) * p + (k_off+1)] = 1
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = 1
        w[k_off * n + i_off] = -1
        result.append((u[:], v[:], w[:]))

        # M6 = (A21-A11)*(B11+B12)
        # A21 at (i_off+1, j_off), A11 at (i_off, j_off)
        # B11 at (j_off, k_off), B12 at (j_off, k_off+1)
        # C22 += M6
        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = 1
        u[i_off * m + j_off] = -1
        v = [0] * (m * p)
        v[j_off * p + k_off] = 1
        v[j_off * p + (k_off+1)] = 1
        w = [0] * (p * n)
        w[(k_off+1) * n + (i_off+1)] = 1
        result.append((u[:], v[:], w[:]))

        # M7 = (A12-A22)*(B21+B22)
        # A12 at (i_off, j_off+1), A22 at (i_off+1, j_off+1)
        # B21 at (j_off+1, k_off), B22 at (j_off+1, k_off+1)
        # C11 += M7
        u = [0] * (n * m)
        u[i_off * m + (j_off+1)] = 1
        u[(i_off+1) * m + (j_off+1)] = -1
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = 1
        v[(j_off+1) * p + (k_off+1)] = 1
        w = [0] * (p * n)
        w[k_off * n + i_off] = 1
        result.append((u[:], v[:], w[:]))

        return result

    # Recursive Strassen for 4x4*4x4
    # The 4x4 blocks are:
    # A11: i_off=0, j_off=0
    # A12: i_off=0, j_off=2
    # A21: i_off=2, j_off=0
    # A22: i_off=2, j_off=2
    # B11: j_off=0, k_off=0
    # B12: j_off=0, k_off=2
    # B21: j_off=2, k_off=0
    # B22: j_off=2, k_off=2

    # M1 = (A11+A22)*(B11+B22)
    # This is a 2x2*2x2 multiplication where:
    #   A' = A11+A22 (sum of two 2x2 blocks)
    #   B' = B11+B22
    # But we can't directly add blocks in the tensor decomposition.
    # Instead, we need to express each product as a sum of rank-1 terms.
    #
    # Actually, the correct approach for recursive Strassen in tensor decomposition:
    # Each of the 7 Strassen products is itself a 2x2*2x2 multiplication.
    # We decompose each one using the 7-term Strassen for 2x2*2x2.
    #
    # But the inputs to each 2x2*2x2 are linear combinations of the original
    # 2x2 blocks. In the tensor decomposition, this means:
    #
    # For M1 = (A11+A22)*(B11+B22):
    # The "A" side has entries from A11 and A22 (both with coefficient 1)
    # The "B" side has entries from B11 and B22 (both with coefficient 1)
    #
    # This is equivalent to: A' * B' where A' is a 2x2 matrix whose (r,c) entry
    # is A[2r, 2c] + A[2r+2, 2c+2] and B' is a 2x2 matrix whose (r,c) entry
    # is B[2r, 2c] + B[2r+2, 2c+2].
    #
    # In the tensor decomposition, we can handle this by having the u-vector
    # have non-zero entries at both the A11 and A22 positions, and similarly
    # for v.
    #
    # So for M1, we apply strassen_2x2 but with a "blended" offset.
    # Actually, it's simpler to think of it this way:
    #
    # M1 involves products of (A[i,j] + A[i+2,j+2]) * (B[j,k] + B[j+2,k+2])
    # for i,j,k in {0,1}.
    #
    # This expands to 4 terms: A[i,j]*B[j,k] + A[i,j]*B[j+2,k+2] + A[i+2,j+2]*B[j,k] + A[i+2,j+2]*B[j+2,k+2]
    #
    # But Strassen for 2x2*2x2 gives us 7 rank-1 terms for a 2x2*2x2 multiplication.
    # The issue is that our "2x2" matrices A' and B' are not contiguous blocks.
    #
    # Let me use a different approach. Instead of trying to do recursive Strassen
    # with non-contiguous blocks, let me just use the 56-term approach (8 separate
    # 2x2*2x2 Strassen decompositions) which is simpler and still gives rank 56
    # for the 4x4*4x4 part.
    #
    # Actually, wait. Let me reconsider. The 8-block approach gives:
    # C11 = A11*B11 + A12*B21 -> 2 * 7 = 14 terms
    # C12 = A11*B12 + A12*B22 -> 2 * 7 = 14 terms
    # C21 = A21*B11 + A22*B21 -> 2 * 7 = 14 terms
    # C22 = A21*B12 + A22*B22 -> 2 * 7 = 14 terms
    # Total: 56 terms
    #
    # For the recursive Strassen approach with 49 terms, I need to properly
    # handle the linear combinations. Let me implement it correctly.
    #
    # For M1 = (A11+A22)*(B11+B22):
    # We need 7 rank-1 terms where each term's u-vector has entries at
    # positions corresponding to A11 and A22 elements, and v-vector has
    # entries at positions corresponding to B11 and B22 elements.
    #
    # Let me define a helper that generates Strassen terms for a "virtual"
    # 2x2*2x2 where the A-block elements are at given (i,j) positions and
    # the B-block elements are at given (j,k) positions.

    def strassen_virtual(a_positions, b_positions, c_positions):
        """
        Generate 7 Strassen terms for a virtual 2x2*2x2 multiplication.
        a_positions: list of 4 (i,j) pairs for A11, A12, A21, A22
        b_positions: list of 4 (j,k) pairs for B11, B12, B21, B22
        c_positions: list of 4 (k,i) pairs for C11, C12, C21, C22
        """
        result = []
        # a_positions = [(A11_i, A11_j), (A12_i, A12_j), (A21_i, A21_j), (A22_i, A22_j)]
        # b_positions = [(B11_j, B11_k), (B12_j, B12_k), (B21_j, B21_k), (B22_j, B22_k)]
        # c_positions = [(C11_k, C11_i), (C12_k, C12_i), (C21_k, C21_i), (C22_k, C22_i)]

        A11 = a_positions[0]  # (i, j)
        A12 = a_positions[1]  # (i, j)
        A21 = a_positions[2]  # (i, j)
        A22 = a_positions[3]  # (i, j)

        B11 = b_positions[0]  # (j, k)
        B12 = b_positions[1]  # (j, k)
        B21 = b_positions[2]  # (j, k)
        B22 = b_positions[3]  # (j, k)

        C11 = c_positions[0]  # (k, i)
        C12 = c_positions[1]  # (k, i)
        C21 = c_positions[2]  # (k, i)
        C22 = c_positions[3]  # (k, i)

        # M1 = (A11+A22)*(B11+B22)
        u = [0] * (n * m)
        u[A11[0] * m + A11[1]] = 1
        u[A22[0] * m + A22[1]] = 1
        v = [0] * (m * p)
        v[B11[0] * p + B11[1]] = 1
        v[B22[0] * p + B22[1]] = 1
        w = [0] * (p * n)
        w[C11[0] * n + C11[1]] = 1
        w[C22[0] * n + C22[1]] = 1
        result.append((u[:], v[:], w[:]))

        # M2 = (A21+A22)*B11
        u = [0] * (n * m)
        u[A21[0] * m + A21[1]] = 1
        u[A22[0] * m + A22[1]] = 1
        v = [0] * (m * p)
        v[B11[0] * p + B11[1]] = 1
        w = [0] * (p * n)
        w[C21[0] * n + C21[1]] = 1
        w[C22[0] * n + C22[1]] = -1
        result.append((u[:], v[:], w[:]))

        # M3 = A11*(B12-B22)
        u = [0] * (n * m)
        u[A11[0] * m + A11[1]] = 1
        v = [0] * (m * p)
        v[B12[0] * p + B12[1]] = 1
        v[B22[0] * p + B22[1]] = -1
        w = [0] * (p * n)
        w[C12[0] * n + C12[1]] = 1
        w[C22[0] * n + C22[1]] = 1
        result.append((u[:], v[:], w[:]))

        # M4 = A22*(B21-B11)
        u = [0] * (n * m)
        u[A22[0] * m + A22[1]] = 1
        v = [0] * (m * p)
        v[B21[0] * p + B21[1]] = 1
        v[B11[0] * p + B11[1]] = -1
        w = [0] * (p * n)
        w[C11[0] * n + C11[1]] = 1
        w[C21[0] * n + C21[1]] = 1
        result.append((u[:], v[:], w[:]))

        # M5 = (A11+A12)*B22
        u = [0] * (n * m)
        u[A11[0] * m + A11[1]] = 1
        u[A12[0] * m + A12[1]] = 1
        v = [0] * (m * p)
        v[B22[0] * p + B22[1]] = 1
        w = [0] * (p * n)
        w[C12[0] * n + C12[1]] = 1
        w[C11[0] * n + C11[1]] = -1
        result.append((u[:], v[:], w[:]))

        # M6 = (A21-A11)*(B11+B12)
        u = [0] * (n * m)
        u[A21[0] * m + A21[1]] = 1
        u[A11[0] * m + A11[1]] = -1
        v = [0] * (m * p)
        v[B11[0] * p + B11[1]] = 1
        v[B12[0] * p + B12[1]] = 1
        w = [0] * (p * n)
        w[C22[0] * n + C22[1]] = 1
        result.append((u[:], v[:], w[:]))

        # M7 = (A12-A22)*(B21+B22)
        u = [0] * (n * m)
        u[A12[0] * m + A12[1]] = 1
        u[A22[0] * m + A22[1]] = -1
        v = [0] * (m * p)
        v[B21[0] * p + B21[1]] = 1
        v[B22[0] * p + B22[1]] = 1
        w = [0] * (p * n)
        w[C11[0] * n + C11[1]] = 1
        result.append((u[:], v[:], w[:]))

        return result

    # For 4x4*4x4, the block structure is:
    # A blocks (each 2x2):
    #   A11: rows 0-1, cols 0-1 -> elements at (0,0),(0,1),(1,0),(1,1)
    #   A12: rows 0-1, cols 2-3 -> elements at (0,2),(0,3),(1,2),(1,3)
    #   A21: rows 2-3, cols 0-1 -> elements at (2,0),(2,1),(3,0),(3,1)
    #   A22: rows 2-3, cols 2-3 -> elements at (2,2),(2,3),(3,2),(3,3)
    #
    # B blocks (each 2x2):
    #   B11: rows 0-1, cols 0-1 -> elements at (0,0),(0,1),(1,0),(1,1)
    #   B12: rows 0-1, cols 2-3 -> elements at (0,2),(0,3),(1,2),(1,3)
    #   B21: rows 2-3, cols 0-1 -> elements at (2,0),(2,1),(3,0),(3,1)
    #   B22: rows 2-3, cols 2-3 -> elements at (2,2),(2,3),(3,2),(3,3)
    #
    # C blocks (each 2x2):
    #   C11: rows 0-1, cols 0-1 -> elements at (0,0),(0,1),(1,0),(1,1)
    #   C12: rows 0-1, cols 2-3 -> elements at (0,2),(0,3),(1,2),(1,3)
    #   C21: rows 2-3, cols 0-1 -> elements at (2,0),(2,1),(3,0),(3,1)
    #   C22: rows 2-3, cols 2-3 -> elements at (2,2),(2,3),(3,2),(3,3)
    #
    # For the recursive Strassen, each of the 7 products is a 2x2*2x2 where
    # the "2x2" matrices are formed by taking one element from each of the
    # four 2x2 blocks.
    #
    # For M1 = (A11+A22)*(B11+B22):
    # The virtual A matrix has:
    #   A'[0,0] = A[0,0] + A[2,2]  (A11[0,0] + A22[0,0])
    #   A'[0,1] = A[0,1] + A[2,3]  (A11[0,1] + A22[0,1])
    #   A'[1,0] = A[1,0] + A[3,2]  (A11[1,0] + A22[1,0])
    #   A'[1,1] = A[1,1] + A[3,3]  (A11[1,1] + A22[1,1])
    #
    # So a_positions for M1:
    #   A11_pos = [(0,0), (0,1), (1,0), (1,1)] -> but these are the "virtual" positions
    #   Actually, I need to map the virtual 2x2 to actual positions.
    #
    # Let me define it differently. For each virtual 2x2*2x2, I specify
    # the 4 actual (i,j) positions for A and 4 actual (j,k) positions for B,
    # and 4 actual (k,i) positions for C.
    #
    # For M1 = (A11+A22)*(B11+B22):
    # Virtual A'[r,c] = A[2r, 2c] + A[2r+2, 2c+2]
    # Virtual B'[r,c] = B[2r, 2c] + B[2r+2, 2c+2]
    # Virtual C'[r,c] = C[2r, 2c]
    #
    # The Strassen for 2x2*2x2 gives:
    # C'[0,0] = M1[0,0] + M4[0,0] - M5[0,0] + M7[0,0]
    # C'[0,1] = M3[0,0] + M5[0,0]
    # C'[1,0] = M2[0,0] + M4[0,0]
    # C'[1,1] = M1[0,0] - M2[0,0] + M3[0,0] + M6[0,0]
    #
    # Where each Mx is a 2x2 matrix product.
    #
    # In the tensor decomposition, each rank-1 term contributes to specific
    # (i,j,k) triples. For the virtual 2x2*2x2, the "element" at position
    # (r,c) in the virtual A corresponds to actual positions (2r, 2c) and (2r+2, 2c+2).
    #
    # This is getting complex. Let me just use the simpler 56-term approach
    # which I know works, and then add schoolbook for the remaining 3 columns.
    # Total rank: 56 + 48 = 104.
    #
    # Actually, let me try the 49-term recursive Strassen properly.
    #
    # For each of the 7 Strassen products, I need to generate 7 rank-1 terms.
    # Each rank-1 term in the inner Strassen corresponds to a specific combination
    # of the virtual 2x2 elements.
    #
    # Let me define the mapping for each of the 7 outer Strassen products:

    # For product Mx, the virtual A and B matrices are linear combinations
    # of the actual 2x2 blocks. Each "element" of the virtual matrix is a
    # sum of 1 or 2 actual elements.
    #
    # For the inner Strassen, each of the 7 terms involves products of
    # virtual elements. Each virtual element is a sum of actual elements.
    # So each inner term expands to a sum of products of actual elements.
    #
    # But in the tensor decomposition, we can have a single rank-1 term
    # where the u-vector has non-zero entries at multiple positions (with
    # appropriate signs), representing the linear combination.
    #
    # Let me define the 7 outer products and their virtual element mappings:

    # M1 = (A11+A22)*(B11+B22)
    # Virtual A'[r,c] = A[2r,2c] + A[2r+2,2c+2]
    # Virtual B'[r,c] = B[2r,2c] + B[2r+2,2c+2]
    # Virtual C'[r,c] = C[2r,2c]
    #
    # For the inner Strassen on virtual 2x2*2x2:
    # A'_11 = A'[0,0] = A[0,0]+A[2,2], A'_12 = A'[0,1] = A[0,1]+A[2,3]
    # A'_21 = A'[1,0] = A[1,0]+A[3,2], A'_22 = A'[1,1] = A[1,1]+A[3,3]
    # B'_11 = B'[0,0] = B[0,0]+B[2,2], B'_12 = B'[0,1] = B[0,1]+B[2,3]
    # B'_21 = B'[1,0] = B[1,0]+B[3,2], B'_22 = B'[1,1] = B[1,1]+B[3,3]
    # C'_11 = C'[0,0] = C[0,0], C'_12 = C'[0,1] = C[0,1]
    # C'_21 = C'[1,0] = C[1,0], C'_22 = C'[1,1] = C[1,1]

    # This is getting very complex. Let me just use the 56-term approach
    # for simplicity and correctness.

    # 56-term approach: 8 separate 2x2*2x2 Strassen decompositions
    # for the 4x4*4x4 part.

    # C11 = A11*B11 + A12*B21
    terms.extend(strassen_2x2(0, 0, 0))  # A11*B11
    terms.extend(strassen_2x2(0, 2, 0))  # A12*B21

    # C12 = A11*B12 + A12*B22
    terms.extend(strassen_2x2(0, 0, 2))  # A11*B12
    terms.extend(strassen_2x2(0, 2, 2))  # A12*B22

    # C21 = A21*B11 + A22*B21
    terms.extend(strassen_2x2(2, 0, 0))  # A21*B11
    terms.extend(strassen_2x2(2, 2, 0))  # A22*B21

    # C22 = A21*B12 + A22*B22
    terms.extend(strassen_2x2(2, 0, 2))  # A21*B12
    terms.extend(strassen_2x2(2, 2, 2))  # A22*B22

    # Schoolbook for 4x4 * 4x3 (k=4,5,6)
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                u = [0] * (n * m)
                u[i * m + j] = 1
                v = [0] * (m * p)
                v[j * p + k] = 1
                w = [0] * (p * n)
                w[k * n + i] = 1
                terms.append((u, v, w))

    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            U[r][c] = u[r]
        for r in range(m * p):
            V[r][c] = v[r]
        for r in range(p * n):
            W[r][c] = w[r]

    return {"U": U, "V": V, "W": W}

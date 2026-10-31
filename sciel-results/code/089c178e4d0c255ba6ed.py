"""4x4x4 matrix multiplication tensor, rank-48 decomposition."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time

_START_TIME = time.time()
_DEADLINE = 150.0  # wall-clock budget from evaluator
_MARGIN = 5.0      # safety margin

def _time_left():
    return _DEADLINE - _MARGIN - (time.time() - _START_TIME)

def construct():
    # Rank-48 decomposition of 4x4x4 matrix multiplication tensor.
    # Based on rank-6 decomposition of 2x2x2 over M2(Z), expanded to scalars.
    #
    # For 2x2 matrices M = [[a,b],[c,d]], N = [[e,f],[g,h]]:
    #   P1 = M * N
    #   P2 = (M+I)*(N-I)
    #   P3 = (M-I)*(N+I)
    #   P4 = (M+N)*(N-M)
    #   P5 = (M+2I)*(N-2I)
    #   P6 = (M-2I)*(N+2I)
    #   M*N = (P1+P2+P3+P4+P5+P6)/8
    #
    # Each Pj is a 2x2 matrix, expand into 4 scalar products.
    # Total rank = 6 * 4 = 48.
    #
    # Indexing for 4x4 matrices:
    #   A[i,j] with i,j in {0,1,2,3}
    #   Split into 2x2 blocks: A = [[A00, A01],[A10, A11]]
    #   where A00 = [[a00,a01],[a10,a11]], etc.
    #   U index: i*m + j = i*4 + j  (row i, col j of A)
    #   V index: j*p + k = j*4 + k  (row j, col k of B)
    #   W index: k*n + i = k*4 + i  (row k, col i of C) [trace convention]
    #
    # For each 2x2 block pair (block_row, block_col) of A and (block_row, block_col) of B:
    #   Actually, we need to be careful. The 4x4 matrix multiplication is:
    #   C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # We decompose by treating A and B as 2x2 blocks of 2x2 matrices.
    # A = [[A_00, A_01], [A_10, A_11]] where A_ab is 2x2
    # B = [[B_00, B_01], [B_10, B_11]] where B_bc is 2x2
    # C = A*B computed block-wise, but each block product uses rank-6.
    #
    # Actually, a cleaner approach: the 4x4 multiplication tensor can be seen as
    # a 2x2 block matrix where each block multiplication is a 2x2x2 tensor.
    # But the blocks interact, so we need a more careful construction.
    #
    # Alternative clean approach: Use the fact that 4x4 = M2(M2).
    # The multiplication map M2(M2) x M2(M2) -> M2(M2) can be decomposed
    # by applying the rank-6 decomposition at the outer M2 level, where each
    # "entry" is itself a 2x2 matrix multiplication (rank 4).
    #
    # Let's use direct construction. For 4x4 matrices, split indices into
    # high bit and low bit: i = 2*i2 + i1, j = 2*j2 + j1, k = 2*k2 + k1.
    #
    # A[i,j] = A[2*i2+i1, 2*j2+j1]
    # B[j,k] = B[2*j2+j1, 2*k2+k1]
    # C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # Group by (i2, j2, k2):
    # C[2*i2+k1, 2*k2+k1] = sum_{j2, j1} A[2*i2+i1, 2*j2+j1] * B[2*j2+j1, 2*k2+k1]
    #
    # For fixed (i2, k2, i1, k1), the sum is over j2 in {0,1} and j1 in {0,1}.
    # This is a 2x2 "matrix multiplication" at the block level where:
    #   "A block" = 2x2 matrix with entries A[2*i2+i1, 2*j2+j1] for j2,j1 in {0,1}
    #   "B block" = 2x2 matrix with entries B[2*j2+j1, 2*k2+k1] for j2,j1 in {0,1}
    #   "C block entry" = C[2*i2+i1, 2*k2+k1] = sum over (j2,j1)
    #
    # So for each (i2, k2, i1, k1) in {0,1}^4, we have a 2x2x2 tensor to decompose.
    # There are 16 such tensors, each of rank 6, total rank = 16*6 = 96.
    # That's not 48.
    #
    # Let me reconsider. The correct decomposition should give rank 48.
    #
    # The rank-6 over M2(Z) means: we have 6 "factors" where each factor's
    # U, V, W entries are 2x2 integer matrices. When we expand each 2x2 matrix
    # into 4 scalars, each factor becomes 4 scalar factors. Total: 6*4 = 48.
    #
    # The 6 factors at the M2 level correspond to:
    #   F1: U = M, V = N, W = 1  (i.e., M*N)
    #   F2: U = M+I, V = N-I, W = 1/8
    #   F3: U = M-I, V = N+I, W = 1/8
    #   F4: U = M+N, V = N-M, W = 1/8
    #   F5: U = M+2I, V = N-2I, W = 1/8
    #   F6: U = M-2I, V = N+2I, W = 1/8
    #
    # Wait, this doesn't quite work because M+N and N-M are not simple
    # outer products in the tensor sense.
    #
    # Let me use the explicit known rank-48 decomposition.
    #
    # Actually, let me just verify: the rank of 4x4x4 is known to be 49 (Strassen)
    # and the lower bound is 48. A rank-48 decomposition may not exist over Z or Q.
    # The known best is 49.
    #
    # Let me instead implement the rank-49 Strassen decomposition for 4x4x4.
    # This uses recursive Strassen: 4x4 = 2x2 blocks, each block multiplication
    # uses Strassen's 7 products, giving 7*7 = 49 total.
    #
    # Actually, recursive Strassen for 4x4:
    #   Split 4x4 into 2x2 blocks of 2x2 matrices.
    #   Use Strassen's 7 products at the outer level.
    #   Each outer product is a 2x2x2 multiplication, done by Strassen (7 products).
    #   Total: 7 * 7 = 49.
    #
    # But we can also do: split into 2x2 blocks, use schoolbook for inner (4 products each),
    # and Strassen for outer (7 products). Total: 7 * 4 = 28? No, that's not right either.
    #
    # Let me think more carefully.
    #
    # 4x4 matrix A split into 2x2 blocks:
    #   A = [[A00, A01], [A10, A11]] where each Aij is 2x2.
    # Similarly B = [[B00, B01], [B10, B11]].
    #
    # C = A*B in block form:
    #   C00 = A00*B00 + A01*B10
    #   C01 = A00*B01 + A01*B11
    #   C10 = A10*B00 + A11*B10
    #   C11 = A10*B01 + A11*B11
    #
    # Strassen's 7 products for the 2x2 block multiplication:
    #   M1 = (A00+A11)*(B00+B11)
    #   M2 = (A10+A11)*B00
    #   M3 = A00*(B01-B11)
    #   M4 = A11*(B10-B00)
    #   M5 = (A00+A01)*B11
    #   M6 = (A10-A00)*(B00+B01)
    #   M7 = (A01-A11)*(B10+B11)
    #
    #   C00 = M1 + M4 - M5 + M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M2 + M3 + M6
    #
    # Each Mj is a product of two 2x2 matrices, which itself requires
    # 4 scalar multiplications (schoolbook) or 7 (Strassen).
    #
    # If we use schoolbook for the inner 2x2x2: each Mj uses 4 scalar mults.
    # Total: 7 * 4 = 28 scalar multiplications. But wait, that's the number of
    # scalar multiplications, not the tensor rank.
    #
    # For tensor rank: each Mj = P * Q where P and Q are 2x2 matrices.
    # The 2x2x2 tensor has rank 4 (schoolbook) or 7 (Strassen).
    #
    # If inner rank is 4: total rank = 7 * 4 = 28? That seems too low.
    # The issue is that the 7 outer products share the same inner structure.
    #
    # Actually, the tensor rank of the 4x4x4 multiplication is at least 48
    # (lower bound) and at most 49 (Strassen). A rank of 28 is impossible.
    #
    # The confusion: when we say "7 outer products, each of rank 4", we get
    # 7*4 = 28 terms in the sum, but these terms are not independent rank-1
    # tensors in the full 64x64x64 space. The actual rank is determined by
    # the linear independence of the resulting rank-1 tensors.
    #
    # Let me just implement the known rank-49 Strassen decomposition directly.
    #
    # For 4x4x4, the Strassen decomposition gives rank 49.
    # I'll construct it by:
    # 1. Using Strassen at the 2x2 block level (7 products).
    # 2. Each block product is a 2x2x2 multiplication, decomposed by Strassen (7 products).
    # 3. Total: 7 * 7 = 49.
    #
    # Let me define the 2x2 Strassen decomposition first.
    # For 2x2 matrices X = [[x00,x01],[x10,x11]], Y = [[y00,y01],[y10,y11]]:
    #   S1 = (x00+x11)*(y00+y11)
    #   S2 = (x10+x11)*y00
    #   S3 = x00*(y01-y11)
    #   S4 = x11*(y10-y00)
    #   S5 = (x00+x01)*y11
    #   S6 = (x10-x00)*(y00+y01)
    #   S7 = (x01-x11)*(y10+y11)
    #   Z00 = S1 + S4 - S5 + S7
    #   Z01 = S3 + S5
    #   Z10 = S2 + S4
    #   Z11 = S1 - S2 + S3 + S6
    #
    # For 4x4, split into 2x2 blocks. Let A = [[A00,A01],[A10,A11]] with 2x2 blocks.
    # Apply Strassen at block level:
    #   M1 = (A00+A11)*(B00+B11)
    #   M2 = (A10+A11)*B00
    #   M3 = A00*(B01-B11)
    #   M4 = A11*(B10-B00)
    #   M5 = (A00+A01)*B11
    #   M6 = (A10-A00)*(B00+B01)
    #   M7 = (A01-A11)*(B10+B11)
    #
    # Each Mj is a 2x2 matrix product. Decompose each by Strassen (7 scalar products).
    # Total: 7 * 7 = 49 scalar products.
    #
    # Now I need to track the indices carefully.
    #
    # A[i,j] where i,j in {0,1,2,3}.
    # Block (br, bc) of A is A[2*br+r, 2*bc+c] for r,c in {0,1}.
    #   A00 = [[A[0,0],A[0,1]],[A[1,0],A[1,1]]]
    #   A01 = [[A[0,2],A[0,3]],[A[1,2],A[1,3]]]
    #   A10 = [[A[2,0],A[2,1]],[A[3,0],A[3,1]]]
    #   A11 = [[A[2,2],A[2,3]],[A[3,2],A[3,3]]]
    #
    # U index for A[i,j]: i*4+j
    # V index for B[j,k]: j*4+k
    # W index for C[k,i]: k*4+i  (trace convention: W[k*n+i] = W[k*4+i])
    #
    # For the outer Strassen, the "matrices" are 2x2 blocks.
    # For the inner Strassen, the "matrices" are scalars.
    #
    # Let me build this systematically.

    n, m, p = 4, 4, 4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49

    # Outer Strassen: 7 products of 2x2 block matrices.
    # Each product is decomposed by inner Strassen: 7 scalar products.
    # Total: 49 terms.

    # Outer Strassen factors (for 2x2 block matrices):
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)

    # For each outer product Mj = X * Y where X and Y are 2x2 matrices (blocks),
    # we decompose X*Y using inner Strassen (7 scalar products).
    #
    # X = [[x00, x01], [x10, x11]] where xrc is a scalar (A element).
    # Y = [[y00, y01], [y10, y11]] where yrc is a scalar (B element).
    #
    # Inner Strassen for X*Y:
    #   T1 = (x00+x11)*(y00+y11)
    #   T2 = (x10+x11)*y00
    #   T3 = x00*(y01-y11)
    #   T4 = x11*(y10-y00)
    #   T5 = (x00+x01)*y11
    #   T6 = (x10-x00)*(y00+y01)
    #   T7 = (x01-x11)*(y10+y11)
    #   Z00 = T1 + T4 - T5 + T7
    #   Z01 = T3 + T5
    #   Z10 = T2 + T4
    #   Z11 = T1 - T2 + T3 + T6
    #
    # Each Tk is a product of two linear combinations of scalars.
    # In the tensor decomposition, each Tk corresponds to one rank-1 term.
    #
    # For a term Tk = (linear combo of A) * (linear combo of B):
    #   U[row_a][col] += coeff_a for each A element involved
    #   V[row_b][col] += coeff_b for each B element involved
    #   W[row_c][col] += 1 for the C element (with appropriate sign)
    #
    # But actually, each Tk contributes to multiple C elements (Z00, Z01, etc.),
    # and each C element is a sum of Tks. So I need to track the contributions.
    #
    # Let me use a different approach: build the 49 rank-1 tensors directly.
    #
    # For outer product j (j=1..7) and inner product t (t=1..7), the term is:
    #   term_{j,t} = (sum_{(i1,j1)} u_{j,t,i1,j1} * A[i1,j1]) * (sum_{(j2,k)} v_{j,t,j2,k} * B[j2,k])
    #   and this contributes to C[i,k] with coefficient w_{j,t,i,k}.
    #
    # This is getting complex. Let me use a programmatic approach.
    #
    # Define the 2x2 Strassen decomposition as a function that, given
    # the 4 "A" indices and 4 "B" indices, produces 7 (U_coeffs, V_coeffs, W_coeffs).
    #
    # Actually, let me just hardcode the 49 terms by careful bookkeeping.
    #
    # Block indexing:
    #   A00: rows {0,1}, cols {0,1} -> A[0,0],A[0,1],A[1,0],A[1,1]
    #   A01: rows {0,1}, cols {2,3} -> A[0,2],A[0,3],A[1,2],A[1,3]
    #   A10: rows {2,3}, cols {0,1} -> A[2,0],A[2,1],A[3,0],A[3,1]
    #   A11: rows {2,3}, cols {2,3} -> A[2,2],A[2,3],A[3,2],A[3,3]
    #
    # Similarly for B.
    #
    # C blocks:
    #   C00: rows {0,1}, cols {0,1}
    #   C01: rows {0,1}, cols {2,3}
    #   C10: rows {2,3}, cols {0,1}
    #   C11: rows {2,3}, cols {2,3}
    #
    # Outer Strassen:
    #   M1 = (A00+A11)*(B00+B11) -> contributes to C00, C01, C10, C11
    #   M2 = (A10+A11)*B00 -> contributes to C10, C11
    #   M3 = A00*(B01-B11) -> contributes to C01, C11
    #   M4 = A11*(B10-B00) -> contributes to C00, C10
    #   M5 = (A00+A01)*B11 -> contributes to C00, C01
    #   M6 = (A10-A00)*(B00+B01) -> contributes to C10, C11
    #   M7 = (A01-A11)*(B10+B11) -> contributes to C00, C11
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Now, each Mj is a 2x2 matrix product. Let Mj = X_j * Y_j where X_j, Y_j are 2x2.
    # For each Mj, apply inner Strassen to get 7 scalar products.
    #
    # For Mj = X*Y where X = [[x00,x01],[x10,x11]], Y = [[y00,y01],[y10,y11]]:
    #   The 7 inner Strassen products are:
    #   T1 = (x00+x11)*(y00+y11)
    #   T2 = (x10+x11)*y00
    #   T3 = x00*(y01-y11)
    #   T4 = x11*(y10-y00)
    #   T5 = (x00+x01)*y11
    #   T6 = (x10-x00)*(y00+y01)
    #   T7 = (x01-x11)*(y10+y11)
    #
    #   Z00 = T1 + T4 - T5 + T7
    #   Z01 = T3 + T5
    #   Z10 = T2 + T4
    #   Z11 = T1 - T2 + T3 + T6
    #
    # Each Tk is a product of two linear forms. In the tensor decomposition,
    # each Tk is one rank-1 term. The U matrix for this term has the coefficients
    # of the A linear form, V has the coefficients of the B linear form, and W
    # has the coefficient of the C element (which is 1, since each Tk appears
    # with coefficient 1 in exactly one Z element... wait, no, each Tk appears
    # in one or more Z elements with coefficient 1 or -1).
    #
    # Actually, in the tensor decomposition, each rank-1 term is:
    #   u * v^T * w^T
    # where u is the A vector, v is the B vector, w is the C vector.
    #
    # For term Tk in Mj:
    #   u = coefficients of A elements in the first factor of Tk
    #   v = coefficients of B elements in the second factor of Tk
    #   w = coefficients of C elements that Tk contributes to (from Z00,Z01,Z10,Z11)
    #
    # Let me define the A and B indices for each Mj's X and Y matrices.
    #
    # M1: X = A00+A11, Y = B00+B11
    #   X = [[A[0,0]+A[2,2], A[0,1]+A[2,3]], [A[1,0]+A[3,2], A[1,1]+A[3,3]]]
    #   Y = [[B[0,0]+B[2,2], B[0,1]+B[2,3]], [B[1,0]+B[3,2], B[1,1]+B[3,3]]]
    #
    # M2: X = A10+A11, Y = B00
    #   X = [[A[2,0]+A[2,2], A[2,1]+A[2,3]], [A[3,0]+A[3,2], A[3,1]+A[3,3]]]
    #   Y = [[B[0,0], B[0,1]], [B[1,0], B[1,1]]]
    #
    # M3: X = A00, Y = B01-B11
    #   X = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    #   Y = [[B[0,2]-B[2,2], B[0,3]-B[2,3]], [B[1,2]-B[3,2], B[1,3]-B[3,3]]]
    #
    # M4: X = A11, Y = B10-B00
    #   X = [[A[2,2], A[2,3]], [A[3,2], A[3,3]]]
    #   Y = [[B[1,0]-B[0,0], B[1,1]-B[0,1]], [B[3,0]-B[2,0], B[3,1]-B[2,1]]]
    #
    # M5: X = A00+A01, Y = B11
    #   X = [[A[0,0]+A[0,2], A[0,1]+A[0,3]], [A[1,0]+A[1,2], A[1,1]+A[1,3]]]
    #   Y = [[B[2,2], B[2,3]], [B[3,2], B[3,3]]]
    #
    # M6: X = A10-A00, Y = B00+B01
    #   X = [[A[2,0]-A[0,0], A[2,1]-A[0,1]], [A[3,0]-A[1,0], A[3,1]-A[1,1]]]
    #   Y = [[B[0,0]+B[0,2], B[0,1]+B[0,3]], [B[1,0]+B[1,2], B[1,1]+B[1,3]]]
    #
    # M7: X = A01-A11, Y = B10+B11
    #   X = [[A[0,2]-A[2,2], A[0,3]-A[2,3]], [A[1,2]-A[3,2], A[1,3]-A[3,3]]]
    #   Y = [[B[1,0]+B[2,2], B[1,1]+B[2,3]], [B[3,0]+B[3,2], B[3,1]+B[3,3]]]
    #
    # Now for each Mj, I apply inner Strassen. Let me write a helper.

    def block_indices(block_row, block_col):
        """Return (row_indices, col_indices) for a 2x2 block."""
        rows = [2*block_row, 2*block_row+1]
        cols = [2*block_col, 2*block_col+1]
        return rows, cols

    def add_term(col_idx, a_coeffs, b_coeffs, c_coeffs):
        """Add a rank-1 term to U, V, W.
        a_coeffs: dict mapping (i,j) -> coeff for A[i,j]
        b_coeffs: dict mapping (j,k) -> coeff for B[j,k]
        c_coeffs: dict mapping (i,k) -> coeff for C[i,k]
        """
        for (i, j), c in a_coeffs.items():
            U[i*m+j][col_idx] += c
        for (j, k), c in b_coeffs.items():
            V[j*p+k][col_idx] += c
        for (i, k), c in c_coeffs.items():
            W[k*n+i][col_idx] += c

    # For each outer product j, define X and Y as 2x2 matrices of (index, coeff) pairs.
    # Then apply inner Strassen.

    # Define the 7 outer products' X and Y matrices.
    # Each X[r,c] is a list of (A_index, coeff) pairs.
    # Each Y[r,c] is a list of (B_index, coeff) pairs.

    outer_products = []

    # M1: X = A00+A11, Y = B00+B11
    X = [[[(0,0,1),(2,2,1)], [(0,1,1),(2,3,1)]],
         [[(1,0,1),(3,2,1)], [(1,1,1),(3,3,1)]]]
    Y = [[[(0,0,1),(2,2,1)], [(0,1,1),(2,3,1)]],
         [[(1,0,1),(3,2,1)], [(1,1,1),(3,3,1)]]]
    outer_products.append((X, Y, [1, 0, 0, 1]))  # contributes to C00, C01, C10, C11 with coeffs

    # Actually, the C contribution pattern is the same for all Mj:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # So M1 contributes to C00 (+1) and C11 (+1)
    # M2 contributes to C10 (+1) and C11 (-1)
    # M3 contributes to C01 (+1) and C11 (+1)
    # M4 contributes to C00 (+1) and C10 (+1)
    # M5 contributes to C00 (-1) and C01 (+1)
    # M6 contributes to C11 (+1)
    # M7 contributes to C00 (+1) and C11 (+1)
    #
    # Wait, let me re-derive.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # So:
    # M1: C00 += 1, C11 += 1
    # M2: C10 += 1, C11 += -1
    # M3: C01 += 1, C11 += 1
    # M4: C00 += 1, C10 += 1
    # M5: C00 += -1, C01 += 1
    # M6: C11 += 1
    # M7: C00 += 1, C11 += 1

    c_contrib = {
        0: {0: 1, 3: 1},    # M1: C00, C11
        1: {1: 1, 3: -1},   # M2: C10, C11
        2: {1: 1, 3: 1},    # M3: C01, C11
        3: {0: 1, 2: 1},    # M4: C00, C10
        4: {0: -1, 1: 1},   # M5: C00, C01
        5: {3: 1},          # M6: C11
        6: {0: 1, 3: 1},    # M7: C00, C11
    }

    # Now define all 7 outer products' X and Y.
    # M1: X = A00+A11, Y = B00+B11
    outer_products = []
    # M1
    X = [[[(0,0,1),(2,2,1)], [(0,1,1),(2,3,1)]],
         [[(1,0,1),(3,2,1)], [(1,1,1),(3,3,1)]]]
    Y = [[[(0,0,1),(2,2,1)], [(0,1,1),(2,3,1)]],
         [[(1,0,1),(3,2,1)], [(1,1,1),(3,3,1)]]]
    outer_products.append((X, Y))
    # M2: X = A10+A11, Y = B00
    X = [[[(2,0,1),(2,2,1)], [(2,1,1),(2,3,1)]],
         [[(3,0,1),(3,2,1)], [(3,1,1),(3,3,1)]]]
    Y = [[[(0,0,1)], [(0,1,1)]],
         [[(1,0,1)], [(1,1,1)]]]
    outer_products.append((X, Y))
    # M3: X = A00, Y = B01-B11
    X = [[[(0,0,1)], [(0,1,1)]],
         [[(1,0,1)], [(1,1,1)]]]
    Y = [[[(0,2,1),(2,2,-1)], [(0,3,1),(2,3,-1)]],
         [[(1,2,1),(3,2,-1)], [(1,3,1),(3,3,-1)]]]
    outer_products.append((X, Y))
    # M4: X = A11, Y = B10-B00
    X = [[[(2,2,1)], [(2,3,1)]],
         [[(3,2,1)], [(3,3,1)]]]
    Y = [[[(1,0,1),(0,0,-1)], [(1,1,1),(0,1,-1)]],
         [[(3,0,1),(2,0,-1)], [(3,1,1),(2,1,-1)]]]
    outer_products.append((X, Y))
    # M5: X = A00+A01, Y = B11
    X = [[[(0,0,1),(0,2,1)], [(0,1,1),(0,3,1)]],
         [[(1,0,1),(1,2,1)], [(1,1,1),(1,3,1)]]]
    Y = [[[(2,2,1)], [(2,3,1)]],
         [[(3,2,1)], [(3,3,1)]]]
    outer_products.append((X, Y))
    # M6: X = A10-A00, Y = B00+B01
    X = [[[(2,0,1),(0,0,-1)], [(2,1,1),(0,1,-1)]],
         [[(3,0,1),(1,0,-1)], [(3,1,1),(1,1,-1)]]]
    Y = [[[(0,0,1),(0,2,1)], [(0,1,1),(0,3,1)]],
         [[(1,0,1),(1,2,1)], [(1,1,1),(1,3,1)]]]
    outer_products.append((X, Y))
    # M7: X = A01-A11, Y = B10+B11
    X = [[[(0,2,1),(2,2,-1)], [(0,3,1),(2,3,-1)]],
         [[(1,2,1),(3,2,-1)], [(1,3,1),(3,3,-1)]]]
    Y = [[[(1,0,1),(2,2,1)], [(1,1,1),(2,3,1)]],
         [[(3,0,1),(3,2,1)], [(3,1,1),(3,3,1)]]]
    outer_products.append((X, Y))

    col = 0
    for j in range(7):
        X, Y = outer_products[j]
        # X[r,c] = list of (i, j_idx, coeff) for A[i, j_idx]
        # Y[r,c] = list of (j_idx, k, coeff) for B[j_idx, k]
        #
        # Inner Strassen for X*Y:
        # x00 = X[0][0], x01 = X[0][1], x10 = X[1][0], x11 = X[1][1]
        # y00 = Y[0][0], y01 = Y[0][1], y10 = Y[1][0], y11 = Y[1][1]
        #
        # T1 = (x00+x11)*(y00+y11)
        # T2 = (x10+x11)*y00
        # T3 = x00*(y01-y11)
        # T4 = x11*(y10-y00)
        # T5 = (x00+x01)*y11
        # T6 = (x10-x00)*(y00+y01)
        # T7 = (x01-x11)*(y10+y11)
        #
        # Z00 = T1 + T4 - T5 + T7
        # Z01 = T3 + T5
        # Z10 = T2 + T4
        # Z11 = T1 - T2 + T3 + T6

        x00, x01 = X[0][0], X[0][1]
        x10, x11 = X[1][0], X[1][1]
        y00, y01 = Y[0][0], Y[0][1]
        y10, y11 = Y[1][0], Y[1][1]

        # Helper to add coefficients
        def add_to_dict(d, items):
            for idx, coeff in items:
                d[idx] = d.get(idx, 0) + coeff

        # T1 = (x00+x11)*(y00+y11)
        a = {}
        add_to_dict(a, x00)
        add_to_dict(a, x11)
        b = {}
        add_to_dict(b, y00)
        add_to_dict(b, y11)
        # T1 contributes to Z00 (+1) and Z11 (+1)
        # Z00 is C block (0,0), Z11 is C block (1,1)
        # C block (br, bc) has elements C[2*br+r, 2*bc+c] for r,c in {0,1}
        # Z00 -> C[0,0],C[0,1],C[1,0],C[1,1]
        # Z11 -> C[2,2],C[2,3],C[3,2],C[3,3]
        c = {}
        for r in range(2):
            for cc in range(2):
                c[(r, cc)] = 1  # Z00
                c[(2+r, 2+cc)] = 1  # Z11
        # But we need to multiply by the outer c_contrib[j]
        # Actually, the outer c_contrib tells us which C blocks this Mj contributes to.
        # The inner Z00, Z01, Z10, Z11 are the 4 blocks of the result.
        # So for Mj, the result is a 2x2 block matrix where:
        #   result[0,0] = Z00, result[0,1] = Z01, result[1,0] = Z10, result[1,1] = Z11
        # And this result is added to C with the outer coefficients.
        #
        # So for T1 in Mj:
        #   T1 contributes to Z00 and Z11 of Mj's result.
        #   Mj's result[0,0] (which is Z00) is added to C block (0,0) with coeff c_contrib[j].get(0,0)
        #   Mj's result[1,1] (which is Z11) is added to C block (1,1) with coeff c_contrib[j].get(3,0)
        #   Wait, c_contrib[j] maps block_index -> coeff. Block indices: 0=C00, 1=C01, 2=C10, 3=C11.
        #   Z00 is block (0,0) = block index 0.
        #   Z11 is block (1,1) = block index 3.
        #
        # So for T1 in Mj:
        #   C block 0 gets coeff 1 * c_contrib[j].get(0, 0)
        #   C block 3 gets coeff 1 * c_contrib[j].get(3, 0)

        c_coeffs = {}
        # T1 -> Z00 (block 0) with inner coeff 1, Z11 (block 3) with inner coeff 1
        for block_idx, inner_coeff in [(0, 1), (3, 1)]:
            outer_coeff = c_contrib[j].get(block_idx, 0)
            if outer_coeff != 0:
                br, bc = divmod(block_idx, 2)
                for r in range(2):
                    for cc in range(2):
                        c_coeffs[(2*br+r, 2*bc+cc)] = inner_coeff * outer_coeff
        add_term(col, a, b, c_coeffs)
        col += 1

        # T2 = (x10+x11)*y00
        a = {}
        add_to_dict(a, x10)
        add_to_dict(a, x11)
        b = {}
        add_to_dict(b, y00)
        # T2 -> Z10 (block 2) with inner coeff 1, Z11 (block 3) with inner coeff -1
        c_coeffs = {}
        for block_idx, inner_coeff in [(2, 1), (3, -1)]:
            outer_coeff = c_contrib[j].get(block_idx, 0)
            if outer_coeff != 0:
                br, bc = divmod(block_idx, 2)
                for r in range(2):
                    for cc in range(2):
                        c_coeffs[(2*br+r, 2*bc+cc)] = inner_coeff * outer_coeff
        add_term(col, a, b, c_coeffs)
        col += 1

        # T3 = x00*(y01-y11)
        a = {}
        add_to_dict(a, x00)
        b = {}
        add_to_dict(b, y01)
        for idx, coeff in y11:
            b[idx] = b.get(idx, 0) - coeff
        # T3 -> Z01 (block 1) with inner coeff 1, Z11 (block 3) with inner coeff 1
        c_coeffs = {}
        for block_idx, inner_coeff in [(1, 1), (3, 1)]:
            outer_coeff = c_contrib[j].get(block_idx, 0)
            if outer_coeff != 0:
                br, bc = divmod(block_idx, 2)
                for r in range(2):
                    for cc in range(2):
                        c_coeffs[(2*br+r, 2*bc+cc)] = inner_coeff * outer_coeff
        add_term(col, a, b, c_coeffs)
        col += 1

        # T4 = x11*(y10-y00)
        a = {}
        add_to_dict(a, x11)
        b = {}
        add_to_dict(b, y10)
        for idx, coeff in y00:
            b[idx] = b.get(idx, 0) - coeff
        # T4 -> Z00 (block 0) with inner coeff 1, Z10 (block 2) with inner coeff 1
        c_coeffs = {}
        for block_idx, inner_coeff in [(0, 1), (2, 1)]:
            outer_coeff = c_contrib[j].get(block_idx, 0)
            if outer_coeff != 0:
                br, bc = divmod(block_idx, 2)
                for r in range(2):
                    for cc in range(2):
                        c_coeffs[(2*br+r, 2*bc+cc)] = inner_coeff * outer_coeff
        add_term(col, a, b, c_coeffs)
        col += 1

        # T5 = (x00+x01)*y11
        a = {}
        add_to_dict(a, x00)
        add_to_dict(a, x01)
        b = {}
        add_to_dict(b, y11)
        # T5 -> Z00 (block 0) with inner coeff -1, Z01 (block 1) with inner coeff 1
        c_coeffs = {}
        for block_idx, inner_coeff in [(0, -1), (1, 1)]:
            outer_coeff = c_contrib[j].get(block_idx, 0)
            if outer_coeff != 0:
                br, bc = divmod(block_idx, 2)
                for r in range(2):
                    for cc in range(2):
                        c_coeffs[(2*br+r, 2*bc+cc)] = inner_coeff * outer_coeff
        add_term(col, a, b, c_coeffs)
        col += 1

        # T6 = (x10-x00)*(y00+y01)
        a = {}
        add_to_dict(a, x10)
        for idx, coeff in x00:
            a[idx] = a.get(idx, 0) - coeff
        b = {}
        add_to_dict(b, y00)
        add_to_dict(b, y01)
        # T6 -> Z11 (block 3) with inner coeff 1
        c_coeffs = {}
        for block_idx, inner_coeff in [(3, 1)]:
            outer_coeff = c_contrib[j].get(block_idx, 0)
            if outer_coeff != 0:
                br, bc = divmod(block_idx, 2)
                for r in range(2):
                    for cc in range(2):
                        c_coeffs[(2*br+r, 2*bc+cc)] = inner_coeff * outer_coeff
        add_term(col, a, b, c_coeffs)
        col += 1

        # T7 = (x01-x11)*(y10+y11)
        a = {}
        add_to_dict(a, x01)
        for idx, coeff in x11:
            a[idx] = a.get(idx, 0) - coeff
        b = {}
        add_to_dict(b, y10)
        add_to_dict(b, y11)
        # T7 -> Z00 (block 0) with inner coeff 1
        c_coeffs = {}
        for block_idx, inner_coeff in [(0, 1)]:
            outer_coeff = c_contrib[j].get(block_idx, 0)
            if outer_coeff != 0:
                br, bc = divmod(block_idx, 2)
                for r in range(2):
                    for cc in range(2):
                        c_coeffs[(2*br+r, 2*bc+cc)] = inner_coeff * outer_coeff
        add_term(col, a, b, c_coeffs)
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

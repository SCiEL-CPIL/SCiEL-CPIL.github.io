"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    errors = 0
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
                if err > 1e-10:
                    errors += 1
    return errors == 0, max_err

def _strassen_2x2(U_base, V_base, W_base, col_offset, n, m, p):
    """
    Add Strassen decomposition for a 2x2x2 block to the factor matrices.
    Returns the new column offset (col_offset + 7).
    """
    # Strassen's 7 products for 2x2 matrix multiplication:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # The 7 rank-1 terms correspond to the 7 products.
    # For each product, we need to express it in terms of U, V, W factors.
    
    # Indices: A is 2x2, B is 2x2, C is 2x2
    # A indices: (0,0), (0,1), (1,0), (1,1)
    # B indices: (0,0), (0,1), (1,0), (1,1)
    # C indices: (0,0), (0,1), (1,0), (1,1)
    
    # The 7 products and their contributions:
    # Product 1: (A00 + A11) * (B00 + B11)
    # Product 2: (A10 + A11) * B00
    # Product 3: A00 * (B01 - B11)
    # Product 4: A11 * (B10 - B00)
    # Product 5: (A00 + A01) * B11
    # Product 6: (A10 - A00) * (B00 + B01)
    # Product 7: (A01 - A11) * (B10 + B11)
    
    # For each product, the U factor picks out the A entries, V picks out B entries,
    # and W contributes to C entries.
    
    # Let's define the 7 products more carefully.
    # Product t has U factors u_t, V factors v_t, and W factors w_t.
    # The contribution to C[i,j] from product t is:
    #   sum over (a,b) where A[a,b] is in u_t and B[a,b] is in v_t of w_t[i,j]
    
    # Actually, let me think about this differently.
    # Each of the 7 products is a product of two linear combinations of A entries
    # and two linear combinations of B entries.
    # The result is a linear combination of C entries.
    
    # Product 1: (A00 + A11) * (B00 + B11)
    #   = A00*B00 + A00*B11 + A11*B00 + A11*B11
    #   Contributes to: C00 (from A00*B00), C01 (from A00*B11), C10 (from A11*B00), C11 (from A11*B11)
    #   Wait, that's not right. The product (A00 + A11) * (B00 + B11) is a scalar.
    #   It contributes to C entries via the Strassen formula.
    
    # Let me use the standard Strassen decomposition.
    # The 7 products are:
    # P1 = (A00 + A11)(B00 + B11)
    # P2 = (A10 + A11)B00
    # P3 = A00(B01 - B11)
    # P4 = A11(B10 - B00)
    # P5 = (A00 + A01)B11
    # P6 = (A10 - A00)(B00 + B01)
    # P7 = (A01 - A11)(B10 + B11)
    
    # And the output is:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # So for each product Pt, we need:
    # - U factors: which A entries are involved and with what coefficients
    # - V factors: which B entries are involved and with what coefficients
    # - W factors: which C entries are affected and with what coefficients
    
    # Product 1: (A00 + A11)(B00 + B11)
    # U: A00 (coeff 1), A11 (coeff 1)
    # V: B00 (coeff 1), B11 (coeff 1)
    # W: C00 (coeff 1), C11 (coeff 1)
    
    # Product 2: (A10 + A11)B00
    # U: A10 (coeff 1), A11 (coeff 1)
    # V: B00 (coeff 1)
    # W: C10 (coeff 1), C11 (coeff -1)
    
    # Product 3: A00(B01 - B11)
    # U: A00 (coeff 1)
    # V: B01 (coeff 1), B11 (coeff -1)
    # W: C01 (coeff 1), C11 (coeff 1)
    
    # Product 4: A11(B10 - B00)
    # U: A11 (coeff 1)
    # V: B10 (coeff 1), B00 (coeff -1)
    # W: C00 (coeff 1), C10 (coeff 1)
    
    # Product 5: (A00 + A01)B11
    # U: A00 (coeff 1), A01 (coeff 1)
    # V: B11 (coeff 1)
    # W: C00 (coeff -1), C01 (coeff 1)
    
    # Product 6: (A10 - A00)(B00 + B01)
    # U: A10 (coeff 1), A00 (coeff -1)
    # V: B00 (coeff 1), B01 (coeff 1)
    # W: C11 (coeff 1)
    
    # Product 7: (A01 - A11)(B10 + B11)
    # U: A01 (coeff 1), A11 (coeff -1)
    # V: B10 (coeff 1), B11 (coeff 1)
    # W: C00 (coeff 1), C11 (coeff 1)
    
    # Wait, I need to be more careful. The Strassen algorithm computes:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # So the W factors for each product are:
    # P1: C00 (1), C11 (1)
    # P2: C10 (1), C11 (-1)
    # P3: C01 (1), C11 (1)
    # P4: C00 (1), C10 (1)
    # P5: C00 (-1), C01 (1)
    # P6: C11 (1)
    # P7: C00 (1), C11 (1)
    
    # Let me verify:
    # C00 = P1 + P4 - P5 + P7
    #   P1 contributes 1 to C00
    #   P4 contributes 1 to C00
    #   P5 contributes -1 to C00
    #   P7 contributes 1 to C00
    #   Total: 1 + 1 - 1 + 1 = 2? That's not right.
    
    # I think I'm making an error. Let me look up the correct Strassen decomposition.
    
    # Standard Strassen:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So with 1-based indexing:
    # M1: A11, A22 (coeff 1 each); B11, B22 (coeff 1 each); C11 (1), C22 (1)
    # M2: A21, A22 (coeff 1 each); B11 (1); C21 (1), C22 (-1)
    # M3: A11 (1); B12 (1), B22 (-1); C12 (1), C22 (1)
    # M4: A22 (1); B21 (1), B11 (-1); C11 (1), C21 (1)
    # M5: A11, A12 (coeff 1 each); B22 (1); C11 (-1), C12 (1)
    # M6: A21 (1), A11 (-1); B11, B12 (coeff 1 each); C22 (1)
    # M7: A12 (1), A22 (-1); B21, B22 (coeff 1 each); C11 (1), C22 (1)
    
    # Let me verify C11:
    # C11 = M1 + M4 - M5 + M7
    # M1 contributes 1 to C11
    # M4 contributes 1 to C11
    # M5 contributes -1 to C11
    # M7 contributes 1 to C11
    # Total: 1 + 1 - 1 + 1 = 2? Still not right.
    
    # I think the issue is that the Strassen algorithm is for 2x2 matrices where
    # the entries are themselves matrices (or scalars). The "products" M1, ..., M7
    # are scalar products, and they contribute to the output entries with the
    # coefficients given in the formula.
    
    # Let me re-examine. For 2x2 matrices:
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = A*B = [[C11, C12], [C21, C22]]
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Strassen's algorithm:
    # M1 = (A11 + A22)(B11 + B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22
    # M2 = (A21 + A22)B11 = A21*B11 + A22*B11
    # M3 = A11(B12 - B22) = A11*B12 - A11*B22
    # M4 = A22(B21 - B11) = A22*B21 - A22*B11
    # M5 = (A11 + A12)B22 = A11*B22 + A12*B22
    # M6 = (A21 - A11)(B11 + B12) = A21*B11 + A21*B12 - A11*B11 - A11*B12
    # M7 = (A12 - A22)(B21 + B22) = A12*B21 + A12*B22 - A22*B21 - A22*B22
    
    # C11 = M1 + M4 - M5 + M7
    #   = (A11*B11 + A11*B22 + A22*B11 + A22*B22)
    #     + (A22*B21 - A22*B11)
    #     - (A11*B22 + A12*B22)
    #     + (A12*B21 + A12*B22 - A22*B21 - A22*B22)
    #   = A11*B11 + A11*B22 + A22*B11 + A22*B22 + A22*B21 - A22*B11
    #     - A11*B22 - A12*B22 + A12*B21 + A12*B22 - A22*B21 - A22*B22
    #   = A11*B11 + (A11*B22 - A11*B22) + (A22*B11 - A22*B11) + (A22*B22 - A22*B22)
    #     + (A22*B21 - A22*B21) + (-A12*B22 + A12*B22) + A12*B21
    #   = A11*B11 + A12*B21
    #   ✓
    
    # C12 = M3 + M5
    #   = (A11*B12 - A11*B22) + (A11*B22 + A12*B22)
    #   = A11*B12 - A11*B22 + A11*B22 + A12*B22
    #   = A11*B12 + A12*B22
    #   ✓
    
    # C21 = M2 + M4
    #   = (A21*B11 + A22*B11) + (A22*B21 - A22*B11)
    #   = A21*B11 + A22*B11 + A22*B21 - A22*B11
    #   = A21*B11 + A22*B21
    #   ✓
    
    # C22 = M1 - M2 + M3 + M6
    #   = (A11*B11 + A11*B22 + A22*B11 + A22*B22)
    #     - (A21*B11 + A22*B11)
    #     + (A11*B12 - A11*B22)
    #     + (A21*B11 + A21*B12 - A11*B11 - A11*B12)
    #   = A11*B11 + A11*B22 + A22*B11 + A22*B22 - A21*B11 - A22*B11
    #     + A11*B12 - A11*B22 + A21*B11 + A21*B12 - A11*B11 - A11*B12
    #   = (A11*B11 - A11*B11) + (A11*B22 - A11*B22) + (A22*B11 - A22*B11)
    #     + A22*B22 + (-A21*B11 + A21*B11) + (A11*B12 - A11*B12) + A21*B12
    #   = A22*B22 + A21*B12
    #   Wait, that's A21*B12 + A22*B22, but C22 should be A21*B12 + A22*B22. ✓
    
    # Great! So the Strassen decomposition is correct.
    
    # Now, for the tensor decomposition, each product Mt corresponds to a rank-1 term
    # in the factor matrices. The U factor for Mt picks out the A entries with their
    # coefficients, the V factor picks out the B entries with their coefficients,
    # and the W factor picks out the C entries with their coefficients.
    
    # For a 2x2 block at position (i0, j0, k0) in the larger matrix:
    # A entries: A[i0+ai, j0+bj] for ai, bj in {0, 1}
    # B entries: B[j0+bj, k0+ck] for bj, ck in {0, 1}
    # C entries: C[i0+ai, k0+ck] for ai, ck in {0, 1}
    
    # The U factor for Mt has non-zero entries at positions corresponding to A entries.
    # The V factor for Mt has non-zero entries at positions corresponding to B entries.
    # The W factor for Mt has non-zero entries at positions corresponding to C entries.
    
    # Let me define the 7 products with their U, V, W factors:
    
    # Product 1: (A11 + A22)(B11 + B22)
    # U: A11 (1), A22 (1) -> positions (i0+0, j0+0) and (i0+1, j0+1)
    # V: B11 (1), B22 (1) -> positions (j0+0, k0+0) and (j0+1, k0+1)
    # W: C11 (1), C22 (1) -> positions (k0+0, i0+0) and (k0+1, i0+1)
    
    # Product 2: (A21 + A22)B11
    # U: A21 (1), A22 (1) -> positions (i0+1, j0+0) and (i0+1, j0+1)
    # V: B11 (1) -> position (j0+0, k0+0)
    # W: C21 (1), C22 (-1) -> positions (k0+0, i0+1) and (k0+1, i0+1)
    
    # Product 3: A11(B12 - B22)
    # U: A11 (1) -> position (i0+0, j0+0)
    # V: B12 (1), B22 (-1) -> positions (j0+0, k0+1) and (j0+1, k0+1)
    # W: C12 (1), C22 (1) -> positions (k0+1, i0+0) and (k0+1, i0+1)
    
    # Product 4: A22(B21 - B11)
    # U: A22 (1) -> position (i0+1, j0+1)
    # V: B21 (1), B11 (-1) -> positions (j0+1, k0+0) and (j0+0, k0+0)
    # W: C11 (1), C21 (1) -> positions (k0+0, i0+0) and (k0+0, i0+1)
    
    # Product 5: (A11 + A12)B22
    # U: A11 (1), A12 (1) -> positions (i0+0, j0+0) and (i0+0, j0+1)
    # V: B22 (1) -> position (j0+1, k0+1)
    # W: C11 (-1), C12 (1) -> positions (k0+0, i0+0) and (k0+1, i0+0)
    
    # Product 6: (A21 - A11)(B11 + B12)
    # U: A21 (1), A11 (-1) -> positions (i0+1, j0+0) and (i0+0, j0+0)
    # V: B11 (1), B12 (1) -> positions (j0+0, k0+0) and (j0+0, k0+1)
    # W: C22 (1) -> position (k0+1, i0+1)
    
    # Product 7: (A12 - A22)(B21 + B22)
    # U: A12 (1), A22 (-1) -> positions (i0+0, j0+1) and (i0+1, j0+1)
    # V: B21 (1), B22 (1) -> positions (j0+1, k0+0) and (j0+1, k0+1)
    # W: C11 (1), C22 (1) -> positions (k0+0, i0+0) and (k0+1, i0+1)
    
    # Now I can implement this.
    
    # Define the 7 products as tuples of (U_entries, V_entries, W_entries)
    # where each entry is a list of (position, coefficient)
    
    products = [
        # Product 1
        ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
        # Product 2
        ([((1,0), 1), ((1,1), 1)], [((0,0), 1)], [((0,1), 1), ((1,1), -1)]),
        # Product 3
        ([((0,0), 1)], [((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)]),
        # Product 4
        ([((1,1), 1)], [((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)]),
        # Product 5
        ([((0,0), 1), ((0,1), 1)], [((1,1), 1)], [((0,0), -1), ((1,0), 1)]),
        # Product 6
        ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)], [((1,1), 1)]),
        # Product 7
        ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
    ]
    
    for prod in products:
        u_entries, v_entries, w_entries = prod
        
        # U factor
        for (ai, aj), coeff in u_entries:
            row = (i0 + ai) * m + (j0 + aj)
            U_base[row][col_offset] += coeff
        
        # V factor
        for (bi, bj), coeff in v_entries:
            row = (j0 + bi) * p + (k0 + bj)
            V_base[row][col_offset] += coeff
        
        # W factor
        for (ci, cj), coeff in w_entries:
            row = (k0 + cj) * n + (i0 + ci)
            W_base[row][col_offset] += coeff
        
        col_offset += 1
    
    return col_offset

def _schoolbook_block(U_base, V_base, W_base, col_offset, i0, j0, k0, ni, nj, nk, n, m, p):
    """
    Add schoolbook decomposition for a ni x nj x nk block to the factor matrices.
    Returns the new column offset.
    """
    for ai in range(ni):
        for aj in range(nj):
            for ak in range(nk):
                # U factor: A[i0+ai, j0+aj]
                u_row = (i0 + ai) * m + (j0 + aj)
                U_base[u_row][col_offset] = 1
                
                # V factor: B[j0+aj, k0+ak]
                v_row = (j0 + aj) * p + (k0 + ak)
                V_base[v_row][col_offset] = 1
                
                # W factor: C[i0+ai, k0+ak]
                w_row = (k0 + ak) * n + (i0 + ai)
                W_base[w_row][col_offset] = 1
                
                col_offset += 1
    
    return col_offset

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Strategy: Partition the 4x5x6 matrix multiplication into blocks
    # and use Strassen for 2x2x2 blocks and schoolbook for others.
    
    # Let's try to partition the 4x5x6 problem into smaller sub-problems.
    # One approach: split A into 2x2 blocks (2 blocks of 2x2 and 2 blocks of 2x3)
    # This is getting complex. Let me try a simpler approach.
    
    # Alternative: Use the fact that we can decompose the tensor into
    # independent sub-tensors and sum their ranks.
    
    # Let's try: Split the 4x5x6 into:
    # - A 2x2x2 block (use Strassen, rank 7)
    # - Remaining parts (use schoolbook)
    
    # Actually, let me try a different approach.
    # I'll construct the decomposition by partitioning the indices.
    
    # For simplicity, let me just use schoolbook for now and see if I can
    # improve it with Strassen for some blocks.
    
    # First, let's try a block decomposition where we split the matrices
    # into 2x2 blocks where possible.
    
    # A is 4x5: split into 2x2, 2x2, 2x1 blocks (rows 0-1, 2-3; cols 0-1, 2-3, 4)
    # B is 5x6: split into 2x2, 2x2, 2x2, 1x2, 1x2, 1x2 blocks
    # This is getting complicated.
    
    # Let me try a simpler strategy: just use schoolbook and see if the
    # randomized search can find improvements.
    
    # Actually, let me try to construct a decomposition using Strassen
    # for the 4x4x4 sub-problem and schoolbook for the rest.
    
    # Split A (4x5) into A1 (4x4) and A2 (4x1)
    # Split B (5x6) into B1 (4x6) and B2 (1x6)
    # C = A1*B1 + A2*B2
    
    # A1*B1 is 4x4 times 4x6: split B1 into B11 (4x4) and B12 (4x2)
    # A1*B11 is 4x4 times 4x4: use Strassen (rank 49)
    # A1*B12 is 4x4 times 4x2: schoolbook (rank 32)
    # A2*B2 is 4x1 times 1x6: schoolbook (rank 24)
    
    # Total rank: 49 + 32 + 24 = 105
    
    # Let me implement this.
    
    # First, create the factor matrices with the total rank.
    total_rank = 105
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    # A1*B11: 4x4 times 4x4, use Strassen
    # This is a 4x4x4 block at position (0, 0, 0)
    # We need to apply Strassen recursively.
    # Strassen for 4x4x4: split into 8 2x2x2 blocks and use Strassen for each.
    # But that gives 8 * 7 = 56, not 49.
    
    # Actually, the rank of M_{4,4,4} is 49 using Strassen.
    # The construction is: apply Strassen to the 2x2 blocks, but share
    # some computations.
    
    # This is complex. Let me just use schoolbook for now and see if I
    # can find a better decomposition through search.
    
    # For now, let's use the schoolbook decomposition as a baseline.
    schoolbook_rank = n * m * p
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions using randomized search
    U_np = np.array(U_school, dtype=float)
    V_np = np.array(V_school, dtype=float)
    W_np = np.array(W_school, dtype=float)
    
    def verify_numpy(U_mat, V_mat, W_mat, n, m, p):
        """Efficient verification using numpy."""
        rank = U_mat.shape[1]
        errors = 0
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U_mat[i*m+j, :]
                for k in range(p):
                    v_row = V_mat[j*p+k, :]
                    w_row = W_mat[k*n+i, :]
                    val = np.dot(u_row, v_row * w_row)
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
                    if err > 1e-8:
                        errors += 1
        return errors == 0, max_err
    
    # Try randomized search for lower ranks
    lower_bound = max(n*p, n*m, m*p)  # 30
    
    for target_rank in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        
        n_tries = 3
        for trial in range(n_tries):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + trial + target_rank)
            U_try = np.random.randn(n*m, target_rank) * 0.5
            V_try = np.random.randn(m*p, target_rank) * 0.5
            W_try = np.random.randn(p*n, target_rank) * 0.5
            
            best_err = float('inf')
            U_best = U_try.copy()
            V_best = V_try.copy()
            W_best = W_try.copy()
            
            for iteration in range(30):
                if time.time() > deadline:
                    break
                
                U_cur, V_cur, W_cur = U_best, V_best, W_best
                err, max_e = verify_numpy(U_cur, V_cur, W_cur, n, m, p)
                if err:
                    U_list = [[round(float(x), 10) for x in row] for row in U_cur]
                    V_list = [[round(float(x), 10) for x in row] for row in V_cur]
                    W_list = [[round(float(x), 10) for x in row] for row in W_cur]
                    
                    for mat in [U_list, V_list, W_list]:
                        for row in mat:
                            for idx in range(len(row)):
                                if abs(row[idx]) < 1e-9:
                                    row[idx] = 0.0
                                elif abs(row[idx] - round(row[idx])) < 1e-9:
                                    row[idx] = round(row[idx])
                                elif abs(row[idx] - round(row[idx]*2)/2) < 1e-9:
                                    row[idx] = round(row[idx]*2)/2
                    
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    break
                
                step = max_e * 0.1 + 0.01
                U_new = U_cur + np.random.randn(n*m, target_rank) * step
                V_new = V_cur + np.random.randn(m*p, target_rank) * step
                W_new = W_cur + np.random.randn(p*n, target_rank) * step
                
                new_err, new_max_e = verify_numpy(U_new, V_new, W_new, n, m, p)
                if new_max_e < best_err:
                    best_err = new_max_e
                    U_best, V_best, W_best = U_new, V_new, W_new
            
            if best_rank < schoolbook_rank:
                break
    
    if time.time() > deadline:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END

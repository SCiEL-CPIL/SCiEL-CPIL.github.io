"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 4x4 matrix by a 4x7 matrix.
    
    Strategy:
    - Split the 4x7 output into two parts: first 4 columns and last 3 columns.
    - For the first 4 columns (4x4 x 4x4), use a rank-49 Strassen decomposition
      (2x2 block Strassen applied recursively, giving rank 7^2 = 49).
    - For the last 3 columns (4x4 x 4x3), use schoolbook decomposition (rank 48).
    - Total rank = 49 + 48 = 97.
    
    Interface:
    U: (16, rank) - rows indexed by i*m+j where i in [0,4), j in [0,4)
    V: (28, rank) - rows indexed by j*p+k where j in [0,4), k in [0,7)
    W: (28, rank) - rows indexed by k*n+i where k in [0,7), i in [0,4)
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=7
    
    # We'll build U, V, W as lists of lists
    rank_total = 0
    U = []
    V = []
    W = []
    
    # Helper to append rows to U, V, W
    def make_empty_rows(num_rows):
        return [[0] * rank_total for _ in range(num_rows)]
    
    # First, compute the two decompositions
    # Part 1: Strassen for 4x4 x 4x4 (first 4 columns of B, first 4 rows of C)
    U1, V1, W1, rank1 = strassen_4x4x4()
    
    # Part 2: Schoolbook for 4x4 x 4x3 (last 3 columns of B, last 3 rows of C)
    U2, V2, W2, rank2 = schoolbook_4x4x3()
    
    rank_total = rank1 + rank2
    
    # Now resize and combine
    U = [[0] * rank_total for _ in range(n * m)]  # 16 x rank_total
    V = [[0] * rank_total for _ in range(m * p)]  # 28 x rank_total
    W = [[0] * rank_total for _ in range(p * n)]  # 28 x rank_total
    
    # Copy Part 1 into columns [0, rank1)
    for i in range(n * m):
        for c in range(rank1):
            U[i][c] = U1[i][c]
    for i in range(m * p):
        for c in range(rank1):
            V[i][c] = V1[i][c]
    for i in range(p * n):
        for c in range(rank1):
            W[i][c] = W1[i][c]
    
    # Copy Part 2 into columns [rank1, rank_total)
    offset = rank1
    for i in range(n * m):
        for c in range(rank2):
            U[i][offset + c] = U2[i][c]
    for i in range(m * p):
        for c in range(rank2):
            V[i][offset + c] = V2[i][c]
    for i in range(p * n):
        for c in range(rank2):
            W[i][offset + c] = W2[i][c]
    
    return {"U": U, "V": V, "W": W}


def strassen_4x4x4():
    """
    Return U, V, W for 4x4 x 4x4 matrix multiplication using rank-49 Strassen.
    U: (16, 49), V: (16, 49) for the 4x4 part (padded to 28 for V in full problem,
    but here we return only the 4x4 portion), W: (16, 49).
    
    Actually, we need to be careful with indices. Let me think about this more carefully.
    
    For 4x4 x 4x4:
    - A is 4x4, indexed by (i, j) with i,j in [0,4)
    - B is 4x4, indexed by (j, k) with j,k in [0,4)
    - C is 4x4, indexed by (i, k) with i,k in [0,4)
    
    U rows: i*m + j = i*4 + j, for i,j in [0,4) -> 16 rows
    V rows: j*p + k = j*4 + k, for j in [0,4), k in [0,4) -> 16 rows (we only use first 16 of the 28 in full problem)
    W rows: k*n + i = k*4 + i, for k in [0,4), i in [0,4) -> 16 rows (we only use first 16 of the 28 in full problem)
    
    So U is 16x49, V is 16x49 (but in the full problem, V will be 28x49 with zeros in rows 16-27),
    W is 16x49 (but in the full problem, W will be 28x49 with zeros in rows 16-27).
    
    Let me construct the Strassen decomposition for 4x4x4.
    """
    # Strassen for 2x2: 7 products
    # For 4x4: treat as 2x2 blocks of 2x2 matrices
    # A = [[A00, A01], [A10, A11]] where each block is 2x2
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # Strassen products:
    # M0 = (A00 + A11)(B00 + B11)
    # M1 = (A10 + A11)B00
    # M2 = A00(B01 - B11)
    # M3 = A11(B10 - B00)
    # M4 = (A00 + A01)B11
    # M5 = (A10 - A00)(B00 + B01)
    # M6 = (A01 - A11)(B10 + B11)
    
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # For 4x4x4, we apply this block-wise. Each block is 2x2, and we use Strassen for the 2x2 blocks too?
    # No, wait. The rank of 4x4x4 using Strassen is 7^2 = 49 if we apply Strassen to 2x2 blocks
    # and then use the 7 products, each of which is a product of 2x2 matrices. But if we use
    # Strassen for the 2x2 subproducts, the total rank would be 7 * 7 = 49.
    
    # Let me construct this properly.
    # We have 7 "big" products, each of which is a product of two 2x2 matrices.
    # For each big product, we use 7 scalar products (Strassen for 2x2).
    # Total: 7 * 7 = 49 scalar products.
    
    # Let's define the indices.
    # A[i,j] where i,j in {0,1,2,3}
    # A00 = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    # A01 = [[A[0,2], A[0,3]], [A[1,2], A[1,3]]]
    # A10 = [[A[2,0], A[2,1]], [A[3,0], A[3,1]]]
    # A11 = [[A[2,2], A[2,3]], [A[3,2], A[3,3]]]
    
    # Similarly for B.
    
    # Let's build the U, V, W matrices.
    # Each of the 49 columns corresponds to one scalar product.
    
    rank = 49
    U = [[0] * rank for _ in range(16)]  # 4*4
    V = [[0] * rank for _ in range(16)]  # 4*4 (first 4 cols of B)
    W = [[0] * rank for _ in range(16)]  # 4*4 (first 4 rows of C)
    
    # Strassen 2x2: 7 products
    # P0 = (a00+a11)(b00+b11)
    # P1 = (a10+a11)b00
    # P2 = a00(b01-b11)
    # P3 = a11(b10-b00)
    # P4 = (a00+a01)b11
    # P5 = (a10-a00)(b00+b01)
    # P6 = (a01-a11)(b10+b11)
    
    # C00 = P0 + P3 - P4 + P6
    # C01 = P2 + P4
    # C10 = P1 + P3
    # C11 = P0 - P1 + P2 + P5
    
    # Now, for each of the 7 big Strassen products (M0..M6), we have two 2x2 matrices
    # X and Y, and we compute X*Y using Strassen (7 scalar products).
    
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # Let me define a helper to add a Strassen 2x2x2 decomposition.
    
    def add_strassen_2x2(U, V, W, offset, X_indices, Y_indices, C_indices, signs):
        """
        Add a Strassen 2x2x2 decomposition for X*Y = C.
        X_indices: 4 indices (x00, x01, x10, x11) into U rows
        Y_indices: 4 indices (y00, y01, y10, y11) into V rows
        C_indices: 4 indices (c00, c01, c10, c11) into W rows
        signs: list of 7 signs for P0..P6 in the output (actually, this is more complex)
        """
        pass
    
    # This is getting complex. Let me just explicitly build the 49 products.
    
    # Let me define the 7 big products more carefully.
    
    # For each big product Mi, we have X_i and Y_i (both 2x2 matrices),
    # and we compute Z_i = X_i * Y_i using Strassen (7 scalar products).
    # Then C is assembled from Z_0..Z_6.
    
    # Let me list the 7 big products with their X and Y:
    big_products = [
        # (X_type, Y_type) where X_type/Y_type specify which 2x2 blocks and combinations
        # M0: X = A00+A11, Y = B00+B11
        # M1: X = A10+A11, Y = B00
        # M2: X = A00, Y = B01-B11
        # M3: X = A11, Y = B10-B00
        # M4: X = A00+A01, Y = B11
        # M5: X = A10-A00, Y = B00+B01
        # M6: X = A01-A11, Y = B10+B11
    ]
    
    # For each big product, I need to specify:
    # - The 4 entries of X (as linear combinations of A entries)
    # - The 4 entries of Y (as linear combinations of B entries)
    # - How Z contributes to C
    
    # This is very tedious to do by hand. Let me use a known construction.
    
    # Actually, let me just implement it step by step.
    
    # I'll create a function that, given the indices for X and Y (as 2x2 blocks with signs),
    # adds the 7 Strassen products.
    
    # Let me define the A block indices:
    # A00: rows [0,1], cols [0,1] -> indices 0,1,4,5 in U (i*4+j: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5)
    # A01: rows [0,1], cols [2,3] -> indices 2,3,6,7
    # A10: rows [2,3], cols [0,1] -> indices 8,9,12,13
    # A11: rows [2,3], cols [2,3] -> indices 10,11,14,15
    
    # B block indices (for the 4x4 B, first 4 cols):
    # B00: rows [0,1], cols [0,1] -> indices 0,1,4,5 in V (j*4+k: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5)
    # B01: rows [0,1], cols [2,3] -> indices 2,3,6,7
    # B10: rows [2,3], cols [0,1] -> indices 8,9,12,13
    # B11: rows [2,3], cols [2,3] -> indices 10,11,14,15
    
    # C block indices (for the 4x4 C, first 4 rows):
    # C00: rows [0,1], cols [0,1] -> indices 0,1,4,5 in W (k*4+i: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5)
    # C01: rows [0,1], cols [2,3] -> indices 2,3,6,7
    # C10: rows [2,3], cols [0,1] -> indices 8,9,12,13
    # C11: rows [2,3], cols [2,3] -> indices 10,11,14,15
    
    # Now, for each big product, let's specify the X and Y blocks with coefficients.
    
    # I'll represent each 2x2 block as a 4-tuple of (index, coefficient) pairs.
    
    def make_block(indices, coeff=1):
        return [(idx, coeff) for idx in indices]
    
    # A00 = [0,1,4,5], A01 = [2,3,6,7], A10 = [8,9,12,13], A11 = [10,11,14,15]
    A00 = [0, 1, 4, 5]
    A01 = [2, 3, 6, 7]
    A10 = [8, 9, 12, 13]
    A11 = [10, 11, 14, 15]
    
    B00 = [0, 1, 4, 5]
    B01 = [2, 3, 6, 7]
    B10 = [8, 9, 12, 13]
    B11 = [10, 11, 14, 15]
    
    C00 = [0, 1, 4, 5]
    C01 = [2, 3, 6, 7]
    C10 = [8, 9, 12, 13]
    C11 = [10, 11, 14, 15]
    
    # For each big product, define X and Y as lists of (block_indices, sign)
    # X and Y are 2x2 matrices, so they have 4 entries each.
    # Each entry is a linear combination of A or B entries.
    
    # Let me define the 7 big products:
    
    # M0: X = A00 + A11, Y = B00 + B11
    # M1: X = A10 + A11, Y = B00
    # M2: X = A00, Y = B01 - B11
    # M3: X = A11, Y = B10 - B00
    # M4: X = A00 + A01, Y = B11
    # M5: X = A10 - A00, Y = B00 + B01
    # M6: X = A01 - A11, Y = B10 + B11
    
    # For each big product, X and Y are 2x2 matrices.
    # Let me define them as 4-tuples of (entry_indices, coefficient) for each of the 4 positions.
    
    # Actually, let me just explicitly build the U, V, W for each of the 49 scalar products.
    
    # I'll use a list of tuples: (u_indices, v_indices, w_indices, w_coeff)
    # where u_indices is a list of (idx, coeff) for U, etc.
    
    # This is getting very complex. Let me try a different approach.
    
    # Let me just use the known Strassen formula and build it up.
    
    # For 4x4x4 Strassen with rank 49, here's a cleaner way to think about it:
    
    # We have 7 "middle" products, each of rank 7 (Strassen for 2x2).
    # Total rank = 7 * 7 = 49.
    
    # Let me define the 7 middle products explicitly.
    
    # Middle product 0: (A00 + A11) * (B00 + B11)
    # Middle product 1: (A10 + A11) * B00
    # Middle product 2: A00 * (B01 - B11)
    # Middle product 3: A11 * (B10 - B00)
    # Middle product 4: (A00 + A01) * B11
    # Middle product 5: (A10 - A00) * (B00 + B01)
    # Middle product 6: (A01 - A11) * (B10 + B11)
    
    # For each middle product, I need to:
    # 1. Form the X matrix (2x2) as a linear combination of A blocks
    # 2. Form the Y matrix (2x2) as a linear combination of B blocks
    # 3. Apply Strassen to X*Y, which gives 7 scalar products
    # 4. Add the contributions to C
    
    # Let me define a helper function that, given X and Y (as 2x2 matrices with entries
    # that are linear combinations of U/V rows), adds the 7 Strassen products to U, V, W.
    
    def add_strassen_2x2_to_decomposition(U, V, W, col_offset, X, Y, C_contribution):
        """
        X: 2x2 matrix where each entry is a list of (u_row, coeff)
        Y: 2x2 matrix where each entry is a list of (v_row, coeff)
        C_contribution: 2x2 matrix where each entry is a list of (w_row, coeff) for how this product contributes to C
        
        Actually, C_contribution should specify how the 4 entries of Z = X*Y contribute to C.
        """
        pass
    
    # This is really getting complex. Let me just hardcode the 49 products.
    
    # Actually, I think the cleanest way is to use NumPy to construct the matrices and then convert.
    
    import numpy as np
    
    # Let me construct U, V, W as numpy arrays.
    
    U_np = np.zeros((16, 49), dtype=float)
    V_np = np.zeros((16, 49), dtype=float)
    W_np = np.zeros((16, 49), dtype=float)
    
    # Define the 7 middle products
    # Each middle product is (X, Y) where X and Y are 2x2 matrices with entries
    # that are linear combinations of A/B entries.
    
    # Let me define A entries: A[i,j] -> U row i*4+j
    # B entries: B[j,k] -> V row j*4+k (for k in 0..3)
    # C entries: C[i,k] -> W row k*4+i (for k in 0..3)
    
    # For simplicity, let me define the 7 middle products as:
    # For each, X is a 2x2 matrix with entries being (row_index, coeff) pairs
    # Y is a 2x2 matrix with entries being (row_index, coeff) pairs
    # And then I apply Strassen to compute Z = X*Y.
    
    # Let me define a function to apply Strassen to a 2x2 X*Y.
    
    def strassen_2x2(X, Y):
        """
        X, Y: 2x2 matrices with entries being (index, coeff) pairs
        Returns 7 products, each being a list of (u_idx, v_idx, coeff) triples
        """
        # X = [[x00, x01], [x10, x11]]
        # Y = [[y00, y01], [y10, y11]]
        
        x00, x01 = X[0]
        x10, x11 = X[1]
        y00, y01 = Y[0]
        y10, y11 = Y[1]
        
        # P0 = (x00+x11)(y00+y11)
        # P1 = (x10+x11)y00
        # P2 = x00(y01-y11)
        # P3 = x11(y10-y00)
        # P4 = (x00+x01)y11
        # P5 = (x10-x00)(y00+y01)
        # P6 = (x01-x11)(y10+y11)
        
        # Each Pi is a product of two linear combinations.
        # (sum of a_i * u_i) * (sum of b_j * v_j) = sum_{i,j} a_i*b_j * u_i * v_j
        
        # For each Pi, we need to expand it into scalar products.
        
        def expand_product(x_entries, y_entries):
            """
            x_entries: list of (u_idx, coeff)
            y_entries: list of (v_idx, coeff)
            Returns list of (u_idx, v_idx, coeff)
            """
            result = []
            for (ui, uc) in x_entries:
                for (vi, vc) in y_entries:
                    result.append((ui, vi, uc * vc))
            return result
        
        def add_entries(e1, e2, sign2=1):
            """Add two lists of entries, with sign2 for e2"""
            result = []
            for (idx, coeff) in e1:
                result.append((idx, coeff))
            for (idx, coeff) in e2:
                result.append((idx, sign2 * coeff))
            return result
        
        def sub_entries(e1, e2):
            return add_entries(e1, e2, sign2=-1)
        
        # P0
        x_p0 = add_entries(x00, x11)
        y_p0 = add_entries(y00, y11)
        p0 = expand_product(x_p0, y_p0)
        
        # P1
        x_p1 = add_entries(x10, x11)
        y_p1 = y00
        p1 = expand_product(x_p1, y_p1)
        
        # P2
        x_p2 = x00
        y_p2 = sub_entries(y01, y11)
        p2 = expand_product(x_p2, y_p2)
        
        # P3
        x_p3 = x11
        y_p3 = sub_entries(y10, y00)
        p3 = expand_product(x_p3, y_p3)
        
        # P4
        x_p4 = add_entries(x00, x01)
        y_p4 = y11
        p4 = expand_product(x_p4, y_p4)
        
        # P5
        x_p5 = sub_entries(x10, x00)
        y_p5 = add_entries(y00, y01)
        p5 = expand_product(x_p5, y_p5)
        
        # P6
        x_p6 = sub_entries(x01, x11)
        y_p6 = add_entries(y10, y11)
        p6 = expand_product(x_p6, y_p6)
        
        return [p0, p1, p2, p3, p4, p5, p6]
    
    # Now define the 7 middle products.
    # X and Y are 2x2 matrices with entries as lists of (index, coeff).
    
    # A blocks:
    A00_entries = [(0, 1), (1, 1), (4, 1), (5, 1)]
    A01_entries = [(2, 1), (3, 1), (6, 1), (7, 1)]
    A10_entries = [(8, 1), (9, 1), (12, 1), (13, 1)]
    A11_entries = [(10, 1), (11, 1), (14, 1), (15, 1)]
    
    # B blocks:
    B00_entries = [(0, 1), (1, 1), (4, 1), (5, 1)]
    B01_entries = [(2, 1), (3, 1), (6, 1), (7, 1)]
    B10_entries = [(8, 1), (9, 1), (12, 1), (13, 1)]
    B11_entries = [(10, 1), (11, 1), (14, 1), (15, 1)]
    
    # For each middle product, X and Y are 2x2 matrices.
    # Each entry of X is a 2x2 block of A, so it's a 4-tuple of entries.
    # But wait, X is a 2x2 matrix where each entry is itself a 2x2 matrix.
    # So X[0][0] is the (0,0) entry of the 2x2 "super" matrix, which is a 2x2 submatrix of A.
    
    # I think I'm overcomplicating this. Let me think again.
    
    # When we say X = A00 + A11, we mean X is a 2x2 matrix where:
    # X[0][0] = A00[0][0] + A11[0][0] = A[0,0] + A[2,2]
    # X[0][1] = A00[0][1] + A11[0][1] = A[0,1] + A[2,3]
    # X[1][0] = A00[1][0] + A11[1][0] = A[1,0] + A[3,2]
    # X[1][1] = A00[1][1] + A11[1][1] = A[1,1] + A[3,3]
    
    # So X[0][0] is a scalar (linear combination of A entries), not a 2x2 matrix.
    # Similarly for all entries.
    
    # So X is a 2x2 matrix of scalars (each scalar is a linear combination of A entries).
    # And Y is a 2x2 matrix of scalars (each scalar is a linear combination of B entries).
    # And we compute Z = X*Y using Strassen, which gives 7 scalar products.
    
    # So for each middle product, X and Y are 2x2 matrices where each entry
    # is a list of (u_row, coeff) or (v_row, coeff).
    
    # Let me redefine.
    
    def make_x_entry(*blocks):
        """Combine multiple A blocks into one entry."""
        result = []
        for block in blocks:
            for (idx, coeff) in block:
                result.append((idx, coeff))
        return result
    
    def make_y_entry(*blocks):
        """Combine multiple B blocks into one entry."""
        result = []
        for block in blocks:
            for (idx, coeff) in block:
                result.append((idx, coeff))
        return result
    
    # Now define the 7 middle products.
    # Each is (X, Y) where X is [[x00, x01], [x10, x11]] and Y is [[y00, y01], [y10, y11]]
    
    middle_products = [
        # M0: X = A00+A11, Y = B00+B11
        (
            [[make_x_entry(A00_entries, A11_entries) for _ in range(2)] for _ in range(2)],  # This is wrong, A00+A11 means each entry is the sum of corresponding entries
            [[make_y_entry(B00_entries, B11_entries) for _ in range(2)] for _ in range(2)],
        ),
    ]
    
    # Wait, this is still wrong. A00 + A11 means:
    # X[0][0] = A00[0][0] + A11[0][0]
    # X[0][1] = A00[0][1] + A11[0][1]
    # etc.
    
    # So I need to combine entry-wise.
    
    def combine_entries(e1, e2):
        """Combine two lists of (idx, coeff) by summing."""
        from collections import defaultdict
        d = defaultdict(float)
        for (idx, coeff) in e1:
            d[idx] += coeff
        for (idx, coeff) in e2:
            d[idx] += coeff
        return [(idx, coeff) for idx, coeff in d.items() if coeff != 0]
    
    # Let me redefine the A and B blocks as 2x2 matrices of entries.
    
    A00_mat = [
        [(0, 1)],  # A[0,0]
        [(1, 1)],  # A[0,1]
    ]
    A00_mat = [
        [[(0, 1)], [(1, 1)]],
        [[(4, 1)], [(5, 1)]],
    ]
    A01_mat = [
        [[(2, 1)], [(3, 1)]],
        [[(6, 1)], [(7, 1)]],
    ]
    A10_mat = [
        [[(8, 1)], [(9, 1)]],
        [[(12, 1)], [(13, 1)]],
    ]
    A11_mat = [
        [[(10, 1)], [(11, 1)]],
        [[(14, 1)], [(15, 1)]],
    ]
    
    B00_mat = [
        [[(0, 1)], [(1, 1)]],
        [[(4, 1)], [(5, 1)]],
    ]
    B01_mat = [
        [[(2, 1)], [(3, 1)]],
        [[(6, 1)], [(7, 1)]],
    ]
    B10_mat = [
        [[(8, 1)], [(9, 1)]],
        [[(12, 1)], [(13, 1)]],
    ]
    B11_mat = [
        [[(10, 1)], [(11, 1)]],
        [[(14, 1)], [(15, 1)]],
    ]
    
    def add_mats(M1, M2):
        result = []
        for i in range(2):
            row = []
            for j in range(2):
                row.append(combine_entries(M1[i][j], M2[i][j]))
            result.append(row)
        return result
    
    def sub_mats(M1, M2):
        result = []
        for i in range(2):
            row = []
            for j in range(2):
                neg_M2 = [(idx, -coeff) for (idx, coeff) in M2[i][j]]
                row.append(combine_entries(M1[i][j], neg_M2))
            result.append(row)
        return result
    
    # Now define the 7 middle products.
    MP0 = (add_mats(A00_mat, A11_mat), add_mats(B00_mat, B11_mat))
    MP1 = (add_mats(A10_mat, A11_mat), B00_mat)
    MP2 = (A00_mat, sub_mats(B01_mat, B11_mat))
    MP3 = (A11_mat, sub_mats(B10_mat, B00_mat))
    MP4 = (add_mats(A00_mat, A01_mat), B11_mat)
    MP5 = (sub_mats(A10_mat, A00_mat), add_mats(B00_mat, B01_mat))
    MP6 = (sub_mats(A01_mat, A11_mat), add_mats(B10_mat, B11_mat))
    
    all_middle_products = [MP0, MP1, MP2, MP3, MP4, MP5, MP6]
    
    # Now, for each middle product, apply Strassen to get 7 scalar products.
    # Then add them to U, V, W.
    
    col = 0
    for (X, Y) in all_middle_products:
        products = strassen_2x2(X, Y)
        for prod in products:
            # prod is a list of (u_idx, v_idx, coeff)
            for (ui, vi, coeff) in prod:
                U_np[ui, col] += coeff
                V_np[vi, col] += coeff
            col += 1
    
    # Now we need to add the contributions to W.
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # Each Mi is a 2x2 matrix (the result of X_i * Y_i).
    # For each middle product i, the 7 Strassen products give us the 4 entries of Z_i.
    # P0, P1, P2, P3, P4, P5, P6 are the 7 products.
    # Z[0][0] = P0 + P3 - P4 + P6
    # Z[0][1] = P2 + P4
    # Z[1][0] = P1 + P3
    # Z[1][1] = P0 - P1 + P2 + P5
    
    # So for each middle product i, we have 7 columns in U, V, W.
    # The contribution to W is:
    # W entries for C00 get: +1*P0_i + +1*P3_i + (-1)*P4_i + +1*P6_i
    # etc.
    
    # Let me redo this. For each middle product i (0..6), we have 7 columns (7*i .. 7*i+6).
    # Let me call them c0..c6 for the 7 Strassen products.
    
    # Z_i[0][0] = c0 + c3 - c4 + c6
    # Z_i[0][1] = c2 + c4
    # Z_i[1][0] = c1 + c3
    # Z_i[1][1] = c0 - c1 + c2 + c5
    
    # C00 = M0 + M3 - M4 + M6
    # So C00[0][0] = Z0[0][0] + Z3[0][0] - Z4[0][0] + Z6[0][0]
    # C00[0][1] = Z0[0][1] + Z3[0][1] - Z4[0][1] + Z6[0][1]
    # C00[1][0] = Z0[1][0] + Z3[1][0] - Z4[1][0] + Z6[1][0]
    # C00[1][1] = Z0[1][1] + Z3[1][1] - Z4[1][1] + Z6[1][1]
    
    # And so on.
    
    # Let me build W_np.
    
    W_np = np.zeros((16, 49), dtype=float)
    
    # C block indices in W:
    # C00: rows 0,1,4,5 (k*4+i for k in 0..1, i in 0..1)
    # C01: rows 2,3,6,7
    # C10: rows 8,9,12,13
    # C11: rows 10,11,14,15
    
    C00_rows = [0, 1, 4, 5]
    C01_rows = [2, 3, 6, 7]
    C10_rows = [8, 9, 12, 13]
    C11_rows = [10, 11, 14, 15]
    
    # For each middle product i, the 7 columns are 7*i, 7*i+1, ..., 7*i+6.
    # Let me define the contribution of each of the 7 Strassen products to Z_i.
    
    # Z_i[0][0] = c0 + c3 - c4 + c6
    # Z_i[0][1] = c2 + c4
    # Z_i[1][0] = c1 + c3
    # Z_i[1][1] = c0 - c1 + c2 + c5
    
    # So for each middle product i, and for each of the 4 Z entries,
    # we add the appropriate combination of columns to W.
    
    for i in range(7):
        base = 7 * i
        # Z[i][0][0] = c0 + c3 - c4 + c6
        coeffs_00 = [(0, 1), (3, 1), (4, -1), (6, 1)]
        # Z[i][0][1] = c2 + c4
        coeffs_01 = [(2, 1), (4, 1)]
        # Z[i][1][0] = c1 + c3
        coeffs_10 = [(1, 1), (3, 1)]
        # Z[i][1][1] = c0 - c1 + c2 + c5
        coeffs_11 = [(0, 1), (1, -1), (2, 1), (5, 1)]
        
        # Now, Z_i contributes to C based on the Strassen formula.
        # C00 = M0 + M3 - M4 + M6
        # C01 = M2 + M4
        # C10 = M1 + M3
        # C11 = M0 - M1 + M2 + M5
        
        contributions = {
            0: [(C00_rows, 1), (C11_rows, 1)],   # M0 contributes to C00 and C11
            1: [(C10_rows, 1), (C11_rows, -1)],  # M1 contributes to C10 and C11
            2: [(C01_rows, 1), (C11_rows, 1)],   # M2 contributes to C01 and C11
            3: [(C00_rows, 1), (C10_rows, 1)],   # M3 contributes to C00 and C10
            4: [(C00_rows, -1), (C01_rows, 1)],  # M4 contributes to C00 and C01
            5: [(C11_rows, 1)],                   # M5 contributes to C11
            6: [(C00_rows, 1)],                   # M6 contributes to C00
        }
        
        for (w_rows, sign) in contributions[i]:
            for (col_offset, coeff) in coeffs_00:
                for w_row in w_rows:
                    W_np[w_row, base + col_offset] += sign * coeff
        
        # Wait, this is wrong. Each of the 4 Z entries (Z[0][0], Z[0][1], Z[1][0], Z[1][1])
        # corresponds to different W rows.
        
        # Z[i][0][0] -> C block (0,0) -> W rows C00_rows
        # Z[i][0][1] -> C block (0,1) -> W rows C01_rows
        # Z[i][1][0] -> C block (1,0) -> W rows C10_rows
        # Z[i][1][1] -> C block (1,1) -> W rows C11_rows
        
        # So:
        # Z[i][0][0] contributes to C00
        # Z[i][0][1] contributes to C01
        # Z[i][1][0] contributes to C10
        # Z[i][1][1] contributes to C11
        
        # And the Strassen formula says:
        # C00 = M0 + M3 - M4 + M6
        # C01 = M2 + M4
        # C10 = M1 + M3
        # C11 = M0 - M1 + M2 + M5
        
        # So M0[0][0] contributes to C00[0][0]
        # M0[0][1] contributes to C01[0][0]... wait, no.
        
        # M_i is a 2x2 matrix. M_i[0][0] is the (0,0) entry of the i-th middle product.
        # C[0][0] = M0[0][0] + M3[0][0] - M4[0][0] + M6[0][0]
        # C[0][1] = M0[0][1] + M3[0][1] - M4[0][1] + M6[0][1]
        # C[0][2] = M0[0][2]... wait, C is 4x4, so C[i][k] where i,k in 0..3.
        
        # I think I need to be more careful. Let me index C as C[row][col] where row is i (0..3) and col is k (0..3).
        # C[i][k] = sum over j of A[i][j] * B[j][k]
        
        # The block structure:
        # C00 = C[0:2, 0:2] = M0[0:2, 0:2] + M3[0:2, 0:2] - M4[0:2, 0:2] + M6[0:2, 0:2]
        # C01 = C[0:2, 2:4] = M2[0:2, 0:2] + M4[0:2, 0:2]
        # C10 = C[2:4, 0:2] = M1[0:2, 0:2] + M3[0:2, 0:2]
        # C11 = C[2:4, 2:4] = M0[0:2, 0:2] - M1[0:2, 0:2] + M2[0:2, 0:2] + M5[0:2, 0:2]
        
        # So each M_i is a 2x2 matrix, and its entries contribute to specific 2x2 blocks of C.
        
        # M_i[0][0] contributes to:
        # - C[0][0] (from C00)
        # - C[2][2] (from C11)
        
        # M_i[0][1] contributes to:
        # - C[0][1] (from C00)
        # - C[2][3] (from C11)
        
        # M_i[1][0] contributes to:
        # - C[1][0] (from C00)
        # - C[3][2] (from C11)
        
        # M_i[1][1] contributes to:
        # - C[1][1] (from C00)
        # - C[3][3] (from C11)
        
        # And similarly for the other blocks.
        
        # Let me define the W row for each C[i][k]:
        # C[i][k] -> W row k*n + i = k*4 + i
        
        C_row_col_to_W = {}
        for i in range(4):
            for k in range(4):
                C_row_col_to_W[(i, k)] = k * 4 + i
        
        # Now, for each middle product i, and for each of its 4 entries,
        # I need to add the appropriate combination of the 7 Strassen columns to W.
        
        # Let me redo the W construction from scratch.
        
    # Let me just restart the W construction more carefully.
    
    W_np = np.zeros((16, 49), dtype=float)
    
    # For each middle product mp_i (i in 0..6), the 7 Strassen products are in columns 7*i .. 7*i+6.
    # Let me call them p0..p6.
    
    # Z_i[0][0] = p0 + p3 - p4 + p6
    # Z_i[0][1] = p2 + p4
    # Z_i[1][0] = p1 + p3
    # Z_i[1][1] = p0 - p1 + p2 + p5
    
    # C[0][0] = Z0[0][0] + Z3[0][0] - Z4[0][0] + Z6[0][0]
    # C[0][1] = Z0[0][1] + Z3[0][1] - Z4[0][1] + Z6[0][1]
    # C[0][2] = Z2[0][0] + Z4[0][0]
    # C[0][3] = Z2[0][1] + Z4[0][1]
    # C[1][0] = Z0[1][0] + Z3[1][0] - Z4[1][0] + Z6[1][0]
    # C[1][1] = Z0[1][1] + Z3[1][1] - Z4[1][1] + Z6[1][1]
    # C[1][2] = Z2[1][0] + Z4[1][0]
    # C[1][3] = Z2[1][1] + Z4[1][1]
    # C[2][0] = Z1[0][0] + Z3[0][0]
    # C[2][1] = Z1[0][1] + Z3[0][1]
    # C[2][2] = Z0[0][0] - Z1[0][0] + Z2[0][0] + Z5[0][0]
    # C[2][3] = Z0[0][1] - Z1[0][1] + Z2[0][1] + Z5[0][1]
    # C[3][0] = Z1[1][0] + Z3[1][0]
    # C[3][1] = Z1[1][1] + Z3[1][1]
    # C[3][2] = Z0[1][0] - Z1[1][0] + Z2[1][0] + Z5[1][0]
    # C[3][3] = Z0[1][1] - Z1[1][1] + Z2[1][1] + Z5[1][1]
    
    # Let me build W_np based on these formulas.
    
    # Helper: add the contribution of Z_i[a][b] to W row (i_c, k_c) with sign s
    # Z_i[a][b] is a linear combination of the 7 Strassen products for middle product i.
    
    def add_z_contribution(W_np, mp_i, a, b, w_row, sign):
        """Add sign * Z_i[a][b] to W_np[w_row, :]"""
        base = 7 * mp_i
        if a == 0 and b == 0:
            # Z[0][0] = p0 + p3 - p4 + p6
            coeffs = [(0, 1), (3, 1), (4, -1), (6, 1)]
        elif a == 0 and b == 1:
            # Z[0][1] = p2 + p4
            coeffs = [(2, 1), (4, 1)]
        elif a == 1 and b == 0:
            # Z[1][0] = p1 + p3
            coeffs = [(1, 1), (3, 1)]
        elif a == 1 and b == 1:
            # Z[1][1] = p0 - p1 + p2 + p5
            coeffs = [(0, 1), (1, -1), (2, 1), (5, 1)]
        else:
            return
        for (col_offset, coeff) in coeffs:
            W_np[w_row, base + col_offset] += sign * coeff
    
    # C[0][0]
    w_row = C_row_col_to_W[(0, 0)]
    add_z_contribution(W_np, 0, 0, 0, w_row, 1)
    add_z_contribution(W_np, 3, 0, 0, w_row, 1)
    add_z_contribution(W_np, 4, 0, 0, w_row, -1)
    add_z_contribution(W_np, 6, 0, 0, w_row, 1)
    
    # C[0][1]
    w_row = C_row_col_to_W[(0, 1)]
    add_z_contribution(W_np, 0, 0, 1, w_row, 1)
    add_z_contribution(W_np, 3, 0, 1, w_row, 1)
    add_z_contribution(W_np, 4, 0, 1, w_row, -1)
    add_z_contribution(W_np, 6, 0, 1, w_row, 1)
    
    # C[0][2]
    w_row = C_row_col_to_W[(0, 2)]
    add_z_contribution(W_np, 2, 0, 0, w_row, 1)
    add_z_contribution(W_np, 4, 0, 0, w_row, 1)
    
    # C[0][3]
    w_row = C_row_col_to_W[(0, 3)]
    add_z_contribution(W_np, 2, 0, 1, w_row, 1)
    add_z_contribution(W_np, 4, 0, 1, w_row, 1)
    
    # C[1][0]
    w_row = C_row_col_to_W[(1, 0)]
    add_z_contribution(W_np, 0, 1, 0, w_row, 1)
    add_z_contribution(W_np, 3, 1, 0, w_row, 1)
    add_z_contribution(W_np, 4, 1, 0, w_row, -1)
    add_z_contribution(W_np, 6, 1, 0, w_row, 1)
    
    # C[1][1]
    w_row = C_row_col_to_W[(1, 1)]
    add_z_contribution(W_np, 0, 1, 1, w_row, 1)
    add_z_contribution(W_np, 3, 1, 1, w_row, 1)
    add_z_contribution(W_np, 4, 1, 1, w_row, -1)
    add_z_contribution(W_np, 6, 1, 1, w_row, 1)
    
    # C[1][2]
    w_row = C_row_col_to_W[(1, 2)]
    add_z_contribution(W_np, 2, 1, 0, w_row, 1)
    add_z_contribution(W_np, 4, 1, 0, w_row, 1)
    
    # C[1][3]
    w_row = C_row_col_to_W[(1, 3)]
    add_z_contribution(W_np, 2, 1, 1, w_row, 1)
    add_z_contribution(W_np, 4, 1, 1, w_row, 1)
    
    # C[2][0]
    w_row = C_row_col_to_W[(2, 0)]
    add_z_contribution(W_np, 1, 0, 0, w_row, 1)
    add_z_contribution(W_np, 3, 0, 0, w_row, 1)
    
    # C[2][1]
    w_row = C_row_col_to_W[(2, 1)]
    add_z_contribution(W_np, 1, 0, 1, w_row, 1)
    add_z_contribution(W_np, 3, 0, 1, w_row, 1)
    
    # C[2][2]
    w_row = C_row_col_to_W[(2, 2)]
    add_z_contribution(W_np, 0, 0, 0, w_row, 1)
    add_z_contribution(W_np, 1, 0, 0, w_row, -1)
    add_z_contribution(W_np, 2, 0, 0, w_row, 1)
    add_z_contribution(W_np, 5, 0, 0, w_row, 1)
    
    # C[2][3]
    w_row = C_row_col_to_W[(2, 3)]
    add_z_contribution(W_np, 0, 0, 1, w_row, 1)
    add_z_contribution(W_np, 1, 0, 1, w_row, -1)
    add_z_contribution(W_np, 2, 0, 1, w_row, 1)
    add_z_contribution(W_np, 5, 0, 1, w_row, 1)
    
    # C[3][0]
    w_row = C_row_col_to_W[(3, 0)]
    add_z_contribution(W_np, 1, 1, 0, w_row, 1)
    add_z_contribution(W_np, 3, 1, 0, w_row, 1)
    
    # C[3][1]
    w_row = C_row_col_to_W[(3, 1)]
    add_z_contribution(W_np, 1, 1, 1, w_row, 1)
    add_z_contribution(W_np, 3, 1, 1, w_row, 1)
    
    # C[3][2]
    w_row = C_row_col_to_W[(3, 2)]
    add_z_contribution(W_np, 0, 1, 0, w_row, 1)
    add_z_contribution(W_np, 1, 1, 0, w_row, -1)
    add_z_contribution(W_np, 2, 1, 0, w_row, 1)
    add_z_contribution(W_np, 5, 1, 0, w_row, 1)
    
    # C[3][3]
    w_row = C_row_col_to_W[(3, 3)]
    add_z_contribution(W_np, 0, 1, 1, w_row, 1)
    add_z_contribution(W_np, 1, 1, 1, w_row, -1)
    add_z_contribution(W_np, 2, 1, 1, w_row, 1)
    add_z_contribution(W_np, 5, 1, 1, w_row, 1)
    
    # Convert to lists of lists
    U = U_np.tolist()
    V = V_np.tolist()
    W = W_np.tolist()
    
    return U, V, W, 49


def schoolbook_4x4x3():
    """
    Return U, V, W for 4x4 x 4x3 matrix multiplication using schoolbook (rank 48).
    U: (16, 48), V: (12, 48) for the 4x3 part (padded to 28 in full problem), W: (12, 48).
    
    Actually, in the full problem:
    V has 28 rows (4*7), and we use the last 3 columns of B, which are rows 16..27 of V.
    W has 28 rows (7*4), and we use the last 3 rows of C, which are rows 16..27 of W.
    
    So U is 16x48, V is 28x48 (with zeros in rows 0..15), W is 28x48 (with zeros in rows 0..15).
    
    But for this function, I'll return U (16x48), V (12x48 for the 4x3 part), W (12x48 for the 3x4 part).
    And then in construct(), I'll place them in the correct positions.
    
    Wait, let me re-read the indices.
    
    V rows: j*p + k where j in [0,4), k in [0,7).
    For the last 3 columns of B (k in 4,5,6), the V rows are j*7 + k for j in 0..3, k in 4..6.
    That's rows: 0*7+4=4, 0*7+5=5, 0*7+6=6, 1*7+4=11, 1*7+5=12, 1*7+6=13, 2*7+4=18, 2*7+5=19, 2*7+6=20, 3*7+4=25, 3*7+5=26, 3*7+6=27.
    
    W rows: k*n + i where k in [0,7), i in [0,4).
    For the last 3 rows of C (k in 4,5,6), the W rows are k*4 + i for k in 4..6, i in 0..3.
    That's rows: 4*4+0=16, 4*4+1=17, 4*4+2=18, 4*4+3=19, 5*4+0=20, 5*4+1=21, 5*4+2=22, 5*4+3=23, 6*4+0=24, 6*4+1=25, 6*4+2=26, 6*4+3=27.
    
    So V rows for the 4x3 part: [4,5,6,11,12,13,18,19,20,25,26,27]
    W rows for the 3x4 part: [16,17,18,19,20,21,22,23,24,25,26,27]
    
    U rows: i*4 + j for i in 0..3, j in 0..3 -> 0..15
    """
    import numpy as np
    
    rank = 48
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((28, rank), dtype=float)  # Full size, will place in correct rows
    W = np.zeros((28, rank), dtype=float)  # Full size, will place in correct rows
    
    # V rows for k in 4,5,6
    v_rows = []
    for j in range(4):
        for k in range(4, 7):
            v_rows.append(j * 7 + k)
    
    # W rows for k in 4,5,6
    w_rows = []
    for k in range(4, 7):
        for i in range(4):
            w_rows.append(k * 4 + i)
    
    # Schoolbook: for each (i, j, k), U[i*4+j, col] = 1, V[j*7+k, col] = 1, W[k*4+i, col] = 1
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                U[i * 4 + j, col] = 1
                V[j * 7 + k, col] = 1
                W[k * 4 + i, col] = 1
                col += 1
    
    return U.tolist(), V.tolist(), W.tolist(), 48

# EVOLVE-BLOCK-END

"""Matrix multiplication 4,5,5 with block decomposition using Strassen for the 4x4x4 core."""

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # We'll decompose into 4 blocks:
    # A is 4x5, B is 5x5
    # Split A into A1 (4x4) and A2 (4x1)
    # Split B into B11 (4x4), B12 (4x1), B21 (1x4), B22 (1x1)
    # C = A1*B11 + A1*B12 + A2*B21 + A2*B22
    # C11 (4x4) = A1*B11
    # C12 (4x1) = A1*B12
    # C21 (4x4) = A2*B21  -- wait, A2 is 4x1, B21 is 1x4, so C21 is 4x4
    # C22 (4x1) = A2*B22
    
    # Actually, let's be more careful.
    # C = A * B
    # A[i,j] for i in 0..3, j in 0..4
    # B[j,k] for j in 0..4, k in 0..4
    # C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    #        = sum_{j=0}^{3} A[i,j] * B[j,k] + A[i,4] * B[4,k]
    # Split k into 0..3 and 4:
    # C[i, 0..3] = sum_{j=0}^{3} A[i,j]*B[j,0..3] + A[i,4]*B[4,0..3]
    # C[i, 4]    = sum_{j=0}^{3} A[i,j]*B[j,4] + A[i,4]*B[4,4]
    
    # Block 1: A1 (4x4) * B11 (4x4) -> C11 (4x4)
    # Block 2: A1 (4x4) * B12 (4x1) -> C12 (4x1)
    # Block 3: A2 (4x1) * B21 (1x4) -> C21 (4x4)
    # Block 4: A2 (4x1) * B22 (1x1) -> C22 (4x1)
    
    # For each block, we need U, V, W matrices.
    # The overall U will have n*m = 20 rows, overall V will have m*p = 25 rows, overall W will have p*n = 20 rows.
    # We'll build these by placing the local block matrices in the correct positions.
    
    # Block 1: M_{4,4,4} using Strassen (rank 56)
    # Local U: 16x56, Local V: 16x56, Local W: 16x56
    # U rows: A1 indices i*4+j for i in 0..3, j in 0..3 -> global rows i*m+j = i*5+j for i in 0..3, j in 0..3
    # V rows: B11 indices j*4+k for j in 0..3, k in 0..3 -> global rows j*p+k = j*5+k for j in 0..3, k in 0..3
    # W rows: C11 indices k*4+i for k in 0..3, i in 0..3 -> global rows k*n+i = k*4+i for k in 0..3, i in 0..3
    
    # Let me build the Strassen decomposition for M_{4,4,4}
    # Strassen for 4x4: we can use the standard 7-multiply algorithm.
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For M_{4,4,4}, the tensor has rank 56.
    # We need to construct U, V, W for this.
    
    # Let's define the Strassen decomposition explicitly.
    # A is 4x4: A11 (2x2), A12 (2x2), A21 (2x2), A22 (2x2)
    # B is 4x4: B11 (2x2), B12 (2x2), B21 (2x2), B22 (2x2)
    # C is 4x4: C11 (2x2), C12 (2x2), C21 (2x2), C22 (2x2)
    
    # The 7 Strassen products are each 2x2x2 matrix multiplications (rank 8 each, but using schoolbook for 2x2x2 which is rank 8).
    # Wait, 2x2x2 schoolbook is rank 8. So 7*8 = 56. Yes.
    
    # For each Strassen product, we need to construct the U, V, W contributions.
    
    # Let me create a helper to build the schoolbook decomposition for M_{a,b,c}.
    def schoolbook_U(a, b, c):
        rank = a * b * c
        U = [[0] * rank for _ in range(a * b)]
        V = [[0] * rank for _ in range(b * c)]
        W = [[0] * rank for _ in range(c * a)]
        col = 0
        for i in range(a):
            for j in range(b):
                for k in range(c):
                    U[i * b + j][col] = 1
                    V[j * c + k][col] = 1
                    W[k * a + i][col] = 1
                    col += 1
        return U, V, W
    
    # Now for Strassen M_{4,4,4}:
    # We'll build it by combining 7 schoolbook M_{2,2,2} decompositions with appropriate row/column selections and signs.
    
    # Let's define the 7 Strassen products. Each is a 2x2x2 multiplication.
    # Product r: M_r = (sum of A blocks with coefficients) * (sum of B blocks with coefficients)
    
    # For each product r, the local 2x2x2 schoolbook gives us 8 terms.
    # We need to map these to the global 4x4x4 indices.
    
    # Let's define the blocks for A:
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    
    # For B:
    # B11: rows 0-1, cols 0-1
    # B12: rows 0-1, cols 2-3
    # B21: rows 2-3, cols 0-1
    # B22: rows 2-3, cols 2-3
    
    # For C:
    # C11: rows 0-1, cols 0-1
    # C12: rows 0-1, cols 2-3
    # C21: rows 2-3, cols 0-1
    # C22: rows 2-3, cols 2-3
    
    # Strassen formulas:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each M_r, it's a 2x2x2 multiplication. Let's denote the 2x2 blocks.
    # For M_r = X * Y where X is a 2x2 sum of A-blocks and Y is a 2x2 sum of B-blocks.
    # The schoolbook for 2x2x2 has 8 terms.
    
    # Let's build the full U, V, W for M_{4,4,4}.
    
    U_444 = [[0] * 56 for _ in range(16)]
    V_444 = [[0] * 56 for _ in range(16)]
    W_444 = [[0] * 56 for _ in range(16)]
    
    # For each Strassen product r (0-6), we add 8 terms.
    # Each term corresponds to a specific i, j, k in the 2x2x2 space.
    
    # Let's define the mapping for each product.
    # Product r has:
    # - A coefficient pattern: for each 2x2 block of A, a coefficient (+1, -1, or 0)
    # - B coefficient pattern: for each 2x2 block of B, a coefficient (+1, -1, or 0)
    # - C output: which C block(s) it contributes to, with what sign
    
    # Let me define this more carefully.
    
    # For product r, the A part is a linear combination of A11, A12, A21, A22.
    # The B part is a linear combination of B11, B12, B21, B22.
    # The result contributes to C blocks with certain signs.
    
    # Let's define for each product:
    # a_coeffs: dict from block name to coefficient
    # b_coeffs: dict from block name to coefficient
    # c_contribs: list of (c_block, sign)
    
    products = [
        # r=0: M1 = (A11+A22)*(B11+B22)
        {"a": {"11": 1, "22": 1}, "b": {"11": 1, "22": 1}, "c": [("11", 1), ("22", 1)]},
        # r=1: M2 = (A21+A22)*B11
        {"a": {"21": 1, "22": 1}, "b": {"11": 1}, "c": [("21", 1), ("22", -1)]},
        # r=2: M3 = A11*(B12-B22)
        {"a": {"11": 1}, "b": {"12": 1, "22": -1}, "c": [("12", 1), ("22", 1)]},
        # r=3: M4 = A22*(B21-B11)
        {"a": {"22": 1}, "b": {"21": 1, "11": -1}, "c": [("11", 1), ("21", 1)]},
        # r=4: M5 = (A11+A12)*B22
        {"a": {"11": 1, "12": 1}, "b": {"22": 1}, "c": [("11", -1), ("12", 1)]},
        # r=5: M6 = (A21-A11)*(B11+B12)
        {"a": {"21": 1, "11": -1}, "b": {"11": 1, "12": 1}, "c": [("22", 1)]},
        # r=6: M7 = (A12-A22)*(B21+B22)
        {"a": {"12": 1, "22": -1}, "b": {"21": 1, "22": 1}, "c": [("11", 1)]},
    ]
    
    # For each product r, and for each 2x2x2 schoolbook term (i, j, k in 0..1):
    # The A index is: for block (br, bc) in A, the local 2x2 indices (i, j) map to global A indices.
    # Wait, I need to be careful. The 2x2x2 multiplication X*Y where X is 2x2 and Y is 2x2.
    # X[i,j] * Y[j,k] -> Z[i,k]
    
    # For a given product r, X is a linear combination of A-blocks.
    # X[i,j] = sum over A-blocks (a_block, coeff) of coeff * a_block[i,j]
    # where a_block is one of A11, A12, A21, A22.
    
    # Similarly for Y.
    
    # The term X[i,j]*Y[j,k] contributes to Z[i,k].
    # Z[i,k] contributes to C-blocks according to c_contribs.
    
    # Let me build this.
    
    col_offset = 0
    for r, prod in enumerate(products):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    col = col_offset + i * 4 + j * 2 + k
                    # Wait, that's not right. Let me just use a counter.
                    pass
    
    # Let me redo this more carefully.
    U_444 = [[0] * 56 for _ in range(16)]
    V_444 = [[0] * 56 for _ in range(16)]
    W_444 = [[0] * 56 for _ in range(16)]
    
    col = 0
    for r, prod in enumerate(products):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # This is term (i,j,k) of the 2x2x2 multiplication for product r.
                    # A part: X[i,j] where X = sum of A-blocks with coefficients
                    # For each A-block (bi, bj) with coefficient ca:
                    #   The global A row is: for block (bi, bj), local (i, j) maps to global row.
                    #   A11: rows 0-1, cols 0-1 -> global row = 0*4 + 0 + ... 
                    #   Actually, for 4x4, the blocks are:
                    #   A11: rows 0-1, cols 0-1
                    #   A12: rows 0-1, cols 2-3
                    #   A21: rows 2-3, cols 0-1
                    #   A22: rows 2-3, cols 2-3
                    #   Local (i, j) in a 2x2 block maps to global:
                    #   A11: row = 0 + i, col = 0 + j
                    #   A12: row = 0 + i, col = 2 + j
                    #   A21: row = 2 + i, col = 0 + j
                    #   A22: row = 2 + i, col = 2 + j
                    
                    # B part: Y[j,k] where Y = sum of B-blocks with coefficients
                    # B11: rows 0-1, cols 0-1
                    # B12: rows 0-1, cols 2-3
                    # B21: rows 2-3, cols 0-1
                    # B22: rows 2-3, cols 2-3
                    # Local (j, k) maps to global:
                    # B11: row = 0 + j, col = 0 + k
                    # B12: row = 0 + j, col = 2 + k
                    # B21: row = 2 + j, col = 0 + k
                    # B22: row = 2 + j, col = 2 + k
                    
                    # C part: Z[i,k] contributes to C-blocks
                    # C11: rows 0-1, cols 0-1
                    # C12: rows 0-1, cols 2-3
                    # C21: rows 2-3, cols 0-1
                    # C22: rows 2-3, cols 2-3
                    # Local (i, k) maps to global:
                    # C11: row = 0 + i, col = 0 + k
                    # C12: row = 0 + i, col = 2 + k
                    # C21: row = 2 + i, col = 0 + k
                    # C22: row = 2 + i, col = 2 + k
                    
                    # For U: we need to set U[global_A_row][col] for each A-block contribution.
                    # But U has one entry per column. The column corresponds to a specific term.
                    # Each term is X[i,j] * Y[j,k].
                    # X[i,j] is a sum over A-blocks. So for each A-block (bi, bj) with coeff ca:
                    #   U[global_row_for_A_block(bi,bj,i,j)][col] += ca
                    
                    # Similarly for V: for each B-block (bi, bj) with coeff cb:
                    #   V[global_row_for_B_block(bi,bj,j,k)][col] += cb
                    
                    # For W: for each C-block (ci, cj) with coeff cc:
                    #   W[global_row_for_C_block(ci,cj,i,k)][col] += cc
                    
                    # Block to global mapping:
                    def a_block_row(block, i, j):
                        # block is like "11", "12", "21", "22"
                        br = int(block[0]) - 1
                        bc = int(block[1]) - 1
                        return (br * 2 + i) * 4 + (bc * 2 + j)  # Wait, no.
                        # For 4x4, row index = br*2 + i, col index = bc*2 + j
                        # Global row in U is row * 4 + col? No, U has n*m rows where n=4, m=4 for local.
                        # U[row, col] where row = i*4 + j for the local 4x4.
                        # Wait, I'm confusing myself. Let me restart the indexing.
                        pass
                    
                    # Let me use a cleaner approach. I'll directly construct the full 4x4x4 Strassen.
                    # Actually, let me just use schoolbook for all 4 blocks to be safe, since the rank is the same.
                    # Wait, no. The point is to use Strassen for the 4x4x4 to get rank 56 instead of 64.
                    
                    # Let me just hardcode the Strassen decomposition for 4x4x4.
                    pass
    
    # OK, this is getting complicated. Let me just use schoolbook for all 4 blocks.
    # Block 1: M_{4,4,4} schoolbook: rank 64
    # Block 2: M_{4,4,1} schoolbook: rank 16
    # Block 3: M_{4,1,4} schoolbook: rank 16
    # Block 4: M_{4,1,1} schoolbook: rank 4
    # Total rank: 64 + 16 + 16 + 4 = 100. Same as before.
    
    # Let me try a different approach. Let me use Strassen for the 4x4x4.
    # I'll build it carefully.
    
    def build_strassen_444():
        """Build U, V, W for 4x4x4 Strassen decomposition. Rank 56."""
        U = [[0] * 56 for _ in range(16)]
        V = [[0] * 56 for _ in range(16)]
        W = [[0] * 56 for _ in range(16)]
        
        # Define the 7 Strassen products.
        # Each product r has:
        # - a_map: dict from (block_row, block_col) to coefficient
        # - b_map: dict from (block_row, block_col) to coefficient  
        # - c_map: dict from (block_row, block_col) to coefficient
        # where block_row, block_col are 0 or 1.
        
        products = [
            # r=0: M1 = (A11+A22)*(B11+B22)
            {"a": {(0,0): 1, (1,1): 1}, "b": {(0,0): 1, (1,1): 1}, "c": {(0,0): 1, (1,1): 1}},
            # r=1: M2 = (A21+A22)*B11
            {"a": {(1,0): 1, (1,1): 1}, "b": {(0,0): 1}, "c": {(1,0): 1, (1,1): -1}},
            # r=2: M3 = A11*(B12-B22)
            {"a": {(0,0): 1}, "b": {(0,1): 1, (1,1): -1}, "c": {(0,1): 1, (1,1): 1}},
            # r=3: M4 = A22*(B21-B11)
            {"a": {(1,1): 1}, "b": {(1,0): 1, (0,0): -1}, "c": {(0,0): 1, (1,0): 1}},
            # r=4: M5 = (A11+A12)*B22
            {"a": {(0,0): 1, (0,1): 1}, "b": {(1,1): 1}, "c": {(0,0): -1, (0,1): 1}},
            # r=5: M6 = (A21-A11)*(B11+B12)
            {"a": {(1,0): 1, (0,0): -1}, "b": {(0,0): 1, (0,1): 1}, "c": {(1,1): 1}},
            # r=6: M7 = (A12-A22)*(B21+B22)
            {"a": {(0,1): 1, (1,1): -1}, "b": {(1,0): 1, (1,1): 1}, "c": {(0,0): 1}},
        ]
        
        col = 0
        for r in range(7):
            prod = products[r]
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        # A part: X[i,j] = sum over A-blocks
                        for (abr, abc), ca in prod["a"].items():
                            # Global A row: (abr*2 + i) * 4 + (abc*2 + j)
                            # Wait, no. For 4x4, the row index is abr*2 + i, col index is abc*2 + j.
                            # U has 16 rows, indexed by row*4 + col.
                            a_row = abr * 2 + i
                            a_col = abc * 2 + j
                            u_idx = a_row * 4 + a_col
                            U[u_idx][col] += ca
                        
                        # B part: Y[j,k] = sum over B-blocks
                        for (bbr, bbc), cb in prod["b"].items():
                            b_row = bbr * 2 + j
                            b_col = bbc * 2 + k
                            v_idx = b_row * 4 + b_col
                            V[v_idx][col] += cb
                        
                        # C part: Z[i,k] contributes to C-blocks
                        for (cbr, cbc), cc in prod["c"].items():
                            c_row = cbr * 2 + i
                            c_col = cbc * 2 + k
                            w_idx = c_row * 4 + c_col
                            W[w_idx][col] += cc
                        
                        col += 1
        
        return U, V, W
    
    # Build the 4x4x4 Strassen decomposition
    U_444, V_444, W_444 = build_strassen_444()
    
    # Now build the full 4x5x5 decomposition.
    # Block 1: A1 (4x4) * B11 (4x4) -> C11 (4x4)
    # U rows: A1 indices i*5+j for i in 0..3, j in 0..3
    # V rows: B11 indices j*5+k for j in 0..3, k in 0..3
    # W rows: C11 indices k*4+i for k in 0..3, i in 0..3
    
    # Block 2: A1 (4x4) * B12 (4x1) -> C12 (4x1)
    # Schoolbook M_{4,4,1}: rank 16
    U_441, V_441, W_441 = schoolbook_U(4, 4, 1)
    
    # Block 3: A2 (4x1) * B21 (1x4) -> C21 (4x4)
    # Schoolbook M_{4,1,4}: rank 16
    U_414, V_414, W_414 = schoolbook_U(4, 1, 4)
    
    # Block 4: A2 (4x1) * B22 (1x1) -> C22 (4x1)
    # Schoolbook M_{4,1,1}: rank 4
    U_411, V_411, W_411 = schoolbook_U(4, 1, 1)
    
    # Now assemble the global U, V, W.
    rank_444 = 56
    rank_441 = 16
    rank_414 = 16
    rank_411 = 4
    total_rank = rank_444 + rank_441 + rank_414 + rank_411
    
    U = [[0] * total_rank for _ in range(20)]
    V = [[0] * total_rank for _ in range(25)]
    W = [[0] * total_rank for _ in range(20)]
    
    # Block 1: Strassen 4x4x4
    # U_444 has 16 rows, 56 cols
    # Map local U row (i*4+j) to global row (i*5+j)
    for local_row in range(16):
        i = local_row // 4
        j = local_row % 4
        global_row = i * 5 + j
        for c in range(56):
            U[global_row][c] = U_444[local_row][c]
    
    # V_444 has 16 rows, 56 cols
    # Map local V row (j*4+k) to global row (j*5+k)
    for local_row in range(16):
        j = local_row // 4
        k = local_row % 4
        global_row = j * 5 + k
        for c in range(56):
            V[global_row][c] = V_444[local_row][c]
    
    # W_444 has 16 rows, 56 cols
    # Map local W row (k*4+i) to global row (k*4+i)  [W has p*n = 20 rows, but C is 4x5, so W row = k*4+i]
    for local_row in range(16):
        k = local_row // 4
        i = local_row % 4
        global_row = k * 4 + i
        for c in range(56):
            W[global_row][c] = W_444[local_row][c]
    
    # Block 2: Schoolbook 4x4x1
    offset = 56
    # U_441 has 16 rows, 16 cols
    # Map local U row (i*4+j) to global row (i*5+j)
    for local_row in range(16):
        i = local_row // 4
        j = local_row % 4
        global_row = i * 5 + j
        for c in range(16):
            U[global_row][offset + c] = U_441[local_row][c]
    
    # V_441 has 4 rows, 16 cols
    # Map local V row (j*1+k) to global row (j*5+4)
    for local_row in range(4):
        j = local_row
        global_row = j * 5 + 4
        for c in range(16):
            V[global_row][offset + c] = V_441[local_row][c]
    
    # W_441 has 4 rows, 16 cols
    # Map local W row (k*4+i) to global row (k*4+4)  [C col is 4]
    for local_row in range(4):
        i = local_row
        global_row = 4 * 4 + i  # k=4, so row = 4*4 + i = 16 + i
        for c in range(16):
            W[global_row][offset + c] = W_441[local_row][c]
    
    # Block 3: Schoolbook 4x1x4
    offset = 56 + 16
    # U_414 has 4 rows, 16 cols
    # Map local U row (i*1+j) to global row (i*5+4)
    for local_row in range(4):
        i = local_row
        global_row = i * 5 + 4
        for c in range(16):
            U[global_row][offset + c] = U_414[local_row][c]
    
    # V_414 has 4 rows, 16 cols
    # Map local V row (j*4+k) to global row (4*5+k)
    for local_row in range(4):
        k = local_row
        global_row = 4 * 5 + k
        for c in range(16):
            V[global_row][offset + c] = V_414[local_row][c]
    
    # W_414 has 16 rows, 16 cols
    # Map local W row (k*4+i) to global row (k*4+i)
    for local_row in range(16):
        k = local_row // 4
        i = local_row % 4
        global_row = k * 4 + i
        for c in range(16):
            W[global_row][offset + c] = W_414[local_row][c]
    
    # Block 4: Schoolbook 4x1x1
    offset = 56 + 16 + 16
    # U_411 has 4 rows, 4 cols
    for local_row in range(4):
        i = local_row
        global_row = i * 5 + 4
        for c in range(4):
            U[global_row][offset + c] = U_411[local_row][c]
    
    # V_411 has 1 row, 4 cols
    global_row = 4 * 5 + 4
    for c in range(4):
        V[global_row][offset + c] = V_411[0][c]
    
    # W_411 has 4 rows, 4 cols
    for local_row in range(4):
        i = local_row
        global_row = 4 * 4 + i
        for c in range(4):
            W[global_row][offset + c] = W_411[local_row][c]
    
    return {"U": U, "V": V, "W": W}

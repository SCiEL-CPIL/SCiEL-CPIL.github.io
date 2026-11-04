import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    """
    Rank-44 decomposition for 2x3 * 3x8 matrix multiplication.
    
    Strategy:
    - Split the inner dimension (size 3) into a 1-dimensional part and a 2-dimensional part.
    - The 2-dimensional part is handled using Strassen's algorithm for 2x2 blocks.
    - We split the 3x8 matrix B into B1 (1x8, the first row) and B2 (2x8, the last two rows).
    - We split the 2x3 matrix A into A1 (2x1, the first column) and A2 (2x2, the last two columns).
    
    The product C = A * B = A1 * B1 + A2 * B2.
    
    1. A1 * B1: (2x1) * (1x8) = 2x8.
       This is a simple outer product of A1 (length 2) and B1 (length 8).
       The rank is 2 * 8 = 16.
       The 16 terms are: A1[i] * B1[j] contributes to C[i, j].
       
    2. A2 * B2: (2x2) * (2x8).
       We can view B2 as a 2x2 block matrix with 4 blocks of 2x2: B2 = [B2_00, B2_01, B2_02, B2_03].
       Let A2 be a 2x2 matrix.
       C2 = A2 * B2 is a 2x8 matrix, which is [C2_0, C2_1, C2_2, C2_3] where each C2_i is 2x2.
       For each 2x2 block multiplication A2 * B2_ij, we use Strassen's algorithm.
       Strassen's algorithm for 2x2 * 2x2 uses 7 multiplications.
       So for 4 blocks, we have 4 * 7 = 28 multiplications.
       
    Total rank = 16 + 28 = 44.
    
    The matrices U, V, W are constructed to represent these 44 scalar multiplications.
    U has shape (2*3, 44) = (6, 44).
    V has shape (3*8, 44) = (24, 44).
    W has shape (8*2, 44) = (16, 44).
    
    Indexing conventions:
    U[i * 3 + j, c] where i in {0,1}, j in {0,1,2}, c in {0,...,43}.
    V[j * 8 + k, c] where j in {0,1,2}, k in {0,...,7}, c in {0,...,43}.
    W[k * 2 + i, c] where k in {0,...,7}, i in {0,1}, c in {0,...,43}.
    """
    n, m, p = DIMENSIONS  # 2, 3, 8
    rank = 44
    U = np.zeros((n * m, rank))  # (6, 44)
    V = np.zeros((m * p, rank))  # (24, 44)
    W = np.zeros((p * n, rank))  # (16, 44)
    
    # --- Part 1: A1 * B1 (rank 16) ---
    # A1 is the first column of A (2x1), so A[i, 0] for i in {0, 1}.
    # B1 is the first row of B (1x8), so B[0, k] for k in {0, ..., 7}.
    # The product C[i, k] gets contribution A[i, 0] * B[0, k].
    c = 0
    for i in range(n):  # i in {0, 1}
        for k in range(p):  # k in {0, ..., 7}
            U[i * m + 0, c] = 1  # A[i, 0]
            V[0 * p + k, c] = 1  # B[0, k]
            W[k * n + i, c] = 1  # C[i, k]
            c += 1
    # c is now 16.
    
    # --- Part 2: A2 * B2 (rank 28) ---
    # A2 is the submatrix A[:, 1:3] (2x2).
    # B2 is the submatrix B[1:3, :] (2x8).
    # We split B2 into 4 blocks of 2x2: B2[1:3, 0:2], B2[1:3, 2:4], B2[1:3, 4:6], B2[1:3, 6:8].
    # Let's define the 4 blocks of B2.
    # Block b (b in 0..3) corresponds to columns 2b, 2b+1 of B2, which are columns 2b, 2b+1 of B.
    # So B2_block_b is a 2x2 matrix with elements B[1, 2b], B[1, 2b+1], B[2, 2b], B[2, 2b+1].
    # A2 is a 2x2 matrix with elements A[0,1], A[0,2], A[1,1], A[1,2].
    
    # Strassen's 7 products for 2x2 * 2x2:
    # Let A2 = [[a, b], [c, d]] and B2_block = [[e, f], [g, h]].
    # The 7 products are:
    # p1 = (a + d) * (e + h)
    # p2 = (c + d) * e
    # p3 = a * (f - h)
    # p4 = d * (g - e)
    # p5 = (a + b) * h
    # p6 = (c - a) * (e + f)
    # p7 = (b - d) * (g + h)
    # The result C2_block is:
    # [[p1 + p4 - p5 + p7, p3 + p5],
    #  [p2 + p4, p1 - p2 + p3 + p6]]
    
    # We will construct U, V, W for these 7 products for each of the 4 blocks.
    # For a given block b, the elements of A2 are:
    # a = A[0, 1], b = A[0, 2], c = A[1, 1], d = A[1, 2]
    # The elements of B2_block_b are:
    # e = B[1, 2b], f = B[1, 2b+1], g = B[2, 2b], h = B[2, 2b+1]
    # The result C2_block_b is a 2x2 matrix. Its elements are:
    # C2[0, 2b] = p1 + p4 - p5 + p7
    # C2[0, 2b+1] = p3 + p5
    # C2[1, 2b] = p2 + p4
    # C2[1, 2b+1] = p1 - p2 + p3 + p6
    
    # We will add to U, V, W for each product p1..p7.
    # For each product, we need to identify the linear combination of A2 and B2_block elements.
    
    # Let's define the coefficients for the 7 products.
    # p1 = (a + d) * (e + h) -> A part: a+d, B part: e+h
    # p2 = (c + d) * e       -> A part: c+d, B part: e
    # p3 = a * (f - h)       -> A part: a, B part: f-h
    # p4 = d * (g - e)       -> A part: d, B part: g-e
    # p5 = (a + b) * h       -> A part: a+b, B part: h
    # p6 = (c - a) * (e + f) -> A part: c-a, B part: e+f
    # p7 = (b - d) * (g + h) -> A part: b-d, B part: g+h
    
    # We will iterate over the 4 blocks and the 7 products.
    for b in range(4):
        # Columns for this block in B (and C):
        col_start = 2 * b
        col_end = 2 * b + 1
        
        # Elements of B2_block_b:
        # e = B[1, col_start], f = B[1, col_end], g = B[2, col_start], h = B[2, col_end]
        # Indices in V: V[j * p + k, c]
        # e is at V[1 * p + col_start], f at V[1 * p + col_end], g at V[2 * p + col_start], h at V[2 * p + col_end]
        
        # Elements of A2:
        # a = A[0, 1], b = A[0, 2], c = A[1, 1], d = A[1, 2]
        # Indices in U: U[i * m + j, c]
        # a is at U[0 * m + 1], b at U[0 * m + 2], c at U[1 * m + 1], d at U[1 * m + 2]
        
        # The 7 products and their contributions to C2_block_b:
        # C2[0, col_start]   = p1 + p4 - p5 + p7
        # C2[0, col_end]     = p3 + p5
        # C2[1, col_start]   = p2 + p4
        # C2[1, col_end]     = p1 - p2 + p3 + p6
        
        # Let's define the products as a list of (A_coeffs, B_coeffs, C_indices, C_coeffs)
        # A_coeffs: dict mapping (i, j) to coeff
        # B_coeffs: dict mapping (j, k) to coeff
        # C_indices: list of (i, k)
        # C_coeffs: list of coeffs corresponding to C_indices
        
        products = [
            # p1
            ({(0,1): 1, (1,2): 1}, {(1, col_start): 1, (2, col_end): 1},
             [(0, col_start), (1, col_end)], [1, 1]),
            # p2
            ({(1,1): 1, (1,2): 1}, {(1, col_start): 1},
             [(1, col_start)], [1]),
            # p3
            ({(0,1): 1}, {(1, col_end): 1, (2, col_end): -1},
             [(0, col_end)], [1]),
            # p4
            ({(1,2): 1}, {(2, col_start): 1, (1, col_start): -1},
             [(0, col_start), (1, col_start)], [1, 1]),
            # p5
            ({(0,1): 1, (0,2): 1}, {(2, col_end): 1},
             [(0, col_start), (0, col_end)], [-1, 1]),
            # p6
            ({(1,1): 1, (0,1): -1}, {(1, col_start): 1, (1, col_end): 1},
             [(1, col_end)], [1]),
            # p7
            ({(0,2): 1, (1,2): -1}, {(2, col_start): 1, (2, col_end): 1},
             [(0, col_start)], [1]),
        ]
        
        for A_coeffs, B_coeffs, C_indices, C_coeffs in products:
            # Fill U
            for (i, j), coeff in A_coeffs.items():
                U[i * m + j, c] += coeff
            # Fill V
            for (j, k), coeff in B_coeffs.items():
                V[j * p + k, c] += coeff
            # Fill W
            for (i, k), coeff in zip(C_indices, C_coeffs):
                W[k * n + i, c] += coeff
            c += 1
            
    # Convert to lists of lists to match the required output format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

import time

def construct():
    """
    Returns a rank-49 tensor decomposition for 4x4 matrix multiplication
    using recursive Strassen's algorithm.
    
    The decomposition is exact and uses integer coefficients.
    """
    # Dimensions
    n, m, p = 4, 4, 4
    
    # Rank of recursive Strassen for 4x4 is 7 * 7 = 49
    rank = 49
    
    # Initialize U, V, W with zeros
    # U: (n*m, rank) -> (16, 49)
    # V: (m*p, rank) -> (16, 49)
    # W: (p*n, rank) -> (16, 49)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 decomposition definitions
    # Each term t in 0..6 defines linear combinations of sub-blocks
    # u_terms: list of (row, col, coeff) for input A (2x2)
    # v_terms: list of (row, col, coeff) for input B (2x2)
    # w_terms: list of (row, col, coeff) for output C (2x2)
    
    inner_defs = [
        # t=0: M0 = (A00+A11)(B00+B11) -> C00 += M0, C11 += M0
        {'u': [(0,0,1),(1,1,1)], 'v': [(0,0,1),(1,1,1)], 'w': [(0,0,1),(1,1,1)]},
        # t=1: M1 = (A10+A11)B00 -> C10 += M1, C11 -= M1
        {'u': [(1,0,1),(1,1,1)], 'v': [(0,0,1)], 'w': [(1,0,1),(1,1,-1)]},
        # t=2: M2 = A00(B01-B11) -> C01 += M2, C11 += M2
        {'u': [(0,0,1)], 'v': [(0,1,1),(1,1,-1)], 'w': [(0,1,1),(1,1,1)]},
        # t=3: M3 = A11(B10-B00) -> C00 += M3, C10 += M3
        {'u': [(1,1,1)], 'v': [(1,0,1),(0,0,-1)], 'w': [(0,0,1),(1,0,1)]},
        # t=4: M4 = (A00+A01)B11 -> C01 += M4, C00 -= M4
        {'u': [(0,0,1),(0,1,1)], 'v': [(1,1,1)], 'w': [(0,1,1),(0,0,-1)]},
        # t=5: M5 = (A10-A00)(B00+B01) -> C11 += M5
        {'u': [(1,0,1),(0,0,-1)], 'v': [(0,0,1),(0,1,1)], 'w': [(1,1,1)]},
        # t=6: M6 = (A01-A11)(B10+B11) -> C00 += M6
        {'u': [(0,1,1),(1,1,-1)], 'v': [(1,0,1),(1,1,1)], 'w': [(0,0,1)]},
    ]
    
    # Outer Strassen definitions for 4x4 (blocks are 2x2)
    # Block indices (P, Q) in {0,1}
    # A[P][Q] corresponds to rows 2P..2P+1, cols 2Q..2Q+1
    # B[P][Q] corresponds to rows 2P..2P+1, cols 2Q..2Q+1
    # C[P][Q] corresponds to rows 2P..2P+1, cols 2Q..2Q+1
    
    outer_defs = [
        # s=0: M0 = (A00+A11)(B00+B11) -> C00 += M0, C11 += M0
        {'a_blocks': [(0,0,1),(1,1,1)], 'b_blocks': [(0,0,1),(1,1,1)], 'c_blocks': [(0,0,1),(1,1,1)]},
        # s=1: M1 = (A10+A11)B00 -> C10 += M1, C11 -= M1
        {'a_blocks': [(1,0,1),(1,1,1)], 'b_blocks': [(0,0,1)], 'c_blocks': [(1,0,1),(1,1,-1)]},
        # s=2: M2 = A00(B01-B11) -> C01 += M2, C11 += M2
        {'a_blocks': [(0,0,1)], 'b_blocks': [(0,1,1),(1,1,-1)], 'c_blocks': [(0,1,1),(1,1,1)]},
        # s=3: M3 = A11(B10-B00) -> C00 += M3, C10 += M3
        {'a_blocks': [(1,1,1)], 'b_blocks': [(1,0,1),(0,0,-1)], 'c_blocks': [(0,0,1),(1,0,1)]},
        # s=4: M4 = (A00+A01)B11 -> C01 += M4, C00 -= M4
        {'a_blocks': [(0,0,1),(0,1,1)], 'b_blocks': [(1,1,1)], 'c_blocks': [(0,1,1),(0,0,-1)]},
        # s=5: M5 = (A10-A00)(B00+B01) -> C11 += M5
        {'a_blocks': [(1,0,1),(0,0,-1)], 'b_blocks': [(0,0,1),(0,1,1)], 'c_blocks': [(1,1,1)]},
        # s=6: M6 = (A01-A11)(B10+B11) -> C00 += M6
        {'a_blocks': [(0,1,1),(1,1,-1)], 'b_blocks': [(1,0,1),(1,1,1)], 'c_blocks': [(0,0,1)]},
    ]
    
    # Construct U, V, W
    # Indexing:
    # U[i*m + j, r] corresponds to coefficient of A[i,j] in the r-th product
    # V[j*p + k, r] corresponds to coefficient of B[j,k] in the r-th product
    # W[k*n + i, r] corresponds to coefficient of C[i,k] in the r-th product (trace convention W[k*n+i,r])
    
    for s, outer in enumerate(outer_defs):
        for t, inner in enumerate(inner_defs):
            r = s * 7 + t
            
            # Process U (Input A)
            for (P, Q, sign_a) in outer['a_blocks']:
                for (ir, jc, sign_x) in inner['u']:
                    coeff = sign_a * sign_x
                    if coeff != 0:
                        # Global row in A: 2*P + ir
                        # Global col in A: 2*Q + jc
                        row = 2 * P + ir
                        col = 2 * Q + jc
                        u_idx = row * m + col
                        U[u_idx][r] += coeff
                        
            # Process V (Input B)
            for (P, Q, sign_b) in outer['b_blocks']:
                for (ir, jc, sign_y) in inner['v']:
                    coeff = sign_b * sign_y
                    if coeff != 0:
                        # Global row in B: 2*P + ir
                        # Global col in B: 2*Q + jc
                        row = 2 * P + ir
                        col = 2 * Q + jc
                        v_idx = row * p + col
                        V[v_idx][r] += coeff
                        
            # Process W (Output C)
            for (P, Q, sign_c) in outer['c_blocks']:
                for (ir, jc, sign_z) in inner['w']:
                    coeff = sign_c * sign_z
                    if coeff != 0:
                        # Global row in C: 2*P + ir
                        # Global col in C: 2*Q + jc
                        row = 2 * P + ir
                        col = 2 * Q + jc
                        # W index is k*n + i, where k is col, i is row
                        w_idx = col * n + row
                        W[w_idx][r] += coeff

    return {"U": U, "V": V, "W": W}

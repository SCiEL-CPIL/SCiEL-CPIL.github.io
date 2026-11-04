"""Rewritten program using a block-Strassen construction for rank-minimal search."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    # Strassen rank: 7 * (2*2*3) = 84
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Strassen definitions
    # A blocks: 11 (0,0), 12 (0,1), 21 (1,0), 22 (1,1) in block coords
    # B blocks: 11 (0,0), 12 (0,1), 21 (1,0), 22 (1,1) in block coords
    # Block sizes: A is 2x2, B is 2x3
    
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Coefficients for A blocks [11, 12, 21, 22]
    # Coefficients for B blocks [11, 12, 21, 22]
    
    strassen_terms = [
        # (a_coeffs, b_coeffs)
        ([1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1]), # M3
        ([0, 0, 0, 1], [0, 0, 1, -1]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1]), # M7
    ]
    
    # Base case: 2x2 by 2x3 multiplication
    # Indices: r in 0..1 (row of X/Z), s in 0..1 (col of X / row of Y), c in 0..2 (col of Y/Z)
    # X is 2x2, Y is 2x3, Z is 2x3
    # Z[r,c] = sum_s X[r,s] Y[s,c]
    
    for a_c, b_c in strassen_terms:
        for r in range(2):
            for s in range(2):
                for c in range(3):
                    # This is a rank-1 term for the base multiplication
                    # U part: X[r,s]
                    # V part: Y[s,c]
                    # W part: Z[r,c]
                    
                    # X[r,s] is a linear combination of A blocks
                    # A block 11 covers global A[0:2, 0:2]
                    # A block 12 covers global A[0:2, 2:4]
                    # A block 21 covers global A[2:4, 0:2]
                    # A block 22 covers global A[2:4, 2:4]
                    
                    # Local indices in block:
                    # Block 11: row r, col s
                    # Block 12: row r, col s
                    # Block 21: row r, col s
                    # Block 22: row r, col s
                    
                    # Global indices for A:
                    # Block 11: i = 0 + r, j = 0 + s
                    # Block 12: i = 0 + r, j = 2 + s
                    # Block 21: i = 2 + r, j = 0 + s
                    # Block 22: i = 2 + r, j = 2 + s
                    
                    # We add a_c[0] to U[global_i, global_j]
                    # etc.
                    
                    # U contributions
                    U[(0 + r) * m + (0 + s)][col] += a_c[0]
                    U[(0 + r) * m + (2 + s)][col] += a_c[1]
                    U[(2 + r) * m + (0 + s)][col] += a_c[2]
                    U[(2 + r) * m + (2 + s)][col] += a_c[3]
                    
                    # V contributions
                    # Y is 2x3. Local s (row of Y), c (col of Y).
                    # B block 11 covers global B[0:2, 0:3]
                    # B block 12 covers global B[0:2, 3:6]
                    # B block 21 covers global B[2:4, 0:3]
                    # B block 22 covers global B[2:4, 3:6]
                    
                    # Global indices for B:
                    # Block 11: j = 0 + s, k = 0 + c
                    # Block 12: j = 0 + s, k = 3 + c
                    # Block 21: j = 2 + s, k = 0 + c
                    # Block 22: j = 2 + s, k = 3 + c
                    
                    V[(0 + s) * p + (0 + c)][col] += b_c[0]
                    V[(0 + s) * p + (3 + c)][col] += b_c[1]
                    V[(2 + s) * p + (0 + c)][col] += b_c[2]
                    V[(2 + s) * p + (3 + c)][col] += b_c[3]
                    
                    # W contributions
                    # Z is 2x3. Local r (row of Z), c (col of Z).
                    # C block 11 covers global C[0:2, 0:3]
                    # C block 12 covers global C[0:2, 3:6]
                    # C block 21 covers global C[2:4, 0:3]
                    # C block 22 covers global C[2:4, 3:6]
                    
                    # Global indices for C:
                    # Block 11: i = 0 + r, k = 0 + c
                    # Block 12: i = 0 + r, k = 3 + c
                    # Block 21: i = 2 + r, k = 0 + c
                    # Block 22: i = 2 + r, k = 3 + c
                    
                    W[(0 + c) * n + (0 + r)][col] += 1
                    W[(3 + c) * n + (0 + r)][col] += 1
                    W[(0 + c) * n + (2 + r)][col] += 1
                    W[(3 + c) * n + (2 + r)][col] += 1
                    
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

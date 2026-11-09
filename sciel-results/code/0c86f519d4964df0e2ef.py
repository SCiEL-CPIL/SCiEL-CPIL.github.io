"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Block-wise Strassen decomposition for 6x6 x 6x6 matrix multiplication.
    
    We view the 6x6 matrices as 3x3 grids of 2x2 blocks.
    Each 2x2 block multiplication uses the rank-7 Strassen algorithm.
    Total rank = 3*3*7 = 63.
    
    Block (a,b) of A (top-left block index a, right-bottom block index b)
    corresponds to rows [2a, 2a+2) and cols [2b, 2b+2) in the 6x6 matrix.
    
    For block multiplication A_block * B_block = C_block:
    A_block is at position (a, b) in A grid: rows 2a..2a+1, cols 2b..2b+1
    B_block is at position (b, c) in B grid: rows 2b..2b+1, cols 2c..2c+1
    C_block is at position (a, c) in C grid: rows 2a..2a+1, cols 2c..2c+1
    
    Strassen for 2x2:
    A = [[A11, A12], [A21, A22]]
    B = [[B11, B12], [B21, B22]]
    
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22) * B11
    M3 = A11 * (B12-B22)
    M4 = A22 * (B21-B11)
    M5 = (A11+A12) * B22
    M6 = (A21-A11) * (B11+B12)
    M7 = (A12-A22) * (B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Index mapping within a 2x2 block:
    Position (r,c) in 2x2 block -> global index (base_r + r, base_c + c)
    A11 = (0,0), A12 = (0,1), A21 = (1,0), A22 = (1,1)
    B11 = (0,0), B12 = (0,1), B21 = (1,0), B22 = (1,1)
    C11 = (0,0), C12 = (0,1), C21 = (1,0), C22 = (1,1)
    """
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Each 2x2 block has 4 elements
    # Block (a,b) in A: row_base = 2*a, col_base = 2*b
    # Element at local (dr, dc) in block (a,b): global row = 2*a+dr, col = 2*b+dc
    # U index for A element (i, j) = i*m + j = i*6 + j
    # V index for B element (j, k) = j*m + k = j*6 + k  
    # W index for C element (k, i) = k*n + i = k*6 + i (trace convention: W[k*n+i, r])
    
    rank = 63
    U = [[0] * rank for _ in range(n * m)]  # 36 x 63
    V = [[0] * rank for _ in range(m * p)]  # 36 x 63
    W = [[0] * rank for _ in range(p * n)]  # 36 x 63
    
    # Strassen coefficients: for each of the 7 terms, the coefficients for A and B sub-blocks
    # Term t uses: A coefficients (for the 4 sub-positions) and B coefficients
    # A positions: 0=A11(0,0), 1=A12(0,1), 2=A21(1,0), 3=A22(1,1)
    # B positions: 0=B11(0,0), 1=B12(0,1), 2=B21(1,0), 3=B22(1,1)
    
    strassen_A = [
        [1, 0, 0, 1],   # M1: A11 + A22
        [0, 0, 1, 1],   # M2: A21 + A22
        [1, 0, 0, 0],   # M3: A11
        [0, 0, 0, 1],   # M4: A22
        [1, 1, 0, 0],   # M5: A11 + A12
        [-1, 0, 1, 0],  # M6: A21 - A11
        [0, 1, 0, -1],  # M7: A12 - A22
    ]
    
    strassen_B = [
        [1, 0, 0, 1],   # M1: B11 + B22
        [1, 0, 0, 0],   # M2: B11
        [0, 1, 0, -1],  # M3: B12 - B22
        [0, 0, 1, -1],  # M4: B21 - B11
        [0, 0, 0, 1],   # M5: B22
        [1, 1, 0, 0],   # M6: B11 + B12
        [0, 0, 1, 1],   # M7: B21 + B22
    ]
    
    # C output positions for each term
    # C positions: 0=C11(0,0), 1=C12(0,1), 2=C21(1,0), 3=C22(1,1)
    strassen_C = [
        [1, 0, 0, 1],   # M1: C11 + C22
        [0, 0, 1, -1],  # M2: C21 - C22
        [0, 1, 0, 1],   # M3: C12 + C22
        [1, 0, 1, 0],   # M4: C11 + C21
        [-1, 1, 0, 0],  # M5: -C11 + C12
        [0, 0, 0, 1],   # M6: C22
        [1, 0, 0, 0],   # M7: C11
    ]
    
    col = 0
    for a in range(3):  # row block of A, row block of C
        for b in range(3):  # col block of A, row block of B
            for c in range(3):  # col block of B, col block of C
                for t in range(7):
                    # A block (a,b): global rows 2a..2a+1, cols 2b..2b+1
                    # For each sub-position (dr, dc) in the 2x2 block:
                    for dr in range(2):
                        for dc in range(2):
                            coeff_A = strassen_A[t][dr * 2 + dc]
                            if coeff_A != 0:
                                # Global row in A: 2*a + dr
                                # Global col in A: 2*b + dc
                                i = 2 * a + dr
                                j = 2 * b + dc
                                U[i * m + j][col] += coeff_A
                    
                    # B block (b,c): global rows 2b..2b+1, cols 2c..2c+1
                    for dr in range(2):
                        for dc in range(2):
                            coeff_B = strassen_B[t][dr * 2 + dc]
                            if coeff_B != 0:
                                # Global row in B: 2*b + dr
                                # Global col in B: 2*c + dc
                                j = 2 * b + dr
                                k = 2 * c + dc
                                V[j * p + k][col] += coeff_B
                    
                    # C block (a,c): global rows 2a..2a+1, cols 2c..2c+1
                    # W index: k*n + i where k is col of C, i is row of C
                    for dr in range(2):
                        for dc in range(2):
                            coeff_C = strassen_C[t][dr * 2 + dc]
                            if coeff_C != 0:
                                # Global row in C: 2*a + dr
                                # Global col in C: 2*c + dc
                                i = 2 * a + dr
                                k = 2 * c + dc
                                W[k * n + i][col] += coeff_C
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

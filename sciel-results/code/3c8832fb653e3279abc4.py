"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 using 8 Strassen blocks of size 2x2
    rank = 56
    
    # Initialize matrices
    # U: (n*m, rank) = (4, 56)
    # V: (m*p, rank) = (32, 56)
    # W: (p*n, rank) = (32, 56)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for A (2x2)
    # Order of A entries: 00, 01, 10, 11
    # Indices in U: i*m + j -> 0:00, 1:01, 2:10, 3:11
    
    # Strassen factors for B (2x2 block)
    # Order of B entries in block: 00, 01, 10, 11
    # Indices in V: i*p + k
    
    # Strassen factors for C (2x2 block)
    # Order of C entries in block: 00, 01, 10, 11
    # Indices in W: k*n + i -> k is col, i is row
    
    # The 7 Strassen products
    # k=0: M1 = (A00+A11)(B00+B11)
    # k=1: M2 = (A10+A11)B00
    # k=2: M3 = A00(B10-B11)
    # k=3: M4 = A11(B01-B00)
    # k=4: M5 = (A00+A01)B11
    # k=5: M6 = (A10-A00)(B00+B01)
    # k=6: M7 = (A01-A11)(B10+B11)
    
    # Coefficients for A (U)
    U_factors = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    # Coefficients for B (V) within a 2x2 block
    # B indices in block: 00, 01, 10, 11
    # We need to map these to global V indices: i*p + k
    # For block j, cols are 2j, 2j+1
    # B_00 -> V[0*p + 2j]
    # B_01 -> V[0*p + 2j+1]
    # B_10 -> V[1*p + 2j]
    # B_11 -> V[1*p + 2j+1]
    
    # Coefficients for C (W) within a 2x2 block
    # C_00 -> W[2j*n + 0]
    # C_01 -> W[(2j+1)*n + 0]
    # C_10 -> W[2j*n + 1]
    # C_11 -> W[(2j+1)*n + 1]
    
    # Contributions of M_k to C
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    C_contribs = [
        # M1
        [1, 0, 0, 1],
        # M2
        [0, 0, 1, -1],
        # M3
        [0, 1, 0, 1],
        # M4
        [1, 0, 1, 0],
        # M5
        [-1, 1, 0, 0],
        # M6
        [0, 0, 0, 1],
        # M7
        [1, 0, 0, 0]
    ]
    
    # B_contribs for each M_k
    # M1: B00 + B11
    # M2: B00
    # M3: B10 - B11
    # M4: B01 - B00
    # M5: B11
    # M6: B00 + B01
    # M7: B10 + B11
    B_contribs = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 1, -1],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    num_blocks = p // 2 # 8 blocks
    
    for j in range(num_blocks):
        col_start = 2 * j
        for k in range(7):
            c = j * 7 + k
            
            # Fill U
            for i in range(4):
                U[i][c] = U_factors[k][i]
                
            # Fill V
            # B_00 at V[0*p + col_start]
            # B_01 at V[0*p + col_start + 1]
            # B_10 at V[1*p + col_start]
            # B_11 at V[1*p + col_start + 1]
            b_coeffs = B_contribs[k]
            if b_coeffs[0] != 0: V[0 * p + col_start][c] += b_coeffs[0]
            if b_coeffs[1] != 0: V[0 * p + col_start + 1][c] += b_coeffs[1]
            if b_coeffs[2] != 0: V[1 * p + col_start][c] += b_coeffs[2]
            if b_coeffs[3] != 0: V[1 * p + col_start + 1][c] += b_coeffs[3]
            
            # Fill W
            # C_00 at W[col_start * n + 0]
            # C_01 at W[(col_start + 1) * n + 0]
            # C_10 at W[col_start * n + 1]
            # C_11 at W[(col_start + 1) * n + 1]
            c_coeffs = C_contribs[k]
            if c_coeffs[0] != 0: W[col_start * n + 0][c] += c_coeffs[0]
            if c_coeffs[1] != 0: W[(col_start + 1) * n + 0][c] += c_coeffs[1]
            if c_coeffs[2] != 0: W[col_start * n + 1][c] += c_coeffs[2]
            if c_coeffs[3] != 0: W[(col_start + 1) * n + 1][c] += c_coeffs[3]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

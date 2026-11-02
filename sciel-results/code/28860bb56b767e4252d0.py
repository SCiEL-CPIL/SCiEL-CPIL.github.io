"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-28 exact tensor decomposition for 2x4 by 4x4 matrix multiplication.
    
    Strategy: Decompose the 2x4 x 4x4 multiplication into four independent 2x2 x 2x2 multiplications,
    each solved via Strassen's algorithm (rank 7). Total rank = 4 * 7 = 28.
    
    Block structure:
      A (2x4) = [A11 A12] where A11, A12 are 2x2
      B (4x4) = [B11 B12; B21 B22] where each block is 2x2
      C (2x4) = [C11 C12] where C11, C12 are 2x2
      
      C11 = A11 * B11 + A12 * B21  -> use Strassen for both products
      C12 = A11 * B12 + A12 * B22  -> use Strassen for both products
      
    For each 2x2 multiplication, we use Strassen's 7 intermediate products.
    The global indices are:
      A: i in {0,1}, j in {0,1,2,3}
      B: j in {0,1,2,3}, k in {0,1,2,3}
      C: k in {0,1,2,3}, i in {0,1}  (trace convention W[k*n+i, r])
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Strassen's algorithm for 2x2 matrices
    # For X (2x2) times Y (2x2):
    # M1 = (X11 + X22) * (Y11 + Y22)
    # M2 = (X21 + X22) * Y11
    # M3 = X11 * (Y12 - Y22)
    # M4 = X22 * (Y21 - Y11)
    # M5 = (X11 + X12) * Y22
    # M6 = (X21 - X11) * (Y11 + Y12)
    # M7 = (X12 - X22) * (Y21 + Y22)
    #
    # Z11 = M1 + M4 - M5 + M7
    # Z12 = M3 + M5
    # Z21 = M2 + M4
    # Z22 = M1 - M2 + M3 + M6
    #
    # Each Mk is a product of a linear combination of X entries and a linear combination of Y entries.
    # In the tensor decomposition, each Mk contributes one column to U, V, W.
    # U has coefficients for X (A matrix), V has coefficients for Y (B matrix),
    # W has coefficients for Z (C matrix output).
    
    def strassen_coeffs():
        """
        Return the 7 Strassen factors as (u_coeffs, v_coeffs, w_coeffs) for each M.
        u_coeffs: dict mapping (x_row, x_col) to coefficient in X
        v_coeffs: dict mapping (y_row, y_col) to coefficient in Y
        w_coeffs: dict mapping (z_row, z_col) to coefficient in Z
        """
        factors = []
        
        # M1 = (X11 + X22) * (Y11 + Y22)
        # u: X11=1, X22=1
        # v: Y11=1, Y22=1
        # w: Z11=1, Z22=1
        factors.append(
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        )
        
        # M2 = (X21 + X22) * Y11
        # u: X21=1, X22=1
        # v: Y11=1
        # w: Z21=1, Z22=-1
        factors.append(
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(1,0): 1, (1,1): -1}
        )
        
        # M3 = X11 * (Y12 - Y22)
        # u: X11=1
        # v: Y12=1, Y22=-1
        # w: Z12=1, Z22=1
        factors.append(
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(0,1): 1, (1,1): 1}
        )
        
        # M4 = X22 * (Y21 - Y11)
        # u: X22=1
        # v: Y21=1, Y11=-1
        # w: Z11=1, Z21=1
        factors.append(
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (1,0): 1}
        )
        
        # M5 = (X11 + X12) * Y22
        # u: X11=1, X12=1
        # v: Y22=1
        # w: Z11=-1, Z12=1
        factors.append(
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): -1, (0,1): 1}
        )
        
        # M6 = (X21 - X11) * (Y11 + Y12)
        # u: X21=1, X11=-1
        # v: Y11=1, Y12=1
        # w: Z22=1
        factors.append(
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        )
        
        # M7 = (X12 - X22) * (Y21 + Y22)
        # u: X12=1, X22=-1
        # v: Y21=1, Y22=1
        # w: Z11=1
        factors.append(
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1}
        )
        
        return factors
    
    strassen_factors = strassen_coeffs()
    
    # Four block multiplications:
    # Block 0: A11 * B11 -> contributes to C11
    # Block 1: A12 * B21 -> contributes to C11
    # Block 2: A11 * B12 -> contributes to C12
    # Block 3: A12 * B22 -> contributes to C12
    
    # For each block b (0..3), we map local (x_row, x_col) to global A indices,
    # local (y_row, y_col) to global B indices, and local (z_row, z_col) to global C indices.
    
    # A is 2x4:
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # So for block 0 (A11): x_row -> i = x_row, x_col -> j = x_col
    # For block 1 (A12): x_row -> i = x_row, x_col -> j = 2 + x_col
    # For block 2 (A11): x_row -> i = x_row, x_col -> j = x_col
    # For block 3 (A12): x_row -> i = x_row, x_col -> j = 2 + x_col
    
    # B is 4x4:
    # B11: rows 0-1, cols 0-1
    # B21: rows 2-3, cols 0-1
    # B12: rows 0-1, cols 2-3
    # B22: rows 2-3, cols 2-3
    # So for block 0 (B11): y_row -> j = y_row, y_col -> k = y_col
    # For block 1 (B21): y_row -> j = 2 + y_row, y_col -> k = y_col
    # For block 2 (B12): y_row -> j = y_row, y_col -> k = 2 + y_col
    # For block 3 (B22): y_row -> j = 2 + y_row, y_col -> k = 2 + y_col
    
    # C is 2x4 (output, but in trace convention W[k*n+i]):
    # C11: cols 0-1 (k=0,1), rows 0-1 (i=0,1)
    # C12: cols 2-3 (k=2,3), rows 0-1 (i=0,1)
    # For blocks 0,1 (contribute to C11): z_row -> i = z_row, z_col -> k = z_col
    # For blocks 2,3 (contribute to C12): z_row -> i = z_row, z_col -> k = 2 + z_col
    
    block_a_col_offset = [0, 2, 0, 2]  # col offset for A block
    block_b_row_offset = [0, 2, 0, 2]  # row offset for B block
    block_b_col_offset = [0, 0, 2, 2]  # col offset for B block
    block_c_col_offset = [0, 0, 2, 2]  # col offset for C block
    
    col = 0
    for b in range(4):
        for f in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen_factors[f]
            
            # Fill U (A matrix coefficients)
            for (xr, xc), coeff in u_coeffs.items():
                i = xr
                j = xc + block_a_col_offset[b]
                U[i * m + j][col] += coeff
            
            # Fill V (B matrix coefficients)
            for (yr, yc), coeff in v_coeffs.items():
                j = yr + block_b_row_offset[b]
                k = yc + block_b_col_offset[b]
                V[j * p + k][col] += coeff
            
            # Fill W (C matrix coefficients, trace convention: W[k*n + i])
            for (zr, zc), coeff in w_coeffs.items():
                i = zr
                k = zc + block_c_col_offset[b]
                W[k * n + i][col] += coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

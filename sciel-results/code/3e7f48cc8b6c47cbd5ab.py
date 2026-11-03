"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-189 decomposition of 6x6x6 matrix multiplication using
    3x3 block structure with 2x2 Strassen decompositions.
    
    Each 2x2 block multiplication uses Strassen's 7-term algorithm.
    Total: 3*3*3 * 7 = 189 rank-1 terms.
    """
    n, m, p = DIMENSIONS  # 6, 6, 6
    block_size = 2
    num_blocks = n // block_size  # 3
    
    # Strassen coefficients for 2x2 matrix multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a)(f-h)
    # M4 = (d)(g-c)
    # M5 = (a+b)(h)
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each Strassen term t in 0..6, define:
    # U factors: which A elements contribute (with sign)
    # V factors: which B elements contribute (with sign)
    # W factors: which C elements contribute (with sign)
    
    # Strassen term definitions: (U_coeffs, V_coeffs, W_coeffs)
    # U: [coeff for A00, A01, A10, A11]
    # V: [coeff for B00, B01, B10, B11]
    # W: [coeff for C00, C01, C10, C11]
    
    strassen_terms = [
        # M1 = (a+d)(e+h)
        (
            [1, 0, 0, 1],   # U: a + d
            [1, 0, 0, 1],   # V: e + h
            [1, 0, 0, 1],   # W: contributes to C00 (+1), C11 (+1)
        ),
        # M2 = (c+d)(e)
        (
            [0, 0, 1, 1],   # U: c + d
            [1, 0, 0, 0],   # V: e
            [0, 0, 1, -1],  # W: C10 (+1), C11 (-1)
        ),
        # M3 = (a)(f-h)
        (
            [1, 0, 0, 0],   # U: a
            [0, 1, 0, -1],  # V: f - h
            [0, 1, 0, 1],   # W: C01 (+1), C11 (+1)
        ),
        # M4 = (d)(g-c)
        (
            [0, 0, 0, 1],   # U: d
            [0, 0, 1, -1],  # V: g - c
            [1, 0, 1, 0],   # W: C00 (+1), C10 (+1)
        ),
        # M5 = (a+b)(h)
        (
            [1, 1, 0, 0],   # U: a + b
            [0, 0, 0, 1],   # V: h
            [-1, 1, 0, 0],  # W: C00 (-1), C01 (+1)
        ),
        # M6 = (c-a)(e+f)
        (
            [0, 0, 1, -1],  # U: c - a
            [1, 1, 0, 0],   # V: e + f
            [0, 0, 0, 1],   # W: C11 (+1)
        ),
        # M7 = (b-d)(g+h)
        (
            [0, 1, 0, -1],  # U: b - d
            [0, 0, 1, 1],   # V: g + h
            [1, 0, 0, 0],   # W: C00 (+1)
        ),
    ]
    
    rank = num_blocks ** 3 * 7  # 27 * 7 = 189
    U = [[0] * rank for _ in range(n * m)]  # 36 x 189
    V = [[0] * rank for _ in range(m * p)]  # 36 x 189
    W = [[0] * rank for _ in range(p * n)]  # 36 x 189
    
    col = 0
    for i in range(num_blocks):       # block row of A
        for k in range(num_blocks):   # block col of A / block row of B
            for j in range(num_blocks):  # block col of B
                # Block A_ik is at rows [i*2:(i+1)*2], cols [k*2:(k+1)*2]
                # Block B_kj is at rows [k*2:(k+1)*2], cols [j*2:(j+1)*2]
                # Block C_ij is at rows [i*2:(i+1)*2], cols [j*2:(j+1)*2]
                
                for t in range(7):
                    U_coeffs, V_coeffs, W_coeffs = strassen_terms[t]
                    
                    # Fill U: for each element in A_ik
                    for a_row in range(2):
                        for a_col in range(2):
                            U_idx = (i * block_size + a_row) * m + (k * block_size + a_col)
                            U[U_idx][col] = U_coeffs[a_row * 2 + a_col]
                    
                    # Fill V: for each element in B_kj
                    for b_row in range(2):
                        for b_col in range(2):
                            V_idx = (k * block_size + b_row) * p + (j * block_size + b_col)
                            V[V_idx][col] = V_coeffs[b_row * 2 + b_col]
                    
                    # Fill W: for each element in C_ij
                    # W[k*n+i, c] convention: W index = output_col * n + output_row
                    # C_ij is at rows [i*2:(i+1)*2], cols [j*2:(j+1)*2]
                    for c_row in range(2):
                        for c_col in range(2):
                            # W index: col * n + row = c_col * 6 + c_row
                            W_idx = (j * block_size + c_col) * n + (i * block_size + c_row)
                            W[W_idx][col] = W_coeffs[c_row * 2 + c_col]
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

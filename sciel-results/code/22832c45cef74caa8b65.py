import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will construct the 49-term Strassen decomposition
    # and then check if we can reduce it to 48.
    
    # Block indices: 00, 01, 10, 11
    # Block (bi, bj) corresponds to rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
    
    # Strassen 7 terms for 2x2 block multiplication
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Coefficients for A blocks: [c00, c01, c10, c11]
    A_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    # Coefficients for B blocks: [c00, c01, c10, c11]
    B_coeffs = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [0, 0, 1, -1], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # Output block contributions for each M
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # Coefficients [c00, c01, c10, c11]
    C_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    # Strassen 7 terms for 2x2 scalar multiplication
    # P1 = (a00 + a11)(b00 + b11)
    # P2 = (a10 + a11)b00
    # P3 = a00(b01 - b11)
    # P4 = a11(b10 - b00)
    # P5 = (a00 + a01)b11
    # P6 = (a10 - a00)(b00 + b01)
    # P7 = (a01 - a11)(b10 + b11)
    
    # Coefficients for a (2x2): [a00, a01, a10, a11]
    a_coeffs = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, 1], # P2
        [1, 0, 0, 0], # P3
        [0, 0, 0, 1], # P4
        [1, 1, 0, 0], # P5
        [-1, 0, 1, 0], # P6
        [0, 1, 0, -1]  # P7
    ]
    
    # Coefficients for b (2x2): [b00, b01, b10, b11]
    b_coeffs = [
        [1, 0, 0, 1], # P1
        [1, 0, 0, 0], # P2
        [0, 1, 0, -1], # P3
        [0, 0, 1, -1], # P4
        [0, 0, 0, 1], # P5
        [1, 1, 0, 0], # P6
        [0, 0, 1, 1]  # P7
    ]
    
    # Output contributions for P in 2x2 result
    # c00 = P1 + P4 - P5 + P7
    # c01 = P3 + P5
    # c10 = P2 + P4
    # c11 = P1 - P2 + P3 + P6
    # Coefficients [c00, c01, c10, c11]
    c_coeffs = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, -1], # P2
        [0, 1, 0, 1], # P3
        [1, 0, 1, 0], # P4
        [-1, 1, 0, 0], # P5
        [0, 0, 0, 1], # P6
        [1, 0, 0, 0]  # P7
    ]
    
    U = [[0] * 49 for _ in range(16)]
    V = [[0] * 49 for _ in range(16)]
    W = [[0] * 49 for _ in range(16)]
    
    col = 0
    for m_idx in range(7):
        for n_idx in range(7):
            # Get block-level coefficients
            a_block_coeffs = A_coeffs[m_idx]
            b_block_coeffs = B_coeffs[m_idx]
            c_block_coeffs = C_coeffs[m_idx]
            
            # Get scalar-level coefficients
            a_scalar_coeffs = a_coeffs[n_idx]
            b_scalar_coeffs = b_coeffs[n_idx]
            c_scalar_coeffs = c_coeffs[n_idx]
            
            # Construct U vector (16 elements)
            for bi in range(2):
                for bj in range(2):
                    # Block (bi, bj)
                    # a_block_coeffs index: bi*2 + bj
                    block_coeff_a = a_block_coeffs[bi * 2 + bj]
                    if block_coeff_a != 0:
                        for si in range(2):
                            for sj in range(2):
                                # Scalar (si, sj) within block
                                # a_scalar_coeffs index: si*2 + sj
                                scalar_coeff_a = a_scalar_coeffs[si * 2 + sj]
                                if scalar_coeff_a != 0:
                                    # Global index in A
                                    i = 2 * bi + si
                                    j = 2 * bj + sj
                                    U[i * 4 + j][col] += block_coeff_a * scalar_coeff_a
                                    
            # Construct V vector (16 elements)
            for bi in range(2):
                for bj in range(2):
                    block_coeff_b = b_block_coeffs[bi * 2 + bj]
                    if block_coeff_b != 0:
                        for si in range(2):
                            for sj in range(2):
                                scalar_coeff_b = b_scalar_coeffs[si * 2 + sj]
                                if scalar_coeff_b != 0:
                                    j = 2 * bi + si
                                    k = 2 * bj + sj
                                    V[j * 4 + k][col] += block_coeff_b * scalar_coeff_b
                                    
            # Construct W vector (16 elements)
            for bi in range(2):
                for bj in range(2):
                    block_coeff_c = c_block_coeffs[bi * 2 + bj]
                    if block_coeff_c != 0:
                        for si in range(2):
                            for sj in range(2):
                                scalar_coeff_c = c_scalar_coeffs[si * 2 + sj]
                                if scalar_coeff_c != 0:
                                    k = 2 * bi + si
                                    i = 2 * bj + sj
                                    # W index is k*n + i = k*4 + i
                                    W[k * 4 + i][col] += block_coeff_c * scalar_coeff_c
                                    
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

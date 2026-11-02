"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    # Strategy: Apply rank-7 Strassen decomposition to each 2x2 block of the 2x10 matrix.
    # n=2, m=2, p=10.
    # A is 2x2. B is 2x10. C is 2x10.
    # We split B into 5 blocks of width 2: B = [B0, B1, B2, B3, B4] where Bi is 2x2.
    # C = A * B = [A*B0, A*B1, ..., A*B4].
    # For each block i, we use Strassen's algorithm (rank 7) to compute A*Bi.
    # This gives a total rank of 5 * 7 = 35.
    # The task requires exact decomposition. We construct U, V, W matrices.
    
    # Dimensions
    n, m, p = DIMENSIONS
    # U shape: (n*m, rank) -> (4, 35)
    # V shape: (m*p, rank) -> (20, 35)
    # W shape: (p*n, rank) -> (20, 35)
    
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 terms for multiplying two 2x2 matrices A and B:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a + d) * (e + h)
    # M2 = (c + d) * e
    # M3 = a * (g + h)
    # M4 = d * (h - g)
    # M5 = (a - b) * (f + h)
    # M6 = (c + d) * (g - f)
    # M7 = (a + b - e - d) * f
    
    # Result C:
    # C00 = M1 - M5 + M4 - M7
    # C01 = M2 + M5
    # C10 = M3 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indices for U (A matrix elements):
    # a -> U[0]
    # b -> U[1]
    # c -> U[2]
    # d -> U[3]
    
    # Indices for V (B matrix elements):
    # For block i, B is 2x2 with elements:
    # e -> V[2*i*2 + 0] -> V[4*i]
    # f -> V[2*i*2 + 1] -> V[4*i + 1]
    # g -> V[2*i*2 + 2] -> V[4*i + 2]
    # h -> V[2*i*2 + 3] -> V[4*i + 3]
    
    # Indices for W (C matrix elements):
    # The output factor follows the official trace convention W[k*n+i, r].
    # Here n=2, p=10. The C matrix is 2x10 (p x n).
    # Row index in W is k*n + i, where k is column index in C (0..9), i is row index in C (0..1).
    # So for C[i, k], the index in W is k*n + i = 2*k + i.
    # For block i (which corresponds to columns 2*i and 2*i+1 in B and C):
    # C00 (row 0, col 2*i) -> W[2*(2*i) + 0] = W[4*i]
    # C01 (row 0, col 2*i+1) -> W[2*(2*i+1) + 0] = W[4*i + 2]
    # C10 (row 1, col 2*i) -> W[2*(2*i) + 1] = W[4*i + 1]
    # C11 (row 1, col 2*i+1) -> W[2*(2*i+1) + 1] = W[4*i + 3]
    
    # Strassen coefficients for U, V, W:
    # term: (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: [a, b, c, d]
    # V_coeffs: [e, f, g, h]
    # W_coeffs: [C00, C01, C10, C11]
    
    strassen_terms = [
        # M1 = (a+d)(e+h)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2 = (c+d)e
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]),
        # M3 = a(g+h)
        ([1, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 0]),
        # M4 = d(h-g)
        ([0, 0, 0, 1], [0, 0, -1, 1], [1, 0, 1, 0]),
        # M5 = (a-b)(f+h)
        ([1, -1, 0, 0], [0, 1, 0, 1], [-1, 1, 0, 0]),
        # M6 = (c+d)(g-f)
        ([0, 0, 1, 1], [0, -1, 1, 0], [0, 0, 0, 1]),
        # M7 = (a+b-e-d)f
        ([1, 1, 0, 0], [0, 1, 0, 0], [-1, 0, 0, 0]),
    ]
    
    # Fill U, V, W
    for block in range(5):
        base_v = block * 4  # 4 elements per block in V
        base_w = block * 4  # 4 elements per block in W
        for t in range(7):
            col = block * 7 + t
            u_coeff, v_coeff, w_coeff = strassen_terms[t]
            for i in range(4):
                U[i][col] = u_coeff[i]
                V[base_v + i][col] = v_coeff[i]
                W[base_w + i][col] = w_coeff[i]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
